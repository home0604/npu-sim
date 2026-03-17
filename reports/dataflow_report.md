# Dataflow Simulation Report

**Date:** 2026-03-17
**Simulator:** npu-sim (cycle formula updated: OS=SR+SC+T-2, WS/IS=2·SR+SC+T-2)
**Config:** 32×32 systolic array, 512KB SRAM, FP16, DRAMSim3, Double Buffer ON

---

## Group 1: Small vs Large Matrix — OS / WS / IS 비교

### M=32, N=32, K=32 (단일 타일, array에 딱 맞음)

| Dataflow | Total Cycles | Tile Size (M,N,K) | DRAM Read | DB Stall |
|---|---|---|---|---|
| OS | 4,237 | (32, 32, 32) | 4 KB | 0 |
| WS | 4,237 | (32, 32, 32) | 4 KB | 0 |
| IS | 4,237 | (32, 32, 32) | 4 KB | 0 |

**관찰:** 단일 타일(행렬이 array에 딱 맞음)에서는 모든 dataflow가 동일. DRAM 접근도 동일하고 stall도 없음. 타일이 하나뿐이라 루프 순서 차이가 의미 없음.

### M=256, N=256, K=256 (멀티 타일)

| Dataflow | Total Cycles | Tile Size (M,N,K) | DRAM Read | Util | DB Stall |
|---|---|---|---|---|---|
| OS | 578,393 | (32, 32, 256) | 2,048 KB | 2.83% | 449,980 |
| WS | 373,013 | (256, 32, 32) | 1,152 KB | 4.39% | 315,427 |
| IS | 372,275 | (32, 256, 32) | 1,152 KB | 4.40% | 314,556 |

**관찰:**
- WS/IS가 OS 대비 약 **35% 사이클 절감** (373K vs 578K)
- OS는 tile_k=256으로 K를 최대화하지만 weight+activation을 매 타일마다 재로드 → DRAM 트래픽 가장 큼
- WS/IS는 tile_m 또는 tile_n을 최대화하여 재사용 → DRAM 트래픽 44% 절감 (2048KB → 1152KB)
- WS와 IS 사이클이 거의 동일 (대칭적 구조)

---

## Group 2: K 차원 스케일 추세 (M=64, N=64)

| K | Dataflow | Total Cycles | Tile Size | DRAM Read | Util |
|---|---|---|---|---|---|
| 64 | OS | 24,277 | (32, 32, 64) | 32 KB | 1.05% |
| 64 | WS | 20,070 | (64, 32, 32) | 24 KB | 1.28% |
| 256 | OS | 32,595 | (32, 32, 256) | 128 KB | 3.14% |
| 256 | WS | 71,174 | (64, 32, 32) | 96 KB | 1.44% |
| 1024 | OS | 64,474 | (32, 32, 1024) | 512 KB | 6.35% |
| 1024 | WS | 275,293 | (64, 32, 32) | 384 KB | 1.49% |

**관찰:**
- **K가 클수록 OS가 유리**: K=64에서 WS가 약간 빠르지만, K=256부터 OS가 역전, K=1024에서 OS가 **4.3배 빠름**
- **이유**: OS는 tile_k를 K 전체로 최대화 (단일 타일, 루프 1회). WS는 tile_k=32 고정이라 K가 커질수록 루프 횟수 증가 (K=1024 → 32 루프)
- **K가 클수록 OS utilization 향상**: tile당 compute 사이클(T=K)이 커지므로 DRAM 대기 비율 감소
- WS는 K가 커져도 tile_k=32 고정이라 utilization 개선 없음

---

## Group 3: Double Buffer 효과 (M=256, N=256, K=256, WS)

| DB | Total Cycles | DRAM Read | DB Hits/Misses | Stall Cycles |
|---|---|---|---|---|
| ON | 373,013 | 1,152 KB | 0/63 | 315,427 |
| OFF | 373,013 | 1,152 KB | 0/0 | 0 |

**관찰:**
- 사이클이 동일하게 나옴 → 현재 workload는 **완전 memory-bound**
- DB ON이어도 모든 타일에서 prefetch miss (0 hits) → double buffer가 stall을 숨기지 못함
- DB OFF의 stall=0은 stall 추적을 안 하는 것이지 실제로 stall이 없는 게 아님
- compute 시간 < DRAM prefetch 시간이므로 double buffer 효과 없음

---

## Group 4: Transformer 레이어 (seq-len 스케일, hidden=768, heads=12)

| seq-len | Dataflow | Total Cycles | Tile Size | DRAM Read | Util | DB Stall |
|---|---|---|---|---|---|---|
| 128 | WS | 40,451,261 | (128, 32, 32) | 71 MB | 2.25% | 31,634,732 |
| 512 | OS | 134,425,162 | (32, 32, 1228) | 480 MB | 2.93% | 70,766,039 |
| 512 | WS | 102,284,694 | (512, 32, 32) | 255 MB | 3.84% | 45,834,992 |
| 1024 | WS | 247,424,908 | (1024, 32, 32) | 545 MB | 3.50% | 70,112,016 |

**관찰:**
- seq-len=512에서 WS가 OS 대비 **24% 빠름** (102M vs 134M cycles)
- seq-len이 클수록 DRAM 트래픽이 선형 증가, 사이클도 거의 선형
- seq-len=128 대비 512는 약 2.5배 사이클 (DRAM 트래픽 3.6배) → DRAM이 지배적
- Transformer는 attention의 Q·K^T, V 연산 등 다수의 GEMM을 포함하므로 DRAM 트래픽이 매우 큼
- Utilization이 2~4%로 낮음 → 전체적으로 심각한 memory-bound 상태

---

## 종합 요약

| 상황 | 유리한 Dataflow | 이유 |
|---|---|---|
| K가 작고 M/N이 큰 경우 | WS / IS | activation 재사용으로 DRAM 절감 |
| K가 매우 큰 경우 | OS | tile_k 최대화로 루프 수 최소화 |
| 단일 타일 (행렬 ≤ array) | 무관 | 차이 없음 |
| Transformer/LLM workload | WS | 큰 M 재사용으로 DRAM 절감 |

**전체 경향:** 모든 workload가 memory-bound. DRAM bandwidth가 병목이므로 DRAM 접근을 최소화하는 dataflow 선택이 핵심. Compute utilization은 최대 6%에 불과하여 실제 하드웨어 활용도는 낮음.
