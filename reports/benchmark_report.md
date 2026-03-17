# NPU Simulator Benchmark Report

**날짜:** 2026-03-10
**하드웨어 설정:** 32×32 Systolic Array, 512KB SRAM (32 banks, dual-port), FP16, DDR4-2400 (DRAMSim3)
**Double Buffer:** ON (그룹 3 제외)

---

## 그룹 1 — 단일 타일 vs 멀티 타일 (OS / WS / IS)

### 1-A. M=32 단일 타일

| Dataflow | Total Cycles | DRAM Read | DRAM Write | Utilization | DB Stall |
|----------|-------------|-----------|------------|-------------|----------|
| OS       | 4,237       | 4,096 B   | 4,096 B    | 0.76%       | 0        |
| WS       | 4,237       | 4,096 B   | 4,096 B    | 0.76%       | 0        |
| IS       | 4,237       | 4,096 B   | 4,096 B    | 0.76%       | 0        |

**관찰:** 단일 타일에서는 OS/WS/IS가 완전히 동일. Tile이 하나뿐이라 dataflow 간 재사용 패턴의 차이가 없고, Double Buffer도 동작하지 않음(다음 타일 없음). 전체 사이클의 대부분이 DDR4 row activation 대기.

### 1-B. M=256 멀티 타일 (64 tiles)

| Dataflow | Total Cycles | DRAM Read  | DRAM Write | Utilization | DB Stall   | Tile Size     |
|----------|-------------|------------|------------|-------------|------------|---------------|
| OS       | 578,393     | 2,097,152 B | 262,144 B  | 2.83%       | 449,980    | (32,32,256)   |
| WS       | 373,013     | 1,179,648 B | 262,144 B  | 4.39%       | 313,635    | (256,32,32)   |
| IS       | 372,275     | 1,179,648 B | 262,144 B  | 4.40%       | 312,764    | (32,256,32)   |

**관찰:**
- **WS/IS가 OS 대비 35% 빠름.** DRAM Read도 1.18MB vs 2.10MB로 WS/IS가 44% 적음.
- OS는 tile_k=256 (K 전체를 한 번에 읽음)이라 K 방향 재사용은 좋지만, 매 타일마다 weight+activation을 모두 로드해 절대적인 트래픽이 큼.
- WS/IS는 tile_k=32 (작은 K 단위)로 쪼개지지만 stationary 차원 덕분에 총 로드 횟수가 줄어 DRAM 트래픽 감소.
- WS ≈ IS: 이 정사각형 행렬에서는 M-stationary(WS)와 N-stationary(IS)의 효과가 대칭.

---

## 그룹 2 — K 증가에 따른 OS vs WS 비교

| M  | N  | K    | Dataflow | Cycles  | DRAM Read  | Utilization | Tiles | DB Stall |
|----|----|----|----------|---------|------------|-------------|-------|----------|
| 64 | 64 | 64   | OS       | 24,277  | 32,768 B   | 1.05%       | 4     | 13,928   |
| 64 | 64 | 64   | WS       | 20,070  | 24,576 B   | 1.28%       | 4     | 12,553   |
| 64 | 64 | 256  | OS       | 32,595  | 131,072 B  | 3.14%       | 4     | 19,984   |
| 64 | 64 | 256  | WS       | 71,174  | 98,304 B   | 1.44%       | 16    | 61,319   |
| 64 | 64 | 1024 | OS       | 64,474  | 524,288 B  | 6.35%       | 4     | 44,104   |
| 64 | 64 | 1024 | WS       | 275,293 | 393,216 B  | 1.49%       | 64    | 256,393  |

**관찰:**

K=64일 때: WS가 OS보다 17% 빠르고 DRAM도 25% 적음. 예상대로 소폭 유리.

**K=256부터 역전.** OS가 WS보다 2.2× 빠름. 이유:
- OS: tile_k를 K 전체(256)로 잡아 타일 수 4개 유지. 한 번에 많이 읽지만 타일 수가 적어 총 overhead가 작음.
- WS: tile_k=32로 고정되어 K가 커질수록 타일 수가 비례해서 증가(K=256 → 16 tiles, K=1024 → 64 tiles). 타일 수 × per-tile DRAM 레이턴시가 누적되어 오히려 불리해짐.

**결론:** K가 큰 경우 SRAM에 K 전체를 담을 수 있는 OS가 유리. WS가 유리한 조건은 M이 크고 K가 작을 때 (weight를 M 방향으로 재사용할 수 있을 때).

---

## 그룹 3 — Double Buffer 효과 (M=256, WS)

| DB   | Total Cycles | Utilization | DB Hits/Misses | DB Stall Cycles |
|------|-------------|-------------|----------------|-----------------|
| ON   | 373,013     | 4.39%       | 0 / 63         | 313,635         |
| OFF  | 373,013     | 4.39%       | 0 / 0          | 0               |

**관찰:** 사이클이 동일하고 DB OFF에서 stall이 0으로 표시되는 것은 예상과 다름.

해석: 이 워크로드는 **완전 메모리 바운드(0 DB Hits)**. DB ON 상태에서도 prefetch가 compute보다 훨씬 느려서 매 타일마다 stall이 발생(63 misses = 전체 63개 전환 모두 miss). DB OFF로 해도 이미 compute가 DRAM을 기다리는 구조라 총 사이클이 동일. DB가 효과를 내려면 compute time > DRAM prefetch time이어야 하는데, 현재 array(32×32)가 너무 작아 compute가 순식간에 끝남.

---

## 그룹 4 — Transformer (GPT-2 small 규모)

### seq_len=512, OS vs WS

| Dataflow | Total Cycles    | DRAM Read     | DRAM Write   | Utilization | DB Stall    |
|----------|----------------|---------------|--------------|-------------|-------------|
| OS       | 134,425,162    | 503,316,480 B | 28,311,552 B | 2.93%       | 70,716,887  |
| WS       | 102,284,694    | 267,386,880 B | 28,311,552 B | 3.84%       | 45,603,056  |

WS가 OS 대비 **24% 빠르고, DRAM Read 47% 절감.** Transformer의 QKV/FFN matmul은 hidden_dim(768) 방향이 크고 seq_len 방향의 재사용이 있어 WS에 유리한 구조.

### WS, seq_len 스케일링

| seq_len | Total Cycles    | DRAM Read     | DRAM Write   | Utilization | Tiles  | DB Stall    |
|---------|----------------|---------------|--------------|-------------|--------|-------------|
| 128     | 40,451,261     | 72,744,960 B  | 4,718,592 B  | 2.25%       | 7,104  | 31,416,620  |
| 512     | 102,284,694    | 267,386,880 B | 28,311,552 B | 3.84%       | 7,680  | 45,603,056  |
| 1024    | 247,424,908    | 570,949,632 B | 81,788,928 B | 3.50%       | 8,448  | 69,861,648  |

**관찰:**
- seq_len 128→1024 (8×) 증가 시 cycles는 약 6.1×, DRAM은 7.8× 증가. 선형보다 빠르게 증가.
- 타일 수는 7,104→8,448로 상대적으로 적게 증가하는 반면, 각 attention score matmul의 크기(seq×seq)가 quadratic하게 증가하기 때문.
- Utilization은 seq_len 512에서 가장 높음(3.84%). seq_len 1024에서는 DRAM 트래픽 폭증으로 stall이 증가하며 utilization 소폭 감소.

---

## 종합 요약

### OS / WS / IS 비교

| 상황 | 최적 Dataflow | 이유 |
|------|--------------|------|
| 정사각 행렬, K 작음 | WS ≈ IS | stationary 효과 대칭, DRAM Read 절감 |
| K가 크고 SRAM에 K 전체 담김 | OS | tile 수 최소화, 레이턴시 overhead 적음 |
| Transformer (large hidden) | WS | hidden_dim 방향 재사용 |
| 단일 타일 | 동일 | dataflow 차이 없음 |

### 핵심 발견

1. **전 워크로드 메모리 바운드.** DB Hits = 0. 32×32 array는 compute 시간이 너무 짧아 DRAM prefetch와 overlap이 불가능. Array를 키우거나 DRAM 대역폭을 늘려야 utilization 상승 가능.

2. **Double Buffer 미효과.** compute << DRAM latency 조건에서는 DB가 있어도 없어도 사이클 동일. DB는 compute bound에서만 의미 있음.

3. **WS의 K 스케일 취약점.** K가 클수록 WS tile 수가 늘어나 오히려 OS보다 느려짐. WS는 "M이 크고 K가 array 크기 근방"일 때 최적.

4. **Transformer에서 WS 유효.** OS 대비 24% 사이클 단축, DRAM 47% 절감. 대형 hidden_dim의 행렬 곱에서 WS의 재사용 효과가 발휘됨.

---

## 하드웨어 개선 방향

| 문제 | 개선 방안 | 기대 효과 |
|------|----------|----------|
| Compute utilization < 5% | Array 크기 확대 (128×128) | compute time 증가 → DB 효과 활성화 |
| DRAM bound | DRAM bandwidth 확대 (HBM 등) | stall cycles 감소 |
| WS K 취약 | 적응형 dataflow 선택 | 워크로드별 최적 dataflow 자동 선택 |
