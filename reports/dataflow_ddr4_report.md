# Dataflow Benchmark Report (DDR4 8Gb x16 2400)

**DRAM 설정**: `DDR4_8Gb_x16_2400.ini` (대역폭 ~25.6 GB/s 이론치, DDR4-2400 단일 채널)
**비교 대상**: 이전 HBM2 결과 (`dataflow_tick_report.md`)

---

## Group 1: M=32 vs M=256, OS/WS/IS 비교

| Dataflow | M=32 cycles | M=256 cycles | Tile Size (M,N,K) | DB Hits/Misses |
|---|---|---|---|---|
| OS | 686 | 236,628 | (32,32,256) | 0/63 |
| WS | 718 | 133,664 | (256,32,32) | 0/63 |
| IS | 718 | 136,191 | (32,256,32) | 0/63 |

**분석**

- M=32: tile 1개, DB 없음. HBM2 대비 큰 단순 latency 차이 (686 vs 273)
- M=256: **전체 DB 0 hits** → DDR4가 memory-bound 심화, 어떤 dataflow도 hiding 불가
  - HBM2: WS 82% Hit Rate → DDR4: WS 0% Hit Rate
  - DDR4 단일 채널 대역폭이 HBM2(460 GB/s)보다 월등히 낮아 prefetch 시간이 compute 시간보다 크게 늦어짐
- OS(236,628) > WS(133,664) ≈ IS(136,191): 여전히 WS/IS가 OS보다 유리 (weight 재로드 감소)
  - WS가 OS보다 **44% 빠름** (HBM2에서는 33% 빠름)

---

## Group 2: K 스케일링 (M=N=64, K=64/256/1024), OS vs WS

| K | OS cycles | WS cycles | OS tile_k | WS tile_k | WS DB Hits/Misses |
|---|---|---|---|---|---|
| 64 | 5,275 | 4,162 | 64 | 32 | 0/3 |
| 256 | 14,608 | 12,306 | 256 | 32 | 0/15 |
| 1024 | 48,980 | 45,748 | 1024 | 32 | 0/63 |

**분석**

- HBM2와 달리 **모든 케이스에서 DB 0 hits** → DDR4 대역폭 부족으로 prefetch가 compute를 따라가지 못함
- WS가 OS보다 여전히 빠름 (weight DRAM traffic 감소 효과):
  - K=64: WS(4,162) < OS(5,275), **21% 빠름** (HBM2에서는 19% 빠름)
  - K=256: WS(12,306) < OS(14,608), **16% 빠름** (HBM2: WS가 40% **느렸음**)
  - K=1024: WS(45,748) < OS(48,980), **7% 빠름** (HBM2: WS가 63% **느렸음**)
- HBM2와 결과 역전 이유: HBM2에서는 K가 클수록 WS tile_k=32 고정으로 activation load 반복이 문제였으나, DDR4에서는 모든 경우가 stall-dominant이므로 weight traffic 감소 효과가 더 dominant

---

## Group 3: Double Buffer ON vs OFF (WS, M=N=K=256)

| 설정 | Total Cycles | DB Hits/Misses | Stall Cycles |
|---|---|---|---|
| DB ON | 133,664 | 0/63 | 89,170 |
| DB OFF | 152,384 | 0/0 | 0 |
| **차이** | **-18,720 (-12%)** | — | — |

**분석**

- DB ON이 12% 빠름 (HBM2: 28% 빠름)
- **DB Hit Rate = 0%** 임에도 DB ON이 빠른 이유:
  - Hit=0이지만 stall 89,170 cycles 발생 → DB ON은 stall 중에도 prefetch를 병렬로 진행
  - DB OFF는 compute 완료 후 순차적으로 load → latency 완전 직렬화
  - 즉 DB ON에서 "Miss"는 compute보다 prefetch가 늦었지만 일부 overlap은 존재
- DDR4 latency(tRCD+tRP ≈ 30ns, 30cycles@1GHz)가 높아 HBM2보다 prefetch가 더 느리고 hiding 효과 감소

---

## Group 4: Transformer (seq=128/512/1024, WS)

| seq_len | Total Cycles | Tile Size | DB Hits/Misses | Stall Cycles | Hit Rate |
|---|---|---|---|---|---|
| 128 | 14,049,943 | (128,32,32) | 0/7076 | 5,784,128 | **0%** |
| 512 | 74,543,976 | (512,32,32) | 0/7652 | 18,649,577 | **0%** |
| 1024 | 215,777,213 | (1024,32,32) | 0/8420 | 38,849,960 | **0%** |

**참고**: seq=512 OS = 97,716,024 cycles, WS(74,543,976)가 OS 대비 **24% 빠름** (HBM2: 5% 빠름)

**분석**

- **전 케이스 Hit Rate 0%**: DDR4 대역폭이 transformer workload(대형 weight tile)의 prefetch를 compute 내에 완료 불가
- seq=1024 총 cycle이 seq=512의 2.9배: tile 수 비례 + stall 누적
- WS가 OS보다 Transformer에서 훨씬 더 유리: 큰 seq로 인한 weight tile 크기 증가가 DDR4에서 weight reuse 이점 극대화

---

## HBM2 vs DDR4 비교 (주요 케이스)

| 케이스 | HBM2 cycles | DDR4 cycles | 증가배수 |
|---|---|---|---|
| MatMul 256³ OS | 37,672 | 236,628 | **6.3×** |
| MatMul 256³ WS DB ON | 25,065 | 133,664 | **5.3×** |
| MatMul 256³ WS DB OFF | 34,932 | 152,384 | **4.4×** |
| Transformer seq=128 WS | 8,053,020 | 14,049,943 | **1.7×** |
| Transformer seq=512 WS | 53,798,913 | 74,543,976 | **1.4×** |
| Transformer seq=1024 WS | 170,818,018 | 215,777,213 | **1.3×** |

**분석**

- MatMul(소규모 DRAM 트래픽)은 DDR4 페널티 매우 큼 (5~6×): tile당 DRAM 접근이 latency-dominant
- Transformer(대규모 serial load)는 DDR4 페널티 상대적으로 작음 (1.3~1.7×): 순차 burst 접근이라 DDR4 row buffer 재활용 효과
- HBM2에서 DB Hit Rate 82~95%였던 케이스가 DDR4에서 전부 0% → DDR4로는 compute hiding 불가
- **결론**: NPU 설계 시 HBM 필수; DDR4는 대역폭 부족으로 systolic array가 항상 memory-bound
