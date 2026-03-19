# Dataflow Benchmark Report (DRAMSim3 batch=16)

**수정 내용**: `_DRAMSIM3_TICK_BATCH` 2048 → 16
**수정 효과**: DRAMSim3 over-tick 제거 → DB ON/OFF 차이 올바르게 반영, 시뮬레이션 속도 향상

---

## Group 1: M=32 vs M=256, OS/WS/IS 비교

| Dataflow | M=32 cycles | M=256 cycles | Tile Size (M,N,K) | DB Hits/Misses |
|---|---|---|---|---|
| OS | 273 | 37,672 | (32,32,256) | 0/63 |
| WS | 305 | 25,065 | (256,32,32) | 52/11 |
| IS | 305 | 25,612 | (32,256,32) | 51/12 |

**분석**

- M=32는 tile 1개로 처리 → DB 없음, 차이 미미
- M=256에서 OS는 tile_m=32로 고정되어 64 tiles → 모두 DB Miss (memory-bound, hiding 불가)
- WS/IS는 tile_m=256으로 큰 tile을 사용 → **DB Hit율 82~81%**, 효과적인 hiding

---

## Group 2: K 스케일링 (M=N=64, K=64/256/1024), OS vs WS

| K | OS cycles | WS cycles | OS tile_k | WS tile_k | WS DB Hits/Misses |
|---|---|---|---|---|---|
| 64 | 1,253 | 1,011 | 64 | 32 | 1/2 |
| 256 | 2,157 | 3,025 | 256 | 32 | 2/13 |
| 1024 | 7,182 | 11,729 | 1024 | 32 | 2/61 |

**분석**

- K=64: WS(1,011) < OS(1,253) → WS가 유리 (weight reuse)
- K=256, 1024: WS가 OS보다 느림 → WS의 tile_k=32로 고정되어 K가 클수록 tile 수 급증 (K=1024 → 32 passes per (m,n)), 매 tile마다 weight 재로드 없이도 activation load 비용 누적
- OS는 tile_k=K 전체 가능 → K가 커도 tile 수는 1로 유지

---

## Group 3: Double Buffer ON vs OFF (WS, M=N=K=256)

| 설정 | Total Cycles | DB Hits/Misses | Stall Cycles |
|---|---|---|---|
| DB ON | 25,065 | 52/11 | 1,939 |
| DB OFF | 34,932 | 0/0 | 0 |
| **차이** | **-9,867 (-28%)** | — | — |

**분석**

- batch=2048 시절: DB ON = DB OFF = 373,013 (버그, over-tick으로 prefetch 시점 동기화 실패)
- batch=16 수정 후: DB ON이 **28% 빠름** → compute hiding 정상 동작
- 이론적 최대 hiding = (64-1) × 350 = 22,050 cycles, 실제 9,867 cycles (~45%)
  - 나머지 차이: WS dataflow에서 act-only tile의 HBM2 실제 load time이 이론치와 다름

---

## Group 4: Transformer (seq=128/512/1024, WS)

| seq_len | Total Cycles | Tile Size | DB Hits/Misses | Stall Cycles | Hit Rate |
|---|---|---|---|---|---|
| 128 | 8,053,020 | (128,32,32) | 6364/712 | 147,744 | 90% |
| 512 | 53,798,913 | (512,32,32) | 7248/404 | 140,445 | 95% |
| 1024 | 170,818,018 | (1024,32,32) | 7824/596 | 293,518 | 93% |

**참고**: seq=512 OS = 56,337,791 cycles (DB Hit율 10%), WS(53,798,913)가 OS 대비 **5% 빠름**

**분석**

- seq가 클수록 tile_m이 커져 weight tile 크기 증가 → load 시간 증가 → stall 증가
- 그럼에도 Hit Rate 90~95% 유지 → DB가 transformer에서 효과적
- seq=1024 총 cycle이 seq=512의 3.2배인 이유: tile 수 비례 증가 + 큰 weight tile로 인한 stall 증가

---

## 이전(batch=2048) vs 현재(batch=16) 비교

| 케이스 | 이전 cycles | 현재 cycles | 변화 | 시뮬 시간 변화 |
|---|---|---|---|---|
| MatMul 256³ WS DB ON | 373,013 | 25,065 | -93% | 1.06s → 0.46s |
| MatMul 256³ WS DB OFF | 373,013 | 34,932 | -91% | — |
| DB ON vs OFF 차이 | **0 (버그)** | **+9,867** | 수정됨 | — |
| Transformer seq=128 WS | 40,451,261 | 8,053,020 | -80% | 40s → 36s |
| Transformer seq=512 WS | 102,284,694 | 53,798,913 | -47% | — |
| Transformer seq=1024 WS | 247,424,908 | 170,818,018 | -31% | — |

**cycle 감소 원인**: 이전에는 over-tick으로 DRAMSim3가 실제 완료 시점보다 최대 2047 cycle 추가 진행,
이것이 누적되어 전체 cycle이 비정상적으로 높았음. 수정 후 실제 HBM2 timing 기준으로 측정.

**시뮬 시간 감소 원인**: over-tick이 없어 DRAMSim3 내부에서 돌리는 총 cycle 수가 줄어듦.
batch size를 줄여도 Python function call overhead 증가보다 DRAMSim3 내부 tick 절감 효과가 더 큼.
