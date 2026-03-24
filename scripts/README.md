# scripts/

NPU-Sim 실행 스크립트 모음.

---

## run_sim.py

단일 워크로드 시뮬레이션. 사이클 카운트, DRAM 트래픽, 활용률 등을 출력한다.

```bash
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --dataflow WS --use-db
python3 scripts/run_sim.py --workload attention --seq-len 128 --hidden-dim 768 --num-heads 12
python3 scripts/run_sim.py --workload transformer --seq-len 128 --hidden-dim 768 --num-heads 12
python3 scripts/run_sim.py --workload matmul --M 64 --N 64 --K 64 --check-correctness
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --config configs/gptvq.yaml
```

주요 옵션:

| 옵션 | 설명 |
|------|------|
| `--dataflow` | OS / WS / IS |
| `--use-db` | Double Buffer 활성화 |
| `--json` | 결과를 JSON 형식으로 출력 |
| `--check-correctness` | PyTorch 결과와 수치 비교 |
| `--config` | YAML config 파일 경로 (기본: `configs/default.yaml`) |

출력 예시:

```
MatMul 256x256x256 (WS, DB=OFF)
  total_cycles        : 12,345
  compute_utilization : 87.3%
  dram_read_bytes     : 196,608
  dram_write_bytes    : 262,144
  tile: 32x32x256, 8x8x1 tiles
```

---

## run_benchmark.py

여러 설정을 한 번에 실행해 그룹별 비교 테이블을 출력한다.
결과는 `reports/results/` 에 `YYYY-MM-DD_HHMM_benchmark.txt` / `.json` 으로 저장된다.

```bash
python3 scripts/run_benchmark.py
```

벤치마크 그룹:

| 그룹 | 목적 |
|------|------|
| Group 1: Dataflow Comparison | OS / WS / IS 사이클·활용률 비교 |
| Group 2: Size Scaling | 행렬 크기 증가에 따른 사이클 추이 (WS) |
| Group 3: Double Buffer | DB ON vs OFF 사이클·stall 비교 (WS) |
| Group 4: K Scaling | K 차원 변화에 따른 메모리 트래픽 비교 |

출력 예시:

```
==================================================
  Group 1: Dataflow Comparison (DB=OFF)
==================================================
      Size  Dataflow  Tiles  Total Cycles  Util  DRAM Read  DRAM Write
--------------------------------------------------
  32x32x32        OS      1         3,201  12.4%     16,384      16,384
  32x32x32        WS      1         2,890  13.7%      8,192      16,384
```

---

## run_dequant_sweep.py

GPTVQ dequantization의 SRAM bank-aware 사이클 모델을 `vector_dim` × `num_stages` 조합으로 스윕한다.
설정은 `configs/gptvq.yaml` 기반이며 파라미터는 스크립트 상단 상수에서 수정한다.

```bash
python3 scripts/run_dequant_sweep.py
```

스크립트 상단 파라미터:

```python
VECTOR_DIMS = [32, 64, 128, 512, 1024]  # 코드북 entry 차원
NUM_STAGES  = [1, 2, 4]                 # RVQ stage 수
TILE_M = 32
TILE_K = 256
```

출력 예시:

```
SRAM: 32 banks × 64B = 16384 bits total bandwidth/cycle
Tile: M=32, K=256

  vector_dim  stages (S)    read_latency     num_vectors    dequant_cycles
--------------------------------------------------------------------------
          32           1           1 cyc             256               259
         128           1           1 cyc              64                67
         512           4           2 cyc              32                67
        1024           4           4 cyc              16                67
--------------------------------------------------------------------------
```

열 설명:

| 열 | 설명 |
|----|------|
| `vector_dim` | 코드북 entry 차원 수 (d) |
| `stages (S)` | RVQ stage 수 |
| `read_latency` | entry 1개를 SRAM에서 읽는 사이클 = `ceil(d × entry_bytes × 8 × S / (N_bank × w_b))` |
| `num_vectors` | 타일 내 총 lookup 횟수 = `tile_m × ceil(tile_k / d)` |
| `dequant_cycles` | 전체 dequant 사이클 = `pipeline_stages + num_vectors × read_latency` |

---

## commands.sh

자주 쓰는 명령어 모음. 실행용이 아니라 복사해서 사용하는 참고 파일.
