# Multi-Dataflow (OS / WS / IS) 지원 변경 내역

날짜: 2026-03-09

## 목적

기존에 OS(Output Stationary)만 지원하던 시뮬레이터에 WS(Weight Stationary), IS(Input Stationary)를 추가.
EONSim의 SR/SC/T 매핑 방식을 채택하여 단일 수식 틀로 3종 dataflow를 통합.
이후 전환 과정에서 남은 dead code를 정리하여 코드베이스를 간결하게 유지.

---

## 변경 파일 목록

### 1. `npu_sim/core/config.py`

- `SystolicArrayConfig`에 `dataflow: str = "OS"` 필드 추가

### 2. `npu_sim/configs/default.yaml`

- `systolic` 섹션에 `dataflow: "OS"` 추가

### 3. `npu_sim/compute/systolic_array.py`

- 클래스 docstring을 OS/WS/IS 통합 설명으로 갱신
- `_map_dataflow(tile_m, tile_n, tile_k, dataflow)` 메서드 추가
  - OS: SR=M, SC=N, T=K
  - WS: SR=K, SC=N, T=M
  - IS: SR=K, SC=M, T=N
- `analytical_cycles()`에 `dataflow: str = "OS"` 인자 추가
- 내부 루프가 SR/SC/T 기반으로 동작하도록 변경 (기본값 OS로 하위호환 유지)

### 4. `npu_sim/dataflow/stationary.py` (신규)

- 기존 `ws_dataflow.py`의 `WSDataflow` → `StationaryDataflow`로 재작성
- `TileOp` dataclass 동일하게 포함
- `generate_schedule(M, N, K)`에서 dataflow별 분기:
  - `_schedule_os()`: M→N→K 루프, 매 타일 weight/activation 로드, 마지막 K에서 store, K>0이면 accumulate
  - `_schedule_ws()`: K→N→M 루프, 첫 M에서만 activation 로드, 매 타일 weight 로드, 마지막 K에서 store, K>0이면 accumulate
  - `_schedule_is()`: M→K→N 루프, 첫 N에서만 weight 로드, 매 타일 activation 로드, 마지막 K에서 store, K>0이면 accumulate

### 5. `npu_sim/dataflow/ws_dataflow.py` (삭제)

- `StationaryDataflow`로 완전히 대체됨

### 6. `npu_sim/dataflow/tiler.py`

- `compute_tiles(M, N, K, dataflow="OS")` — dataflow 인자 추가
  - OS: tile_m ≤ rows, tile_n ≤ cols, tile_k 최대화
  - WS: tile_k ≤ rows, tile_n ≤ cols, tile_m 최대화
  - IS: tile_k ≤ rows, tile_m ≤ cols, tile_n 최대화
- `estimate_dram_traffic(M, N, K, dataflow="OS")` — dataflow별 재사용 패턴 반영

### 7. `npu_sim/dataflow/scheduler.py`

- import: `ws_dataflow.TileOp` → `stationary.TileOp`
- `TileScheduler.__init__`에 `dataflow: str = "OS"` 인자 추가
- `analytical_cycles()` 호출 3곳에 `dataflow=self.dataflow` 전달

### 8. `npu_sim/sim/simulator.py`

- import: `WSDataflow` → `StationaryDataflow`
- `self.dataflow = StationaryDataflow(self.tiler, dataflow=config.systolic.dataflow)`
- `_create_scheduler()`에 `dataflow=self.config.systolic.dataflow` 전달
- `_get_fp32_functional_components()`도 `StationaryDataflow` 사용

### 9. `npu_sim/sim/functional_check.py`

- import: `WSDataflow` → `StationaryDataflow`
- 미사용 함수 제거: `_attention_with_matmul_func()`, `reference_attention_tiled()`, `reference_transformer_layer_tiled()`
  - `_attention_with_matmul_func()` 로직은 `run_attention_functional()` 안에 인라인
- `Any` import 제거

### 10. `tests/test_systolic.py`

- 기존 4개 테스트를 OS 명시 (`test_analytical_cycles_os_*`)
- 추가: `test_analytical_cycles_ws` — WS 매핑 수식 검증
- 추가: `test_analytical_cycles_is` — IS 매핑 수식 검증
- 추가: `test_analytical_cycles_default_is_os` — 기본값이 OS인지 확인

### 11. `tests/test_tiler.py`

- 추가: `test_ws_tile_k_maps_to_rows` — WS에서 tile_k ≤ rows
- 추가: `test_is_tile_k_maps_to_rows` — IS에서 tile_k ≤ rows
- 추가: `test_os_tile_m_n_maps_to_array` — OS에서 tile_m/n ≤ array
- 추가: `test_ws_tiles_cover_full_matrix` — WS 타일 커버리지
- 추가: `test_is_tiles_cover_full_matrix` — IS 타일 커버리지

### 12. `tests/test_e2e.py`

- `make_simulator()`에 `dataflow="OS"` 인자 추가
- 추가: `test_matmul_os_dataflow` — OS correctness
- 추가: `test_matmul_ws_dataflow` — WS correctness
- 추가: `test_matmul_is_dataflow` — IS correctness
- 추가: `test_different_dataflows_same_mac_ops` — OS/WS/IS 동일 MAC ops
- 추가: `test_matmul_ws_multi_tile` — WS 멀티타일 correctness
- 추가: `test_matmul_is_multi_tile` — IS 멀티타일 correctness

---

## 변경하지 않은 파일

- `TileOp` dataclass 구조 — 필드 동일
- `TileScheduler` 핵심 로직 — 플래그 기반이라 변경 불필요
- `DoubleBufferController`, `BankedSRAM`, `MemoryController` — 변경 없음
- `EventQueue`, `SimulationEngine` — 변경 없음
- `scripts/run_sim.py` — 직접 WSDataflow를 import하지 않아 변경 불필요

---

## 테스트 결과

```
57 passed in 1.59s
```

기존 43개 + 새로운 14개 = 총 57개 테스트 전체 통과.
