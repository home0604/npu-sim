# CLAUDE.md

## Rules

- Do not include Co-Authored-By lines in commit messages
- Keep code concise — avoid verbose or sprawling implementations
- When making large structural changes (file renames, new modules, architecture changes), update this CLAUDE.md as part of the same task

## Commands

```bash
# Run all tests (~91 tests)
pytest tests/

# Run simulator
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256
python3 scripts/run_sim.py --workload matmul --M 64 --N 64 --K 64 --vq --config configs/quip_2bpv.yaml
python3 scripts/run_sim.py --workload attention --seq-len 128 --hidden-dim 768 --num-heads 12

# Flags: --check-correctness, --json, --vq, --dataflow OS/WS/IS, --config path

# Tile breakdown (single-tile Gantt chart, non-VQ vs VQ)
python3 scripts/run_tile_breakdown.py --config configs/gptvq_3.125bpv.yaml --plot

# Benchmark (saves to reports/benchmark/)
python3 scripts/run_benchmark.py

# Build DRAMSim3 (optional)
bash scripts/setup_dramsim3.sh
```

## Architecture

**Flow:** `run_sim.py` → `NPUSimulator` → Tiler → StationaryDataflow → TileScheduler → (SystolicArray + MemoryController + DoubleBuffer)

- **core/clock.py** — `SimulationEngine`: heapq-based event-driven (jump-to-next-event)
- **core/config.py** — YAML → nested dataclasses (`NPUConfig`). Supports `extends:` inheritance
- **compute/systolic_array.py** — Multi-dataflow (OS/WS/IS) with SR/SC/T abstraction, multi-pass
- **compute/dequant_unit.py** — VQ dequantization: codebook lookup cycles from SRAM bank structure
- **memory/sram.py** — `BankedSRAM`: interleaved banking, single/dual-port conflict modeling
- **memory/dram_interface.py** — `SimpleDRAMModel` (analytical) / `DRAMSim3Adapter` (cycle-accurate). Begin/end pattern for multi-channel parallel reads
- **memory/memory_controller.py** — Blocking (`load_from_dram`) and non-blocking (`begin_load_from_dram`/`end_load_from_dram`) APIs
- **memory/double_buffer.py** — Two-slot state machine for compute-prefetch overlap
- **dataflow/tiler.py** — Tile sizing within SRAM budget per dataflow
- **dataflow/stationary.py** — Tile schedule generation for OS/WS/IS with load/store flags
- **dataflow/scheduler.py** — Tile execution with double buffering, parallel DRAM issue, WS/IS accumulation cost
- **dataflow/vq_tiler.py** — VQ tile sizing (codebook/index/scale/dequant_weight/activation/output buffers)
- **dataflow/vq_dataflow.py** — VQ schedule generation with dequant step
- **dataflow/vq_scheduler.py** — VQ execution with dequant pipeline
- **sim/simulator.py** — `NPUSimulator` orchestrator: `run_matmul()`, `run_attention()`, `run_transformer_layer()`

## Key Design Decisions

- **Event-driven**: heapq jump-to-next-event, no per-cycle ticking
- **Analytical cycles + PyTorch functional correctness**: `--check-correctness` verifies both
- **Multi-dataflow OS/WS/IS**: unified SR/SC/T mapping in systolic array
- **Begin/end DRAM**: weight/activation issued concurrently; DRAMSim3 multi-channel parallelism. SimpleDRAMModel serializes (single bus)
- **WS/IS accumulation**: partial-sum SRAM read-modify-write at tile boundaries
- **Per-buffer BankedSRAM**: each buffer group owns its SRAM instance (no cross-buffer conflicts)
- **VQ support**: GPTVQ, AQLM, QuIP# via unified VQ pipeline. Codebook dtype includes INT4. `num_stages` for RVQ/AQ

## Configuration

HW params in `configs/default.yaml`. VQ configs inherit via `extends: default.yaml`:
- `configs/gptvq_3.125bpv.yaml` — k=64, d=2, INT8 codebook
- `configs/aqlm_2bpv.yaml` — k=256, d=8, FP16, M=2 (additive)
- `configs/quip_2bpv.yaml` — k=256, d=8, INT4 (E8P lattice)

## Python

Requires Python 3.10+. Dependencies: torch >= 2.0, pyyaml >= 6.0, numpy >= 1.24.
