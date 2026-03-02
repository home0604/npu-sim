# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cycle-accurate event-driven NPU (Neural Processing Unit) simulator with a weight-stationary systolic array, banked SRAM, double buffering, and optional DRAMSim3 integration. Written in Python, uses PyTorch for functional correctness and analytical formulas for cycle counting.

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests (39 tests across 6 files)
pytest tests/

# Run a single test file
pytest tests/test_e2e.py

# Run a specific test
pytest tests/test_e2e.py::TestE2E::test_small_matmul -v

# Run simulator
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256
python3 scripts/run_sim.py --workload attention --seq-len 128 --hidden-dim 768 --num-heads 12
python3 scripts/run_sim.py --workload transformer --seq-len 128 --hidden-dim 768 --num-heads 12

# Flags: --check-correctness, --use-dramsim3, --json, --config path/to/config.yaml

# Build DRAMSim3 (optional, requires g++ and cmake)
bash scripts/setup_dramsim3.sh
```

## Architecture

**Simulation flow:** `run_sim.py` → `NPUSimulator` → Tiler → WSDataflow → TileScheduler → (SystolicArray + MemoryController + DoubleBuffer)

Key architectural layers:

- **core/clock.py** — `SimulationEngine`: heapq-based event-driven simulation (jump-to-next-event, no idle cycle ticking). All timing goes through `schedule_event(delay, type)` / `schedule_event_at(cycle, type)`.
- **core/events.py** — `EventType` enum and `EventQueue`. Events: DRAM_READ_COMPLETE, SRAM_WRITE_COMPLETE, COMPUTE_DONE, PREFETCH_DONE, etc.
- **core/config.py** — Loads YAML config into nested dataclasses (`NPUConfig` with `SystolicConfig`, `SRAMConfig`, `DRAMConfig`, etc.). Default config: `configs/default.yaml`.
- **compute/systolic_array.py** — Weight-stationary systolic array. Uses `torch.matmul()` for computation, analytical cycle formula: `fill + K + drain = (M+N-1) + K + (M+N-1)`. Handles multi-pass when tiles exceed array dimensions.
- **memory/sram.py** — `BankedSRAM` with interleaved banking (`bank_id = (addr / bank_width) % num_banks`). Models single-port (one R/W per bank/cycle) and dual-port (one R + one W per bank/cycle) conflicts.
- **memory/dram_interface.py** — `SimpleDRAMModel` (analytical: latency + transfer cycles with bus contention) and `DRAMSim3Adapter` (cycle-accurate via pybind11 wrapper).
- **memory/double_buffer.py** — Two-slot state machine (EMPTY→LOADING→READY→COMPUTING→DRAINING) enabling compute-prefetch overlap.
- **dataflow/tiler.py** — Computes tile sizes (M, N, K) fitting within SRAM budget. Double buffering halves per-slot budget.
- **dataflow/ws_dataflow.py** — Weight-Stationary schedule: outer loops over (m, n) tiles load weight once, inner k loop loads activations and computes. Minimizes weight DRAM traffic.
- **dataflow/scheduler.py** — `TileScheduler` executes the tile schedule with double buffering overlap, detects memory-bound stalls when prefetch can't complete before compute finishes.
- **sim/simulator.py** — `NPUSimulator` orchestrates everything. Entry points: `run_matmul()`, `run_attention()`, `run_transformer_layer()`.
- **workloads/attention.py** — Decomposes attention and transformer layers into sequences of matmul + special function operations.

## Key Design Decisions

- **Event-driven, not cycle-stepped**: The simulator jumps between events (heapq), critical for efficiency when compute blocks span thousands of cycles.
- **Analytical + functional**: Cycle counts use closed-form formulas; correctness uses PyTorch tensors. The `--check-correctness` flag runs both and compares.
- **Weight-stationary dataflow**: Weights loaded once per (m, n) tile pair and reused across all k-tiles.
- **SRAM buffer partitioning**: Weight/activation/output buffers are fixed fractions of total SRAM (configurable via `weight_buffer_fraction`, `activation_buffer_fraction`, `output_buffer_fraction` in config).

## Configuration

All hardware parameters are in `configs/default.yaml`. Key knobs: `systolic.rows/cols`, `sram.total_size_kb`, `sram.num_banks`, `sram.port_type` (single/dual), `dtype.compute_dtype` (INT8/FP16/BF16/FP32), `dram.bandwidth_gbps`, `double_buffer.enabled`.

Config can also be modified programmatically via `NPUConfig` dataclass before passing to `NPUSimulator`.

## Python Version

Requires Python 3.10+. Dependencies: torch >= 2.0, pyyaml >= 6.0, numpy >= 1.24.
