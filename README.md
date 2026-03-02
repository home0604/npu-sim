All done! Here's a summary of what was built:

## NPU Simulator (`npu_sim/`) - Complete

### Structure (20 files)
```
npu_sim/
├── configs/default.yaml           # 32x32 array, 512KB SRAM, INT8, dual-port
├── core/
│   ├── config.py                  # YAML -> dataclass config loader
│   ├── clock.py                   # Event-driven SimulationEngine (heapq jump-to-next)
│   ├── events.py                  # EventType enum + EventQueue
│   ├── stats.py                   # ComputeStats, MemoryStats, TileStats
│   └── datatypes.py               # DataType (INT8/FP16/BF16/FP32)
├── compute/
│   ├── systolic_array.py          # PyTorch matmul + analytical cycle counting
│   └── special_functions.py       # Softmax, LayerNorm, GELU cycle models
├── memory/
│   ├── sram.py                    # BankedSRAM (single/dual-port, conflict modeling)
│   ├── sram_buffers.py            # Buffer partitioning + double buffer slots
│   ├── double_buffer.py           # State machine (EMPTY->LOADING->READY->COMPUTING->DRAINING)
│   ├── dram_interface.py          # SimpleDRAMModel + DRAMSim3 pybind11 interface
│   └── memory_controller.py       # DRAM<->SRAM data movement coordination
├── dataflow/
│   ├── tiler.py                   # SRAM-aware tile size calculator
│   ├── ws_dataflow.py             # Weight Stationary tile loop schedule
│   └── scheduler.py               # Tile execution with double buffering overlap
├── workloads/attention.py         # Attention + Transformer layer op generation
├── sim/simulator.py               # Top-level orchestrator (run_matmul/attention/transformer)
├── scripts/
│   ├── setup_dramsim3.sh          # DRAMSim3 clone + build + pybind11 wrapper
│   └── run_sim.py                 # CLI entry point
└── tests/ (39 tests, all passing)
```

### Key Design Points
- **Systolic array computation**: PyTorch `torch.matmul()` for correctness, analytical formulas for cycles (`K + 2*(M+N-1)`)
- **Event-driven engine**: heapq-based, jumps to next event (no idle cycle simulation)
- **SRAM**: interleaved banking, single/dual-port modeling, per-bank busy tracking
- **Double buffering**: overlaps prefetch with compute; correctly identifies memory-bound stalls
- **DRAMSim3**: pybind11 wrapper ready (`scripts/setup_dramsim3.sh`), with `SimpleDRAMModel` fallback

### Usage
```bash
python3 npu_sim/scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256
python3 npu_sim/scripts/run_sim.py --workload attention --seq-len 128 --hidden-dim 768 --num-heads 12
python3 npu_sim/scripts/run_sim.py --workload transformer --seq-len 128 --hidden-dim 768 --num-heads 12
```

### Using DRAMSim3
- Build DRAMSim3 and the pybind11 wrapper: `scripts/setup_dramsim3.sh`
- Enable via config: in `configs/default.yaml` set `dram.use_dramsim3: true`, or run with `--use-dramsim3`
- The simulator uses a synchronous adapter so the existing scheduler (no event-driven DRAM) works unchanged; completion cycles are resolved by ticking DRAMSim3 and the event queue until the request finishes.

### Next Steps for Research
- `scripts/setup_dramsim3.sh` 실행하여 DRAMSim3 연동
- SRAM banking/port type sweep: `config.yaml`에서 `num_banks`, `port_type` 변경
- Array size sweep: `rows`/`cols` 변경하여 compute utilization 비교
- FP16/BF16 워크로드 실행하여 data type별 DRAM traffic 비교