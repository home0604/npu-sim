All done! Here's a summary of what was built:

## NPU Simulator (`npu_sim/`) - Complete

### Structure
```
npu_sim/
├── configs/
│   ├── default.yaml               # 32x32 array, 512KB SRAM, FP16, WS dataflow
│   └── gptvq.yaml                 # GPTVQ config (R=16, d=2, 4-bit index, OS dataflow)
├── core/
│   ├── config.py                  # YAML -> dataclass config loader (NPUConfig, GPTVQConfig)
│   ├── clock.py                   # Event-driven SimulationEngine (heapq jump-to-next)
│   ├── events.py                  # EventType enum + EventQueue (incl. DEQUANT events)
│   ├── stats.py                   # ComputeStats, MemoryStats, GPTVQStats, TileStats
│   └── datatypes.py               # DataType (INT8/FP16/BF16/FP32)
├── compute/
│   ├── systolic_array.py          # SystolicArray (WS) + OSSystolicArray (OS)
│   ├── dequant_unit.py            # GPTVQ DequantizationUnit (codebook lookup + scaling)
│   └── special_functions.py       # Softmax, LayerNorm, GELU cycle models
├── memory/
│   ├── sram.py                    # BankedSRAM (single/dual-port, conflict modeling)
│   ├── sram_buffers.py            # Buffer partitioning (WS 3-way / GPTVQ 5-way)
│   ├── double_buffer.py           # State machine (EMPTY->LOADING->READY->COMPUTING->DRAINING)
│   ├── dram_interface.py          # SimpleDRAMModel + DRAMSim3 pybind11 interface
│   └── memory_controller.py       # DRAM<->SRAM data movement coordination
├── dataflow/
│   ├── tiler.py                   # SRAM-aware tile size calculator (WS)
│   ├── gptvq_tiler.py             # GPTVQ-aware tiler (codebook/index/scale budgets)
│   ├── ws_dataflow.py             # Weight Stationary tile loop schedule
│   ├── os_dataflow.py             # Output Stationary tile loop schedule (GPTVQ)
│   ├── scheduler.py               # WS tile execution with double buffering
│   └── gptvq_scheduler.py         # GPTVQ tile execution (load→dequant→compute)
├── workloads/attention.py         # Attention + Transformer layer op generation
├── sim/
│   ├── simulator.py               # NPUSimulator (WS + GPTVQ mode)
│   └── functional_check.py        # Correctness verification (WS + GPTVQ)
├── scripts/
│   ├── setup_dramsim3.sh          # DRAMSim3 clone + build + pybind11 wrapper
│   └── run_sim.py                 # CLI entry point (--gptvq, --use-scaling, etc.)
└── tests/ (67 tests, all passing)
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


### Using GPTVQ (Vector Quantized Weights + Output-Stationary Dataflow)
GPTVQ는 weight를 codebook + index로 압축하여 DRAM 트래픽을 줄이고, Output-Stationary dataflow 기반 systolic array에서 연산합니다.

**기본 사용법 (codebook + index만, scaling 없음):**
```bash
# GPTVQ 전용 config 사용
python3 npu_sim/scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --gptvq

# correctness check (tile-by-tile dequant+matmul 결과를 reference와 비교)
python3 npu_sim/scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --gptvq --check-correctness
```

**Scaling 적용 (s * codebook[idx] + z):**
```bash
# CLI 플래그로 scaling 활성화
python3 npu_sim/scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --gptvq --use-scaling

# 또는 config YAML에서 gptvq.use_scaling: true 설정
```

**Codebook 크기/벡터 차원 변경:**
```bash
# R=256 (8-bit index), d=4
python3 npu_sim/scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --gptvq --codebook-size 256 --vector-dim 4

# R=16 (4-bit index), d=8
python3 npu_sim/scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --gptvq --codebook-size 16 --vector-dim 8
```

**GPTVQ Simulation Flow:**
```
DRAM → [codebook, weight index, (scale, zero_point)] → SRAM
                          ↓
              Dequantization Unit
         codebook[index] (* scale + zero_point)
                          ↓
              Dequantized Weight [tile_m, tile_k]
                          ↓
DRAM → [activation] → SRAM → Output-Stationary Systolic Array
                                (output이 PE에 고정, W와 A가 스트리밍)
                          ↓
                    Output → SRAM → DRAM
```

**GPTVQ 출력 예시:**
```
============================================================
NPU Simulation Summary
============================================================
  Total Cycles:          530,688
  Compute Utilization:   3.09%
  DRAM Read:             1,314,816 bytes    ← WS 대비 감소 (2,097,152 → 1,314,816)
  --- GPTVQ ---
  Dequant Cycles:        262,336            ← dequantization에 소요된 cycle
  Vectors Dequantized:   262,144
  Codebook Load:         4,096 bytes
  Index Load:            262,144 bytes
  Scale Load:            0 bytes            ← scaling 비활성
  Compression Ratio:     3.99x              ← 원본 weight 대비 압축률
============================================================
```

### Next Steps for Research
- `scripts/setup_dramsim3.sh` 실행하여 DRAMSim3 연동
- SRAM banking/port type sweep: `config.yaml`에서 `num_banks`, `port_type` 변경
- Array size sweep: `rows`/`cols` 변경하여 compute utilization 비교
- FP16/BF16 워크로드 실행하여 data type별 DRAM traffic 비교