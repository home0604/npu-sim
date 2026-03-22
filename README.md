# NPU Simulator

Cycle-accurate event-driven NPU simulator with multi-dataflow systolic array, banked SRAM, double buffering, DRAMSim3 integration, and GPTVQ support.

## Project Structure

```
npu-sim/
├── configs/
│   ├── default.yaml               # 32x32 array, 512KB SRAM, FP16, OS dataflow
│   └── gptvq.yaml                 # GPTVQ config (R=16, d=2, 4-bit index)
├── core/
│   ├── config.py                  # YAML → dataclass config loader (NPUConfig, GPTVQConfig)
│   ├── clock.py                   # Event-driven SimulationEngine (heapq, jump-to-next)
│   ├── events.py                  # EventType enum + EventQueue
│   ├── stats.py                   # ComputeStats, MemoryStats, GPTVQStats, TileStats
│   └── datatypes.py               # DataType (INT8/FP16/BF16/FP32)
├── compute/
│   ├── systolic_array.py          # OS/WS/IS 통합 systolic array (SR/SC/T 추상화)
│   ├── dequant_unit.py            # GPTVQ dequantization unit (codebook lookup + scaling)
│   └── special_functions.py       # Softmax, LayerNorm, GELU cycle models
├── memory/
│   ├── sram.py                    # BankedSRAM (single/dual-port, conflict modeling)
│   ├── sram_buffers.py            # Buffer partitioning (standard 3-way / GPTVQ 5-way)
│   ├── double_buffer.py           # Two-slot state machine for compute-prefetch overlap
│   ├── dram_interface.py          # SimpleDRAMModel + DRAMSim3 pybind11 adapter
│   └── memory_controller.py       # DRAM↔SRAM data movement coordination
├── dataflow/
│   ├── tiler.py                   # SRAM-aware tile size calculator
│   ├── stationary.py              # OS/WS/IS dataflow schedule generation (TileOp)
│   ├── scheduler.py               # Tile execution with double buffering overlap
│   ├── gptvq_tiler.py             # GPTVQ-aware tiler (codebook/index/scale budgets)
│   ├── gptvq_dataflow.py          # GPTVQ OS dataflow schedule generation (GPTVQTileOp)
│   └── gptvq_scheduler.py         # GPTVQ tile execution (load → dequant → compute)
├── workloads/
│   └── attention.py               # Attention + Transformer layer op decomposition
├── sim/
│   ├── simulator.py               # NPUSimulator (standard + GPTVQ mode)
│   └── functional_check.py        # Tile-by-tile correctness verification
├── scripts/
│   ├── run_sim.py                 # CLI entry point
│   ├── run_benchmark.py           # Benchmark automation (grouped tables + JSON)
│   ├── commands.sh                # 명령어 모음 (복사용)
│   └── setup_dramsim3.sh          # DRAMSim3 build script
├── reports/                       # Benchmark results (txt + json)
└── tests/
```

## Setup

```bash
pip install -e ".[dev]"

# DRAMSim3 (optional)
bash scripts/setup_dramsim3.sh
```

## Usage

### MatMul

```bash
# 기본 실행
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256

# Dataflow 선택 (OS / WS / IS)
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --dataflow WS

# Double buffering 활성화
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --dataflow WS --use-db

# 연산 정합성 검증
python3 scripts/run_sim.py --workload matmul --M 64 --N 64 --K 64 --check-correctness

# JSON 출력
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --json
```

### Attention / Transformer

```bash
python3 scripts/run_sim.py --workload attention --seq-len 128 --hidden-dim 768 --num-heads 12
python3 scripts/run_sim.py --workload transformer --seq-len 512 --hidden-dim 768 --num-heads 12
```

### DRAMSim3

```bash
# setup 후 사용
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --use-dramsim3

# 또는 config에서 dram.use_dramsim3: true
```

### Benchmark

```bash
# 전체 벤치마크 실행 (reports/results/ 에 txt + json 저장)
python3 scripts/run_benchmark.py
```

## Dataflow

단일 `SystolicArray` 클래스가 OS/WS/IS를 모두 지원. SR/SC/T 추상화로 통합:

- **OS** (Output Stationary): SR=M, SC=N, T=K — output이 PE에 고정, W/A 스트리밍
- **WS** (Weight Stationary): SR=K, SC=N, T=M — weight이 PE에 고정
- **IS** (Input Stationary): SR=K, SC=M, T=N — input이 PE에 고정

Cycle formula:
- OS: `SR + SC + T - 2` (preload 없음)
- WS/IS: `2*SR + SC + T - 2` (stationary data preload 포함)

---

## GPTVQ (Vector Quantized Weights)

Weight를 codebook + index로 압축하여 DRAM 트래픽을 줄이고, dequantization 후 OS dataflow로 연산.

### Flow

```
DRAM → [codebook, index, (scale, zero_point)] → SRAM
                        ↓
            Dequantization Unit
       codebook[index] (* scale + zero_point)
                        ↓
            Dequantized Weight [tile_m, tile_k]
                        ↓
DRAM → [activation] → SRAM → Systolic Array (OS)
                        ↓
                  Output → SRAM → DRAM
```

### 사용법

```bash
# 기본 GPTVQ (R=16, d=2, 4-bit index, scaling 없음)
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --gptvq

# Correctness check
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --gptvq --check-correctness

# Scaling 활성화
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --gptvq --use-scaling

# Codebook 크기 / 벡터 차원 변경
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --gptvq --codebook-size 256 --vector-dim 4
```

### 출력 예시

```
============================================================
NPU Simulation Summary
============================================================
  Total Cycles:          530,688
  Compute Utilization:   3.09%
  DRAM Read:             1,314,816 bytes

  --- GPTVQ ---
  Dequant Cycles:        262,336
  Vectors Dequantized:   262,144
  Codebook Load:         4,096 bytes
  Index Load:            262,144 bytes
  Scale Load:            0 bytes
  Compression Ratio:     3.99x
============================================================
```

### Config (gptvq.yaml)

```yaml
gptvq:
  enabled: true
  codebook_size: 16       # R: centroids 수 (4-bit index)
  vector_dim: 2           # d: codebook vector 차원
  index_bits: 4           # bits per index = log2(R)
  use_scaling: false      # per-group scaling (s * codebook[idx] + z)
  dequant_pipeline_stages: 3
  dequant_throughput: 1   # vectors/cycle
```
