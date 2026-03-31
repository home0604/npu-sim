#!/usr/bin/env python3
"""Main entry point for NPU simulator."""

import argparse
import json
import sys
from pathlib import Path

# Add project root (npu-sim) to path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from core.config import load_config
from sim.functional_check import format_diff_report
from sim.simulator import NPUSimulator


def main():
    parser = argparse.ArgumentParser(description="Systolic Array NPU Simulator")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_root / "configs" / "default.yaml"),
        help="Path to NPU config YAML file",
    )
    parser.add_argument("--workload", type=str, default="matmul", choices=["matmul", "attention", "transformer"])
    parser.add_argument("--M", type=int, default=256, help="M dimension for MatMul")
    parser.add_argument("--N", type=int, default=256, help="N dimension for MatMul")
    parser.add_argument("--K", type=int, default=256, help="K dimension for MatMul")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for attention/transformer")
    parser.add_argument("--hidden-dim", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Run functionality check: compare tile-by-tile output to reference (matmul/attention/transformer)",
    )
    parser.add_argument(
        "--use-dramsim3",
        action="store_true",
        help="Use DRAMSim3 for DRAM (requires build via scripts/setup_dramsim3.sh)",
    )
    parser.add_argument(
        "--dataflow",
        type=str,
        default=None,
        choices=["OS", "WS", "IS"],
        help="Override dataflow from config (OS, WS, or IS). If not set, uses config file.",
    )
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Enable double buffering (disabled by default)",
    )
    parser.add_argument(
        "--vq",
        action="store_true",
        help="Enable VQ mode (codebook + index → dequantized weight, OS dataflow)",
    )
    parser.add_argument("--codebook-size", type=int, default=None, help="VQ codebook size R (default: 16)")
    parser.add_argument("--vector-dim", type=int, default=None, help="VQ vector dimension d (default: 2)")
    parser.add_argument(
        "--use-scaling",
        action="store_true",
        help="Enable VQ per-group scaling (s, z)",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    config.double_buffer.enabled = args.use_db
    if args.use_dramsim3:
        config.dram.use_dramsim3 = True
    if args.vq:
        config.vq.enabled = True
    if args.codebook_size is not None:
        config.vq.codebook_size = args.codebook_size
        import math
        config.vq.index_bits = int(math.ceil(math.log2(args.codebook_size)))
    if args.vector_dim is not None:
        config.vq.vector_dim = args.vector_dim
    if args.use_scaling:
        config.vq.use_scaling = True
    if args.dataflow is not None:
        config.systolic.dataflow = args.dataflow
    sim = NPUSimulator(config)

    dram_backend = type(sim.dram).__name__
    print(f"NPU Config: {config.name}")
    print(f"  Array: {config.systolic.rows}x{config.systolic.cols}")
    print(f"  Dataflow: {config.systolic.dataflow}")
    print(f"  SRAM: {config.sram.total_size_kb}KB, {config.sram.num_banks} banks, {config.sram.port_type}-port")
    print(f"  DRAM: {dram_backend}")
    print(f"  Dtype: {config.dtype.compute_dtype}")
    print(f"  Double Buffer: {config.double_buffer.enabled}")
    if config.vq.enabled:
        print(f"  VQ: enabled (R={config.vq.codebook_size}, d={config.vq.vector_dim}, "
              f"{config.vq.index_bits}-bit index, scaling={'on' if config.vq.use_scaling else 'off'})")
    print()

    if args.workload == "matmul":
        print(f"Running MatMul: C[{args.M},{args.N}] = A[{args.M},{args.K}] x B[{args.K},{args.N}]")
        if args.check_correctness:
            print("  (functionality check enabled: output will be compared to reference)")
        result = sim.run_matmul(args.M, args.N, args.K, check_correctness=args.check_correctness)
    elif args.workload == "attention":
        print(f"Running Attention: seq_len={args.seq_len}, hidden={args.hidden_dim}, heads={args.num_heads}")
        if args.check_correctness:
            print("  (functionality check enabled: output will be compared to reference)")
        result = sim.run_attention(
            args.seq_len, args.hidden_dim, args.num_heads,
            check_correctness=args.check_correctness,
        )
    elif args.workload == "transformer":
        print(f"Running Transformer Layer: seq_len={args.seq_len}, hidden={args.hidden_dim}, heads={args.num_heads}")
        if args.check_correctness:
            print("  (functionality check enabled: output will be compared to reference)")
        result = sim.run_transformer_layer(
            args.seq_len, args.hidden_dim, args.num_heads,
            check_correctness=args.check_correctness,
        )

    print()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        sim.stats.print_summary()
        if args.check_correctness and "correctness" in result:
            print(f"  Correctness: {result['correctness']} - {result.get('correctness_message', '')}")
            if "correctness_report" in result:
                print(format_diff_report(result["correctness_report"]))


if __name__ == "__main__":
    main()
