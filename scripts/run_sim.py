#!/usr/bin/env python3
"""Main entry point for NPU simulator."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from npu_sim.core.config import load_config
from npu_sim.sim.functional_check import format_diff_report
from npu_sim.sim.simulator import NPUSimulator


def main():
    parser = argparse.ArgumentParser(description="Systolic Array NPU Simulator")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "default.yaml"),
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
        "--gptvq",
        action="store_true",
        help="Enable GPTVQ mode (codebook + index → dequantized weight, OS dataflow)",
    )
    parser.add_argument("--codebook-size", type=int, default=None, help="GPTVQ codebook size R (default: 16)")
    parser.add_argument("--vector-dim", type=int, default=None, help="GPTVQ vector dimension d (default: 2)")
    parser.add_argument(
        "--use-scaling",
        action="store_true",
        help="Enable GPTVQ per-group scaling (s, z)",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.use_dramsim3:
        config.dram.use_dramsim3 = True
    if args.gptvq:
        config.gptvq.enabled = True
    if args.codebook_size is not None:
        config.gptvq.codebook_size = args.codebook_size
        import math
        config.gptvq.index_bits = int(math.ceil(math.log2(args.codebook_size)))
    if args.vector_dim is not None:
        config.gptvq.vector_dim = args.vector_dim
    if args.use_scaling:
        config.gptvq.use_scaling = True
    sim = NPUSimulator(config)

    dram_backend = type(sim.dram).__name__
    print(f"NPU Config: {config.name}")
    print(f"  Array: {config.systolic.rows}x{config.systolic.cols}")
    print(f"  SRAM: {config.sram.total_size_kb}KB, {config.sram.num_banks} banks, {config.sram.port_type}-port")
    print(f"  DRAM: {dram_backend}")
    print(f"  Dtype: {config.dtype.compute_dtype}")
    print(f"  Double Buffer: {config.double_buffer.enabled}")
    if config.gptvq.enabled:
        print(f"  GPTVQ: enabled (R={config.gptvq.codebook_size}, d={config.gptvq.vector_dim}, "
              f"{config.gptvq.index_bits}-bit index, scaling={'on' if config.gptvq.use_scaling else 'off'})")
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
