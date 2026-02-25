#!/usr/bin/env python3
"""Main entry point for NPU simulator."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from npu_sim.core.config import load_config
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

    args = parser.parse_args()

    config = load_config(args.config)
    sim = NPUSimulator(config)

    print(f"NPU Config: {config.name}")
    print(f"  Array: {config.systolic.rows}x{config.systolic.cols}")
    print(f"  SRAM: {config.sram.total_size_kb}KB, {config.sram.num_banks} banks, {config.sram.port_type}-port")
    print(f"  Dtype: {config.dtype.compute_dtype}")
    print(f"  Double Buffer: {config.double_buffer.enabled}")
    print()

    if args.workload == "matmul":
        print(f"Running MatMul: C[{args.M},{args.N}] = A[{args.M},{args.K}] x B[{args.K},{args.N}]")
        result = sim.run_matmul(args.M, args.N, args.K)
    elif args.workload == "attention":
        print(f"Running Attention: seq_len={args.seq_len}, hidden={args.hidden_dim}, heads={args.num_heads}")
        result = sim.run_attention(args.seq_len, args.hidden_dim, args.num_heads)
    elif args.workload == "transformer":
        print(f"Running Transformer Layer: seq_len={args.seq_len}, hidden={args.hidden_dim}, heads={args.num_heads}")
        result = sim.run_transformer_layer(args.seq_len, args.hidden_dim, args.num_heads)

    print()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        sim.stats.print_summary()


if __name__ == "__main__":
    main()
