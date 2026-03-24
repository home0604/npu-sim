#!/usr/bin/env python3
"""Dequantization cycle sweep: bank-aware read latency across vector_dim and num_stages."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from compute.dequant_unit import DequantizationUnit
from core.config import load_config

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
VECTOR_DIMS = [32, 64, 128, 512, 1024]
NUM_STAGES  = [1, 2, 4]

# Tile size for cycle estimation
TILE_M = 32
TILE_K = 256


def main():
    config = load_config(_root / "configs" / "gptvq.yaml")
    sram = config.sram

    print(f"SRAM: {sram.num_banks} banks × {sram.bank_width_bytes}B = "
          f"{sram.num_banks * sram.bank_width_bytes * 8} bits total bandwidth/cycle")
    print(f"Tile: M={TILE_M}, K={TILE_K}\n")

    # Header
    col_w = [12, 12, 16, 16, 18]
    headers = ["vector_dim", "stages (S)", "read_latency", "num_vectors", "dequant_cycles"]
    header = "".join(h.rjust(w) for h, w in zip(headers, col_w))
    sep = "-" * sum(col_w)
    print(header)
    print(sep)

    for S in NUM_STAGES:
        for d in VECTOR_DIMS:
            config.gptvq.vector_dim = d
            config.gptvq.num_stages = S

            unit = DequantizationUnit(config.gptvq, sram)
            read_latency = unit._read_latency_per_lookup()
            num_vectors = TILE_M * -(-TILE_K // d)  # ceil division
            cycles = unit.dequant_cycles(TILE_M, TILE_K)

            row = [str(d), str(S), f"{read_latency} cyc", str(num_vectors), str(cycles)]
            print("".join(v.rjust(w) for v, w in zip(row, col_w)))

        print(sep)


if __name__ == "__main__":
    main()
