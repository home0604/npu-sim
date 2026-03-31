#!/usr/bin/env python3
"""Dequantization cycle sweep: bank-aware read latency across vec_dim, stages, cb_banks, bank_width."""

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from compute.dequant_unit import DequantizationUnit
from core.config import load_config

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
VECTOR_DIMS    = [8, 32, 64]      # d: codebook vector dimension
NUM_STAGES     = [1, 2]      # S: RVQ stages
CODEBOOK_BANKS = [4, 16, 32]    # N_bank: codebook SRAM banks
BANK_WIDTHS    = [8, 32, 64]    # w_b: bytes per bank per cycle

# Tile size for cycle estimation
# TILE_M and TILE_K determine dequant cycles (N has no effect on dequant).
TILE_M = 32
TILE_N = 32     # to compute reference line 
TILE_K = 256


def main():
    parser = argparse.ArgumentParser(description="Dequantization cycle sweep")
    parser.add_argument("--config", type=str,
                        default=str(_root / "configs" / "vq.yaml"),
                        help="YAML config path (default: configs/vq.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    sram = config.sram

    print(f"Tile: M={TILE_M}, K={TILE_K}\n")

    col_w = [12, 12, 18, 14, 14]
    headers = ["vec_dim (d)", "stages (S)", "cyc/lookup", "num_lookups", "total_cycles"]
    header = "  " + "".join(h.rjust(w) for h, w in zip(headers, col_w))
    sep    = "  " + "-" * sum(col_w)

    for bw in BANK_WIDTHS:
        sram.bank_width_bytes = bw
        print(f"bank_width = {bw}B")

        for n_banks in CODEBOOK_BANKS:
            sram.codebook_sram.num_banks = n_banks
            bw_bits = n_banks * bw * 8
            print(f"  cb_banks = {n_banks:2d}  ({bw_bits:5d} bits/cycle)")
            print(header)
            print(sep)

            for S in NUM_STAGES:
                for d in VECTOR_DIMS:
                    config.vq.vector_dim = d
                    config.vq.num_stages = S

                    unit = DequantizationUnit(config.vq, sram)
                    latency   = unit._latency_per_lookup()
                    num_lookups  = TILE_M * -(-TILE_K // d)
                    cycles       = unit.dequant_cycles(TILE_M, TILE_K)

                    row = [str(d), str(S), str(latency), str(num_lookups), str(cycles)]
                    print("  " + "".join(v.rjust(w) for v, w in zip(row, col_w)))

                if S != NUM_STAGES[-1]:
                    print()

            print()


if __name__ == "__main__":
    main()
