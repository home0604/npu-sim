#!/usr/bin/env python3
"""Benchmark script: run multiple configurations and display grouped comparison tables."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sim(cli_args: list[str]) -> dict | None:
    cmd = [
        sys.executable, str(_root / "scripts" / "run_sim.py"),
        "--json", *cli_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}", file=sys.stderr)
        return None

    lines = result.stdout.strip().split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            try:
                return json.loads("\n".join(lines[i:]))
            except json.JSONDecodeError:
                return None
    return None


def run_matmul(M, N, K, df, db=False) -> dict | None:
    args = [
        "--workload", "matmul",
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--dataflow", df,
    ]
    if db:
        args.append("--use-db")
    return run_sim(args)


def extract_row(data: dict, label: str = "") -> dict:
    return {
        "label": label,
        "tiles": data["tile"]["total_tiles"],
        "total_cycles": data["total_cycles"],
        "util": data["compute_utilization"],
        "dram_r": data["memory"]["dram_read_bytes"],
        "dram_w": data["memory"]["dram_write_bytes"],
        "dram_r_cycles": data["memory"]["dram_read_cycles"],
        "db_hits": data["memory"]["double_buffer_hits"],
        "db_misses": data["memory"]["double_buffer_misses"],
        "db_stalls": data["memory"]["double_buffer_stall_cycles"],
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def write_line(out: list[str], text: str):
    """Append line to output buffer and print to terminal."""
    out.append(text)
    print(text)


def write_table(out: list[str], title: str, headers: list[str], rows: list[list[str]],
                col_widths: list[int] | None = None):
    """Write a formatted table."""
    if not col_widths:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for row in rows:
                if i < len(row):
                    w = max(w, len(row[i]))
            col_widths.append(w + 2)

    total_w = sum(col_widths)
    write_line(out, "")
    write_line(out, "=" * total_w)
    write_line(out, f"  {title}")
    write_line(out, "=" * total_w)

    header_str = "".join(h.rjust(w) for h, w in zip(headers, col_widths))
    write_line(out, header_str)
    write_line(out, "-" * total_w)

    for row in rows:
        row_str = "".join(
            (row[i] if i < len(row) else "").rjust(col_widths[i])
            for i in range(len(headers))
        )
        write_line(out, row_str)

    write_line(out, "-" * total_w)


# ---------------------------------------------------------------------------
# Benchmark groups
# ---------------------------------------------------------------------------

def group_dataflow_comparison(out: list[str], cache: dict):
    """Group 1: Dataflow comparison (OS vs WS vs IS) for each size."""
    sizes = [(32, 32, 32), (64, 64, 64), (256, 256, 256)]

    headers = ["Size", "Dataflow", "Tiles", "Total Cycles", "Util", "DRAM Read", "DRAM Write"]
    rows = []

    for M, N, K in sizes:
        for df in ["OS", "WS", "IS"]:
            key = (M, N, K, df, False)
            if key not in cache:
                data = run_matmul(M, N, K, df)
                if data:
                    cache[key] = data
            data = cache.get(key)
            if not data:
                continue

            r = extract_row(data)
            rows.append([
                f"{M}x{N}x{K}",
                df,
                str(r["tiles"]),
                f"{r['total_cycles']:,}",
                r["util"],
                f"{r['dram_r']:,}",
                f"{r['dram_w']:,}",
            ])

    write_table(out, "Group 1: Dataflow Comparison (DB=OFF)", headers, rows)


def group_size_scaling(out: list[str], cache: dict):
    """Group 2: Size scaling for WS dataflow."""
    sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]

    headers = ["Size", "Tiles", "Total Cycles", "Util", "DRAM Read", "MACs"]
    rows = []

    for M, N, K in sizes:
        key = (M, N, K, "WS", False)
        if key not in cache:
            data = run_matmul(M, N, K, "WS")
            if data:
                cache[key] = data
        data = cache.get(key)
        if not data:
            continue

        r = extract_row(data)
        rows.append([
            f"{M}x{N}x{K}",
            str(r["tiles"]),
            f"{r['total_cycles']:,}",
            r["util"],
            f"{r['dram_r']:,}",
            f"{data['compute']['total_mac_ops']:,}",
        ])

    write_table(out, "Group 2: Size Scaling (WS, DB=OFF)", headers, rows)


def group_db_comparison(out: list[str], cache: dict):
    """Group 3: Double Buffer ON vs OFF."""
    sizes = [(64, 64, 64), (256, 256, 256)]

    headers = ["Size", "DB", "Total Cycles", "Util", "DB Hits/Miss", "Stalls"]
    rows = []

    for M, N, K in sizes:
        for db in [False, True]:
            key = (M, N, K, "WS", db)
            if key not in cache:
                data = run_matmul(M, N, K, "WS", db=db)
                if data:
                    cache[key] = data
            data = cache.get(key)
            if not data:
                continue

            r = extract_row(data)
            rows.append([
                f"{M}x{N}x{K}",
                "ON" if db else "OFF",
                f"{r['total_cycles']:,}",
                r["util"],
                f"{r['db_hits']}/{r['db_misses']}",
                f"{r['db_stalls']:,}",
            ])

    write_table(out, "Group 3: Double Buffer ON vs OFF (WS)", headers, rows)


def group_k_scaling(out: list[str], cache: dict):
    """Group 4: K scaling (M=N=64, K varies)."""
    k_values = [64, 256, 1024]

    headers = ["K", "Dataflow", "Tiles", "Total Cycles", "Util", "DRAM Read"]
    rows = []

    for K in k_values:
        for df in ["OS", "WS"]:
            key = (64, 64, K, df, False)
            if key not in cache:
                data = run_matmul(64, 64, K, df)
                if data:
                    cache[key] = data
            data = cache.get(key)
            if not data:
                continue

            r = extract_row(data)
            rows.append([
                str(K),
                df,
                str(r["tiles"]),
                f"{r['total_cycles']:,}",
                r["util"],
                f"{r['dram_r']:,}",
            ])

    write_table(out, "Group 4: K Scaling (M=N=64, DB=OFF)", headers, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_ts = datetime.now().strftime("%Y-%m-%d_%H%M")

    out = []  # output buffer for txt file
    cache = {}  # (M,N,K,df,db) -> result, avoid duplicate runs

    write_line(out, f"NPU-Sim Benchmark  ({timestamp})")
    write_line(out, f"Config: 32x32 array, FP16, HBM2")

    # Count total unique runs needed
    print("\nRunning simulations...", flush=True)

    # Run all groups
    group_dataflow_comparison(out, cache)
    group_size_scaling(out, cache)
    group_db_comparison(out, cache)
    group_k_scaling(out, cache)

    # Save files
    out_dir = _root / "reports" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save txt (human-readable)
    txt_file = out_dir / f"{file_ts}_hbm2.txt"
    with open(txt_file, "w") as f:
        f.write("\n".join(out) + "\n")

    # Save json (programmatic)
    all_results = []
    for (M, N, K, df, db), data in cache.items():
        all_results.append({
            "config": {"M": M, "N": N, "K": K, "dataflow": df, "db": db},
            "result": data,
        })
    json_file = out_dir / f"{file_ts}_hbm2.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)

    write_line(out, f"\nSaved: {txt_file.name}, {json_file.name}")


if __name__ == "__main__":
    main()
