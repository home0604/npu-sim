#!/usr/bin/env python3
"""Single-tile cycle timeline: detailed Gantt chart comparing non-VQ vs VQ."""

import argparse
import sys
import math
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from core.config import load_config, NPUConfig
from sim.simulator import NPUSimulator

# ---------------------------------------------------------------------------
# Problem size (single tile)
# ---------------------------------------------------------------------------
M, N, K = 32, 32, 128


def _dram_channels(config: NPUConfig) -> int:
    """Parse channel count from DRAMSim3 config file."""
    import re
    cfg_path = getattr(config.dram, "config_file", "")
    try:
        with open(_root / cfg_path) as f:
            for line in f:
                m = re.match(r'channels\s*=\s*(\d+)', line.strip())
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return 1


# ---------------------------------------------------------------------------
# Timeline tracers
# ---------------------------------------------------------------------------

def trace_non_vq(config: NPUConfig) -> list[tuple[str, str, int, int]]:
    """Run non-VQ single tile, return [(layer, op, start, end), ...]."""
    sim = NPUSimulator(config)
    tc, schedule = sim.dataflow.generate_schedule(M, N, K)
    assert len(schedule) == 1
    tile = schedule[0]

    events = []
    cycle = 0

    # Weight + Activation: DRAM → SRAM (parallel via begin/end)
    size_w = tile.tile_m * tile.tile_k * sim.dtype.num_bytes
    size_a = tile.tile_k * tile.tile_n * sim.dtype.num_bytes
    w_addr = tile.weight_dram_offset
    a_addr = size_w + tile.activation_dram_offset  # separate address space

    w_token = sim.dram.begin_read(w_addr, size_w, cycle, "weight")
    a_token = sim.dram.begin_read(a_addr, size_a, cycle, "activation")

    w_dram_resp = sim.dram.end_read(w_token)
    a_dram_resp = sim.dram.end_read(a_token)

    w_sram_done = sim.buffers["weight"][0].write(0, size_w, w_dram_resp.complete_cycle)
    a_sram_done = sim.buffers["activation"][0].write(0, size_a, a_dram_resp.complete_cycle)

    # Single-channel: A waits for W (serial); multi-channel: both start at issue cycle
    a_draw_start = cycle if _dram_channels(config) > 1 else w_dram_resp.complete_cycle
    events.append(("DRAM", "W Read", cycle, w_dram_resp.complete_cycle))
    events.append(("SRAM", "W Write", w_dram_resp.complete_cycle, w_sram_done))
    events.append(("DRAM", "A Read", a_draw_start, a_dram_resp.complete_cycle))
    events.append(("SRAM", "A Write", a_dram_resp.complete_cycle, a_sram_done))
    data_ready = max(w_sram_done, a_sram_done)

    # Compute
    ci = sim.systolic.analytical_cycles(
        tile.tile_m, tile.tile_n, tile.tile_k, dataflow=config.systolic.dataflow
    )
    compute_done = data_ready + ci.total_cycles
    events.append(("NPU", "MAC", data_ready, compute_done))

    # Output: SRAM → DRAM
    size_o = tile.tile_m * tile.tile_n * sim.acc_dtype.num_bytes
    o_res = sim.mem_ctrl.store_to_dram(
        sim.buffers["output"][0], tile.output_dram_offset, size_o, compute_done, "output"
    )
    events.append(("DRAM", "O Write", compute_done, o_res.complete_cycle))

    return events


def trace_vq(config: NPUConfig) -> list[tuple[str, str, int, int]]:
    """Run VQ single tile, return [(layer, op, start, end), ...]."""
    sim = NPUSimulator(config)
    g = config.vq
    d = g.vector_dim

    tc, schedule = sim.vq_dataflow.generate_schedule(
        M, N, K, dataflow=config.systolic.dataflow
    )
    assert len(schedule) == 1
    tile = schedule[0]

    events = []
    cycle = 0

    # Codebook: DRAM → SRAM
    cb_size = int(g.codebook_size * d * g.codebook_entry_bytes)
    cb_addr = 0
    cb_res = sim.mem_ctrl.load_from_dram(
        cb_addr, sim.vq_buffers["codebook"][0], cb_size, cycle, "codebook"
    )
    events.append(("DRAM", "CB Read", cycle, cycle + cb_res.dram_cycles))
    events.append(("SRAM", "CB Write", cycle + cb_res.dram_cycles, cb_res.complete_cycle))
    cb_done = cb_res.complete_cycle

    # Index: DRAM → SRAM
    num_groups = math.ceil(tile.tile_k / d)
    idx_size = tile.tile_m * num_groups * g.index_elem_bytes
    idx_addr = cb_size
    idx_res = sim.mem_ctrl.load_from_dram(
        idx_addr, sim.vq_buffers["index"][0], idx_size, cb_done, "index"
    )
    events.append(("DRAM", "Idx Read", cb_done, cb_done + idx_res.dram_cycles))
    events.append(("SRAM", "Idx Write", cb_done + idx_res.dram_cycles, idx_res.complete_cycle))
    load_done = idx_res.complete_cycle

    # Dequant (SRAM internal: codebook read → dequant_weight write, pipelined)
    dq_cycles = sim.dequant_unit.dequant_cycles(tile.tile_m, tile.tile_k)
    dequant_done = load_done + dq_cycles
    events.append(("SRAM", "Dequant", load_done, dequant_done))

    # Activation: DRAM → SRAM (parallel with dequant — DRAM bus is free)
    size_a = tile.tile_k * tile.tile_n * sim.dtype.num_bytes
    a_addr = cb_size + idx_size
    a_res = sim.mem_ctrl.load_from_dram(
        a_addr, sim.vq_buffers["activation"][0], size_a, load_done, "activation"
    )
    events.append(("DRAM", "A Read", load_done, load_done + a_res.dram_cycles))
    events.append(("SRAM", "A Write", load_done + a_res.dram_cycles, a_res.complete_cycle))

    data_ready = max(dequant_done, a_res.complete_cycle)

    # Compute
    ci = sim.systolic.analytical_cycles(
        tile.tile_m, tile.tile_n, tile.tile_k, dataflow=config.systolic.dataflow
    )
    compute_done = data_ready + ci.total_cycles
    events.append(("NPU", "MAC", data_ready, compute_done))

    # Output: SRAM → DRAM
    size_o = tile.tile_m * tile.tile_n * sim.acc_dtype.num_bytes
    o_res = sim.mem_ctrl.store_to_dram(
        sim.vq_buffers["output"][0], 0, size_o, compute_done, "output"
    )
    events.append(("DRAM", "O Write", compute_done, o_res.complete_cycle))

    return events


# ---------------------------------------------------------------------------
# ASCII Gantt chart
# ---------------------------------------------------------------------------

LAYER_ORDER = ["NPU", "SRAM", "DRAM"]
SYMBOLS = {
    "W Read": "W", "W Write": "W",
    "A Read": "A", "A Write": "A",
    "O Write": "O", "O Read": "O",
    "CB Read": "C", "CB Write": "C",
    "Idx Read": "I", "Idx Write": "I",
    "Dequant": "D",
    "MAC": "M",
}


def render_gantt(title: str, events: list[tuple[str, str, int, int]], width: int = 72):
    max_cycle = max(e[3] for e in events)
    if max_cycle == 0:
        return

    scale = (width - 1) / max_cycle

    # Tick marks
    tick_interval = max(1, max_cycle // 5)
    ticks = list(range(0, max_cycle + 1, tick_interval))
    if ticks[-1] != max_cycle:
        ticks.append(max_cycle)

    tick_line = [" "] * width
    tick_label = [" "] * width
    for t in ticks:
        pos = min(int(t * scale), width - 1)
        tick_line[pos] = "|"
        label = str(t)
        lpos = max(0, min(pos - len(label) // 2, width - len(label)))
        for i, ch in enumerate(label):
            if lpos + i < width:
                tick_label[lpos + i] = ch

    print(f"\n  {title}  (total = {max_cycle} cycles)")
    print(f"  {'':6s} {''.join(tick_label)}")
    print(f"  {'':6s} {''.join(tick_line)}")

    for layer in LAYER_ORDER:
        layer_events = [(op, s, e) for (l, op, s, e) in events if l == layer]
        if not layer_events:
            continue

        # Group overlapping ops into separate sub-rows
        rows: list[list[tuple[str, int, int]]] = []
        for op, start, end in layer_events:
            placed = False
            for row in rows:
                if all(end <= rs or start >= re for _, rs, re in row):
                    row.append((op, start, end))
                    placed = True
                    break
            if not placed:
                rows.append([(op, start, end)])

        for ri, row in enumerate(rows):
            line = [" "] * width
            for op, start, end in row:
                c_s = int(start * scale)
                c_e = max(c_s + 1, int(end * scale))
                c_e = min(c_e, width)
                sym = SYMBOLS.get(op, "#")
                for i in range(c_s, c_e):
                    line[i] = sym
            label = layer if ri == 0 else ""
            print(f"  {label:5s} |{''.join(line)}|")

    # Legend for this chart
    used_ops = set(op for _, op, _, _ in events)
    legend_parts = []
    for op in used_ops:
        sym = SYMBOLS.get(op, "#")
        legend_parts.append(f"{sym}={op}")
    print(f"  {'':6s} {', '.join(sorted(legend_parts))}")
    print()


def print_detail_table(events: list[tuple[str, str, int, int]]):
    max_cycle = max(e[3] for e in events) if events else 0
    col_w = [7, 12, 8, 8, 8, 7]
    headers = ["Layer", "Operation", "Start", "End", "Dur", "%"]
    print("  " + "".join(h.rjust(w) for h, w in zip(headers, col_w)))
    print("  " + "-" * sum(col_w))

    for layer, op, start, end in events:
        dur = end - start
        pct = dur / max_cycle * 100 if max_cycle > 0 else 0
        row = [layer, op, str(start), str(end), str(dur), f"{pct:.1f}%"]
        print("  " + "".join(v.rjust(w) for v, w in zip(row, col_w)))
    print()


# ---------------------------------------------------------------------------
# Matplotlib figure
# ---------------------------------------------------------------------------

OP_COLORS = {
    "W Read": "#4A90D9", "W Write": "#4A90D9",
    "A Read": "#7BC47F", "A Write": "#7BC47F",
    "O Write": "#E8A838", "O Read": "#E8A838",
    "CB Read": "#9B59B6", "CB Write": "#9B59B6",
    "Idx Read": "#E74C3C", "Idx Write": "#E74C3C",
    "Dequant": "#F39C12",
    "MAC": "#2ECC71",
}


def plot_gantt(events_dict: dict[str, list], save_path: str):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_charts = len(events_dict)
    fig, axes = plt.subplots(n_charts, 1, figsize=(14, 3.5 * n_charts), squeeze=False)

    for ax_idx, (title, events) in enumerate(events_dict.items()):
        ax = axes[ax_idx, 0]
        max_cycle = max(e[3] for e in events)

        # Build sub-rows per layer to handle overlapping ops
        bar_height = 0.35
        sub_row_gap = 0.4
        y_positions = {}  # layer → base y
        current_y = 0
        for layer in reversed(LAYER_ORDER):
            layer_events = [(op, s, e) for (l, op, s, e) in events if l == layer]
            if not layer_events:
                continue
            # Split into non-overlapping rows
            rows: list[list[tuple[str, int, int]]] = []
            for op, start, end in layer_events:
                placed = False
                for row in rows:
                    if all(end <= rs or start >= re for _, rs, re in row):
                        row.append((op, start, end))
                        placed = True
                        break
                if not placed:
                    rows.append([(op, start, end)])
            y_positions[layer] = (current_y, rows)
            current_y += len(rows) * sub_row_gap + 0.3

        for layer, (base_y, rows) in y_positions.items():
            for ri, row in enumerate(rows):
                y = base_y + ri * sub_row_gap
                for op, start, end in row:
                    dur = end - start
                    color = OP_COLORS.get(op, "#888888")
                    ax.barh(y, dur, left=start, height=bar_height, color=color,
                            edgecolor="white", linewidth=0.5)
                    if dur / max_cycle > 0.04:
                        ax.text(start + dur / 2, y, f"{op}\n({dur})",
                                ha="center", va="center", fontsize=7,
                                fontweight="bold", color="white")

        # Y-axis labels at center of each layer's sub-rows
        yticks, ylabels = [], []
        for layer, (base_y, rows) in y_positions.items():
            center = base_y + (len(rows) - 1) * sub_row_gap / 2
            yticks.append(center)
            ylabels.append(layer)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=11, fontweight="bold")
        ax.set_xlim(0, max_cycle * 1.02)
        ax.set_xlabel("Cycles", fontsize=10)
        ax.set_title(f"{title}  (total = {max_cycle} cycles, tile = {M}x{N}x{K})",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    # Legend (group by color to merge Read/Write of same operand)
    LEGEND_LABELS = {
        "#4A90D9": "Weight", "#7BC47F": "Activation", "#E8A838": "Output",
        "#9B59B6": "Codebook", "#E74C3C": "Index", "#F39C12": "Dequant",
        "#2ECC71": "MAC",
    }
    used_colors = set()
    for events in events_dict.values():
        for _, op, _, _ in events:
            used_colors.add(OP_COLORS.get(op, "#888"))
    handles = [mpatches.Patch(color=c, label=LEGEND_LABELS.get(c, c))
               for c in LEGEND_LABELS if c in used_colors]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _config_tag(config: NPUConfig) -> str:
    """Extract tag from DRAM config (e.g. 'HBM2_8Gb', 'DDR4_8Gb')."""
    cfg_path = getattr(config.dram, "config_file", "")
    if cfg_path:
        stem = Path(cfg_path).stem  # e.g. "HBM2_8Gb_x128"
        parts = stem.split("_")
        # Take type + size (first two parts): "HBM2_8Gb"
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0]
    return "simple"


def main():
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Single-tile cycle breakdown")
    parser.add_argument("--config", type=str,
                        default=str(_root / "configs" / "vq.yaml"),
                        help="VQ config path (default: configs/vq.yaml)")
    parser.add_argument("--plot", action="store_true", help="Save Gantt chart figure")
    args = parser.parse_args()

    configs_dir = _root / "configs"

    print(f"Single-tile cycle timeline: M={M}, N={N}, K={K}")
    print("=" * 80)

    # Non-VQ
    cfg_novq = load_config(configs_dir / "default.yaml")
    events_novq = trace_non_vq(cfg_novq)
    render_gantt("non-VQ", events_novq)
    print_detail_table(events_novq)

    # VQ — derive label from config filename
    vq_config_path = Path(args.config)
    vq_label = vq_config_path.stem.replace("_", " ").upper()  # e.g. "GPTVQ 3.125BPV"
    cfg_vq = load_config(vq_config_path)
    events_vq = trace_vq(cfg_vq)
    render_gantt(vq_label, events_vq)
    print_detail_table(events_vq)

    # Comparison
    t_novq = max(e[3] for e in events_novq)
    t_vq = max(e[3] for e in events_vq)
    diff = t_vq - t_novq
    overhead = diff / t_novq * 100 if t_novq > 0 else 0

    print("=" * 80)
    print(f"  non-VQ:      {t_novq:,} cycles")
    print(f"  {vq_label}:  {t_vq:,} cycles  (overhead: {diff:+,}, {overhead:+.1f}%)")
    print()

    # Save results
    file_ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    dram_tag = _config_tag(cfg_novq)
    vq_tag = vq_config_path.stem
    out_dir = _root / "reports" / "tile_breakdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot:
        png_path = out_dir / f"{file_ts}_{dram_tag}_{vq_tag}.png"
        plot_gantt({f"Cycle (non-VQ)": events_novq, f"Cycle ({vq_label})": events_vq}, str(png_path))


if __name__ == "__main__":
    main()
