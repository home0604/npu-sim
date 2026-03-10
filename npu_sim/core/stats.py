from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ComputeStats:
    total_mac_ops: int = 0
    total_compute_cycles: int = 0
    total_tiles_processed: int = 0
    pipeline_fill_cycles: int = 0
    pipeline_drain_cycles: int = 0
    steady_state_cycles: int = 0


@dataclass
class MemoryStats:
    dram_read_bytes: int = 0
    dram_write_bytes: int = 0
    dram_read_count: int = 0
    dram_write_count: int = 0
    dram_total_latency_cycles: int = 0

    sram_read_bytes: int = 0
    sram_write_bytes: int = 0
    sram_read_count: int = 0
    sram_write_count: int = 0
    sram_bank_conflicts: int = 0
    sram_port_conflicts: int = 0

    double_buffer_hits: int = 0
    double_buffer_misses: int = 0
    double_buffer_stall_cycles: int = 0


@dataclass
class TileStats:
    tile_m: int = 0
    tile_n: int = 0
    tile_k: int = 0
    num_m_tiles: int = 0
    num_n_tiles: int = 0
    num_k_tiles: int = 0
    total_tiles: int = 0


@dataclass
class LayerStats:
    name: str = ""
    op_type: str = ""
    cycles: int = 0
    mac_ops: int = 0
    dram_bytes: int = 0


class SimStats:
    """Aggregate simulation statistics."""

    def __init__(self, array_rows: int = 32, array_cols: int = 32, clock_freq_mhz: int = 1000):
        self.array_rows = array_rows
        self.array_cols = array_cols
        self.clock_freq_mhz = clock_freq_mhz
        self.total_cycles: int = 0
        self.compute = ComputeStats()
        self.memory = MemoryStats()
        self.tile = TileStats()
        self.layer_stats: list[LayerStats] = []
        self._timeline: list[dict] = []

    def record_event(self, cycle: int, event: str, data: dict | None = None) -> None:
        self._timeline.append({"cycle": cycle, "event": event, **(data or {})})

    @property
    def compute_utilization(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        peak_ops = self.array_rows * self.array_cols * self.total_cycles
        return self.compute.total_mac_ops / peak_ops if peak_ops > 0 else 0.0

    @property
    def dram_bandwidth_utilization(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        total_bytes = self.memory.dram_read_bytes + self.memory.dram_write_bytes
        # peak_bw = bandwidth_gbps * 1e9 bytes/sec, cycles = total_cycles / freq_mhz * 1e6
        elapsed_sec = self.total_cycles / (self.clock_freq_mhz * 1e6)
        if elapsed_sec == 0:
            return 0.0
        actual_bw = total_bytes / elapsed_sec
        return actual_bw  # return raw BW, caller can compare to peak

    def summary(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "compute_utilization": f"{self.compute_utilization:.2%}",
            "compute": {
                "total_mac_ops": self.compute.total_mac_ops,
                "total_compute_cycles": self.compute.total_compute_cycles,
                "tiles_processed": self.compute.total_tiles_processed,
                "fill_cycles": self.compute.pipeline_fill_cycles,
                "drain_cycles": self.compute.pipeline_drain_cycles,
            },
            "memory": {
                "dram_read_bytes": self.memory.dram_read_bytes,
                "dram_write_bytes": self.memory.dram_write_bytes,
                "sram_bank_conflicts": self.memory.sram_bank_conflicts,
                "double_buffer_hits": self.memory.double_buffer_hits,
                "double_buffer_misses": self.memory.double_buffer_misses,
                "double_buffer_stall_cycles": self.memory.double_buffer_stall_cycles,
            },
            "tile": {
                "tile_m": self.tile.tile_m,
                "tile_n": self.tile.tile_n,
                "tile_k": self.tile.tile_k,
                "total_tiles": self.tile.total_tiles,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.summary(), indent=indent)

    def print_summary(self) -> None:
        s = self.summary()
        print("=" * 60)
        print(f"NPU Simulation Summary")
        print("=" * 60)
        print(f"  Total Cycles:          {s['total_cycles']:,}")
        print(f"  Compute Utilization:   {s['compute_utilization']}")
        print(f"  Total MACs:            {s['compute']['total_mac_ops']:,}")
        print(f"  Tiles Processed:       {s['compute']['tiles_processed']:,}")
        print(f"  Tile Size (M,N,K):     ({s['tile']['tile_m']}, {s['tile']['tile_n']}, {s['tile']['tile_k']})")
        print(f"  DRAM Read:             {s['memory']['dram_read_bytes']:,} bytes")
        print(f"  DRAM Write:            {s['memory']['dram_write_bytes']:,} bytes")
        print(f"  SRAM Bank Conflicts:   {s['memory']['sram_bank_conflicts']:,}")
        print(f"  DB Hits/Misses:        {s['memory']['double_buffer_hits']}/{s['memory']['double_buffer_misses']}")
        print(f"  DB Stall Cycles:       {s['memory']['double_buffer_stall_cycles']:,}")
        if s["memory"]["double_buffer_misses"] > 0 and s["memory"]["double_buffer_hits"] == 0:
            print("  (DB 0 hits: prefetch slower than compute → memory bound)")
        print("=" * 60)
