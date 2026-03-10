"""GPTVQ tile scheduler with dequantization step and double buffering.

Execution flow per tile:
1. Load codebook (if first k-tile), indices, scales, activation from DRAM → SRAM
2. Dequantize: indices + codebook + scales → dequantized weight (analytical cycles)
3. Compute: dequantized_weight × activation → output (systolic array)
4. Store output if last K-tile

Double buffering overlaps (load + dequant) of next tile with compute of current tile.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from ..compute.dequant_unit import DequantizationUnit
from ..compute.systolic_array import OSSystolicArray
from ..core.stats import SimStats
from ..memory.double_buffer import DoubleBufferController
from ..memory.memory_controller import MemoryController
from ..memory.sram_buffers import SRAMBuffer
from .os_dataflow import GPTVQTileOp

if TYPE_CHECKING:
    from ..core.clock import SimulationEngine


class GPTVQScheduler:
    """Orchestrates GPTVQ tile execution with dequantization and double buffering.

    Flow for each tile:
    1. Load codebook (if needed), indices, scales, activation from DRAM → SRAM
    2. Dequantize: codebook lookup + optional scaling (analytical cycles)
    3. Compute: systolic array (analytical cycles)
    4. While computing, prefetch + dequant next tile into other buffer slot
    5. When compute done, store output if needed, swap buffers
    """

    def __init__(
        self,
        engine: SimulationEngine,
        systolic: OSSystolicArray,
        dequant_unit: DequantizationUnit,
        mem_ctrl: MemoryController,
        double_buf: DoubleBufferController,
        codebook_bufs: list[SRAMBuffer],
        index_bufs: list[SRAMBuffer],
        scale_bufs: list[SRAMBuffer],
        act_bufs: list[SRAMBuffer],
        output_buf: SRAMBuffer,
        stats: SimStats,
        codebook_base_dram: int = 0,
        index_base_dram: int = 0,
        scale_base_dram: int = 0,
        act_base_dram: int = 0,
        output_base_dram: int = 0,
    ):
        self.engine = engine
        self.systolic = systolic
        self.dequant_unit = dequant_unit
        self.mem_ctrl = mem_ctrl
        self.double_buf = double_buf
        self.codebook_bufs = codebook_bufs
        self.index_bufs = index_bufs
        self.scale_bufs = scale_bufs
        self.act_bufs = act_bufs
        self.output_buf = output_buf
        self.stats = stats
        self.codebook_base_dram = codebook_base_dram
        self.index_base_dram = index_base_dram
        self.scale_base_dram = scale_base_dram
        self.act_base_dram = act_base_dram
        self.output_base_dram = output_base_dram

    def execute_schedule(self, schedule: list[GPTVQTileOp], start_cycle: int = 0) -> int:
        """Execute a GPTVQ tile schedule and return total completion cycle.

        Uses double buffering: overlaps (load + dequant) with compute.
        - First tile: full latency exposed (load + dequant + compute)
        - Subsequent tiles: (load + dequant) overlapped with previous compute
        """
        if not schedule:
            return start_cycle

        tiles = deque(schedule)
        cycle = start_cycle
        use_double_buffer = len(self.index_bufs) >= 2 and len(self.act_bufs) >= 2

        if not use_double_buffer:
            return self._execute_single_buffer(tiles, cycle)

        self.double_buf.reset()

        # --- Load + dequant first tile (no overlap, full latency) ---
        first_tile = tiles.popleft()
        slot_idx = self.double_buf.compute_slot.value

        load_done, dequant_done = self._load_and_dequant(first_tile, slot_idx, cycle)
        act_done = self._load_activation(first_tile, slot_idx, cycle)
        data_ready_cycle = max(dequant_done, act_done)

        # Compute first tile
        cycles_info = self.systolic.analytical_cycles(
            first_tile.tile_m, first_tile.tile_n, first_tile.tile_k
        )
        compute_done_cycle = data_ready_cycle + cycles_info.total_cycles
        self._update_compute_stats(first_tile, cycles_info)
        self._update_dequant_stats(first_tile, dequant_done - load_done)

        # Store output if last K-tile
        if first_tile.store_output:
            compute_done_cycle = self._store_output(first_tile, compute_done_cycle)

        # --- Process remaining tiles with double buffering ---
        prev_compute_done = compute_done_cycle

        while tiles:
            current_tile = tiles.popleft()
            next_slot_idx = 1 - slot_idx

            # Prefetch: start loading + dequant at same time as prev compute start
            prefetch_start = data_ready_cycle

            load_done, dequant_done = self._load_and_dequant(
                current_tile, next_slot_idx, prefetch_start
            )
            act_done = self._load_activation(current_tile, next_slot_idx, prefetch_start)
            next_data_ready = max(dequant_done, act_done)

            # Wait for both: previous compute done AND next data ready
            compute_start = max(prev_compute_done, next_data_ready)

            if next_data_ready > prev_compute_done:
                stall = next_data_ready - prev_compute_done
                self.double_buf.prefetch_misses += 1
                self.double_buf.stall_cycles += stall
                self.stats.memory.double_buffer_misses += 1
                self.stats.memory.double_buffer_stall_cycles += stall
            else:
                self.double_buf.prefetch_hits += 1
                self.stats.memory.double_buffer_hits += 1

            # Compute current tile
            cycles_info = self.systolic.analytical_cycles(
                current_tile.tile_m, current_tile.tile_n, current_tile.tile_k
            )
            compute_done = compute_start + cycles_info.total_cycles
            self._update_compute_stats(current_tile, cycles_info)
            self._update_dequant_stats(
                current_tile, dequant_done - max(load_done, prefetch_start)
            )

            # Store output
            if current_tile.store_output:
                compute_done = self._store_output(current_tile, compute_done)

            data_ready_cycle = compute_start
            prev_compute_done = compute_done
            slot_idx = next_slot_idx

        cycle = prev_compute_done
        self.stats.total_cycles = max(self.stats.total_cycles, cycle)
        return cycle

    def _execute_single_buffer(self, tiles: deque[GPTVQTileOp], cycle: int) -> int:
        """Sequential execution without double buffering."""
        for tile in tiles:
            load_done, dequant_done = self._load_and_dequant(tile, 0, cycle)
            act_done = self._load_activation(tile, 0, cycle)
            data_ready = max(dequant_done, act_done)

            cycles_info = self.systolic.analytical_cycles(
                tile.tile_m, tile.tile_n, tile.tile_k
            )
            compute_done = data_ready + cycles_info.total_cycles
            self._update_compute_stats(tile, cycles_info)
            self._update_dequant_stats(tile, dequant_done - load_done)

            if tile.store_output:
                compute_done = self._store_output(tile, compute_done)

            cycle = compute_done

        self.stats.total_cycles = max(self.stats.total_cycles, cycle)
        return cycle

    def _load_and_dequant(
        self, tile: GPTVQTileOp, slot_idx: int, cycle: int
    ) -> tuple[int, int]:
        """Load codebook + indices + scales, then dequantize.

        Returns (load_done_cycle, dequant_done_cycle).
        Dequantization starts after codebook + indices (+ scales) are all loaded.
        """
        cb_done = cycle
        if tile.load_codebook:
            cb_done = self._load_codebook(tile, cycle)

        idx_done = self._load_indices(tile, slot_idx, cycle)

        scale_done = cycle
        if tile.load_scales:
            scale_done = self._load_scales(tile, slot_idx, cycle)

        # Dequant starts after all compressed weight data is available
        load_done = max(cb_done, idx_done, scale_done)

        # Dequantization cycles
        dequant_cycles = self.dequant_unit.dequant_cycles(tile.tile_m, tile.tile_k)
        dequant_done = load_done + dequant_cycles

        return load_done, dequant_done

    def _load_codebook(self, tile: GPTVQTileOp, cycle: int) -> int:
        """Load codebook from DRAM to SRAM."""
        dram_addr = self.codebook_base_dram + tile.codebook_dram_offset
        sram_buf = self.codebook_bufs[0]  # single slot
        d = self.dequant_unit.vector_dim
        size_bytes = self.dequant_unit.codebook_size * d * self.dequant_unit.config.codebook_entry_bytes

        done = self.mem_ctrl.load_from_dram(
            dram_addr, sram_buf.base_addr, size_bytes, cycle, "codebook"
        )
        self.stats.memory.dram_read_bytes += size_bytes
        self.stats.memory.dram_read_count += 1
        self.stats.gptvq.codebook_load_bytes += size_bytes
        return done

    def _load_indices(self, tile: GPTVQTileOp, slot_idx: int, cycle: int) -> int:
        """Load weight indices from DRAM to SRAM."""
        dram_addr = self.index_base_dram + tile.index_dram_offset
        sram_buf = self.index_bufs[min(slot_idx, len(self.index_bufs) - 1)]
        d = self.dequant_unit.vector_dim
        num_groups = math.ceil(tile.tile_k / d)
        size_bytes = tile.tile_m * num_groups * self.dequant_unit.config.index_elem_bytes

        done = self.mem_ctrl.load_from_dram(
            dram_addr, sram_buf.base_addr, size_bytes, cycle, "index"
        )
        self.stats.memory.dram_read_bytes += size_bytes
        self.stats.memory.dram_read_count += 1
        self.stats.gptvq.index_load_bytes += size_bytes
        return done

    def _load_scales(self, tile: GPTVQTileOp, slot_idx: int, cycle: int) -> int:
        """Load scaling factors (and zero points) from DRAM to SRAM."""
        dram_addr = self.scale_base_dram + tile.scale_dram_offset
        sram_buf = self.scale_bufs[min(slot_idx, len(self.scale_bufs) - 1)]
        d = self.dequant_unit.vector_dim
        num_groups = math.ceil(tile.tile_k / d)
        scale_b = self.dequant_unit.config.scale_bytes
        zp_b = self.dequant_unit.config.zero_point_bytes
        size_bytes = num_groups * (scale_b + zp_b)

        if size_bytes <= 0:
            return cycle

        done = self.mem_ctrl.load_from_dram(
            dram_addr, sram_buf.base_addr, size_bytes, cycle, "scale"
        )
        self.stats.memory.dram_read_bytes += size_bytes
        self.stats.memory.dram_read_count += 1
        self.stats.gptvq.scale_load_bytes += size_bytes
        return done

    def _load_activation(self, tile: GPTVQTileOp, slot_idx: int, cycle: int) -> int:
        """Load activation tile from DRAM to SRAM."""
        dram_addr = self.act_base_dram + tile.activation_dram_offset
        sram_buf = self.act_bufs[min(slot_idx, len(self.act_bufs) - 1)]
        size_bytes = tile.tile_k * tile.tile_n * self.systolic.dtype.num_bytes

        done = self.mem_ctrl.load_from_dram(
            dram_addr, sram_buf.base_addr, size_bytes, cycle, "activation"
        )
        self.stats.memory.dram_read_bytes += size_bytes
        self.stats.memory.dram_read_count += 1
        return done

    def _store_output(self, tile: GPTVQTileOp, cycle: int) -> int:
        """Store output tile from SRAM to DRAM."""
        dram_addr = self.output_base_dram + tile.output_dram_offset
        size_bytes = tile.tile_m * tile.tile_n * self.systolic.acc_dtype.num_bytes

        done = self.mem_ctrl.store_to_dram(
            self.output_buf.base_addr, dram_addr, size_bytes, cycle, "output"
        )
        self.stats.memory.dram_write_bytes += size_bytes
        self.stats.memory.dram_write_count += 1
        return done

    def _update_compute_stats(self, tile: GPTVQTileOp, cycles_info) -> None:
        self.stats.compute.total_mac_ops += tile.tile_m * tile.tile_n * tile.tile_k
        self.stats.compute.total_compute_cycles += cycles_info.total_cycles
        self.stats.compute.pipeline_fill_cycles += cycles_info.pipeline_fill_cycles
        self.stats.compute.pipeline_drain_cycles += cycles_info.pipeline_drain_cycles
        self.stats.compute.total_tiles_processed += 1

    def _update_dequant_stats(self, tile: GPTVQTileOp, dequant_cycles: int) -> None:
        d = self.dequant_unit.vector_dim
        num_vectors = tile.tile_m * math.ceil(tile.tile_k / d)
        self.stats.gptvq.total_dequant_cycles += max(0, dequant_cycles)
        self.stats.gptvq.total_vectors_dequantized += num_vectors
