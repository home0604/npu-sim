from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from compute.systolic_array import SystolicArray
from core.events import EventType
from core.stats import SimStats
from memory.double_buffer import BufferSlot, DoubleBufferController, SlotState
from memory.memory_controller import MemoryController
from memory.sram_buffers import SRAMBuffer
from dataflow.stationary import TileOp

if TYPE_CHECKING:
    from core.clock import SimulationEngine


class TileScheduler:
    """Orchestrates tile execution with double buffering.

    Flow for each tile:
    1. Load weight tile from DRAM -> SRAM (if load_weight)
    2. Load activation tile from DRAM -> SRAM
    3. When both ready, start systolic array compute (analytical cycles)
    4. While computing, prefetch next tile into the other buffer slot
    5. When compute done, store output if needed, swap buffers
    """

    def __init__(
        self,
        engine: SimulationEngine,
        systolic: SystolicArray,
        mem_ctrl: MemoryController,
        double_buf: DoubleBufferController,
        weight_bufs: list[SRAMBuffer],
        act_bufs: list[SRAMBuffer],
        output_buf: SRAMBuffer,
        stats: SimStats,
        weight_base_dram: int = 0,
        act_base_dram: int = 0,
        output_base_dram: int = 0,
        dataflow: str = "OS",
    ):
        self.engine = engine
        self.systolic = systolic
        self.mem_ctrl = mem_ctrl
        self.double_buf = double_buf
        self.weight_bufs = weight_bufs
        self.act_bufs = act_bufs
        self.output_buf = output_buf
        self.stats = stats
        self.weight_base_dram = weight_base_dram
        self.act_base_dram = act_base_dram
        self.output_base_dram = output_base_dram
        self.dataflow = dataflow

    def execute_schedule(self, schedule: list[TileOp], start_cycle: int = 0) -> int:
        """Execute a tile schedule and return total completion cycle.

        Uses a sequential approach with double buffering overlap:
        - First tile: full load latency exposed
        - Subsequent tiles: compute overlaps with next tile's prefetch
        """
        if not schedule:
            return start_cycle

        tiles = deque(schedule)
        cycle = start_cycle
        use_double_buffer = len(self.weight_bufs) >= 2 and len(self.act_bufs) >= 2

        if not use_double_buffer:
            return self._execute_single_buffer(tiles, cycle)

        self.double_buf.reset()

        # --- Load first tile (no overlap, full latency exposed) ---
        first_tile = tiles.popleft()
        slot_idx = self.double_buf.compute_slot.value

        weight_done = cycle
        if first_tile.load_weight:
            weight_done = self._load_weight(first_tile, slot_idx, cycle)

        act_done = self._load_activation(first_tile, slot_idx, cycle)
        data_ready_cycle = max(weight_done, act_done)

        # Compute first tile
        cycles_info = self.systolic.analytical_cycles(
            first_tile.tile_m, first_tile.tile_n, first_tile.tile_k,
            dataflow=self.dataflow,
        )
        compute_done_cycle = data_ready_cycle + cycles_info.total_cycles
        self._update_compute_stats(first_tile, cycles_info)

        # Store output if this is the last K-tile
        if first_tile.store_output:
            compute_done_cycle = self._store_output(first_tile, compute_done_cycle)

        # --- Process remaining tiles with double buffering ---
        prev_compute_done = compute_done_cycle

        while tiles:
            current_tile = tiles.popleft()
            next_slot_idx = 1 - slot_idx  # alternate between 0 and 1

            # Prefetch: start loading next tile at the same time as previous compute
            prefetch_start = data_ready_cycle  # start prefetch when prev compute starts

            weight_done = prefetch_start
            if current_tile.load_weight:
                weight_done = self._load_weight(current_tile, next_slot_idx, prefetch_start)

            act_done = self._load_activation(current_tile, next_slot_idx, prefetch_start)
            next_data_ready = max(weight_done, act_done)

            # Wait for both: previous compute done AND next data ready
            compute_start = max(prev_compute_done, next_data_ready)

            if next_data_ready > prev_compute_done:
                # Stall: prefetch didn't finish in time
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
                current_tile.tile_m, current_tile.tile_n, current_tile.tile_k,
                dataflow=self.dataflow,
            )
            compute_done = compute_start + cycles_info.total_cycles
            self._update_compute_stats(current_tile, cycles_info)

            # Store output
            if current_tile.store_output:
                compute_done = self._store_output(current_tile, compute_done)

            # Update for next iteration
            data_ready_cycle = compute_start
            prev_compute_done = compute_done
            slot_idx = next_slot_idx

        cycle = prev_compute_done
        self.stats.total_cycles = max(self.stats.total_cycles, cycle)
        return cycle

    def _execute_single_buffer(self, tiles: deque[TileOp], cycle: int) -> int:
        """Simple sequential execution without double buffering."""
        for tile in tiles:
            weight_done = cycle
            if tile.load_weight:
                weight_done = self._load_weight(tile, 0, cycle)

            act_done = self._load_activation(tile, 0, cycle)
            data_ready = max(weight_done, act_done)

            cycles_info = self.systolic.analytical_cycles(
                tile.tile_m, tile.tile_n, tile.tile_k, dataflow=self.dataflow,
            )
            compute_done = data_ready + cycles_info.total_cycles
            self._update_compute_stats(tile, cycles_info)

            if tile.store_output:
                compute_done = self._store_output(tile, compute_done)

            cycle = compute_done

        self.stats.total_cycles = max(self.stats.total_cycles, cycle)
        return cycle

    def _load_weight(self, tile: TileOp, slot_idx: int, cycle: int) -> int:
        """Load weight tile from DRAM to SRAM buffer."""
        dram_addr = self.weight_base_dram + tile.weight_dram_offset
        sram_buf = self.weight_bufs[slot_idx]
        size_bytes = tile.tile_m * tile.tile_k * self.systolic.dtype.num_bytes

        result = self.mem_ctrl.load_from_dram(dram_addr, sram_buf.base_addr, size_bytes, cycle, "weight")
        self.stats.memory.dram_read_bytes += size_bytes
        self.stats.memory.dram_read_count += 1
        self.stats.memory.dram_read_cycles += result.dram_cycles
        self.stats.memory.sram_write_cycles += result.sram_cycles
        return result.complete_cycle

    def _load_activation(self, tile: TileOp, slot_idx: int, cycle: int) -> int:
        """Load activation tile from DRAM to SRAM buffer."""
        dram_addr = self.act_base_dram + tile.activation_dram_offset
        sram_buf = self.act_bufs[slot_idx]
        size_bytes = tile.tile_k * tile.tile_n * self.systolic.dtype.num_bytes

        result = self.mem_ctrl.load_from_dram(dram_addr, sram_buf.base_addr, size_bytes, cycle, "activation")
        self.stats.memory.dram_read_bytes += size_bytes
        self.stats.memory.dram_read_count += 1
        self.stats.memory.dram_read_cycles += result.dram_cycles
        self.stats.memory.sram_write_cycles += result.sram_cycles
        return result.complete_cycle

    def _store_output(self, tile: TileOp, cycle: int) -> int:
        """Store output tile from SRAM to DRAM."""
        dram_addr = self.output_base_dram + tile.output_dram_offset
        size_bytes = tile.tile_m * tile.tile_n * self.systolic.acc_dtype.num_bytes

        result = self.mem_ctrl.store_to_dram(self.output_buf.base_addr, dram_addr, size_bytes, cycle, "output")
        self.stats.memory.dram_write_bytes += size_bytes
        self.stats.memory.dram_write_count += 1
        self.stats.memory.sram_read_cycles += result.sram_cycles
        self.stats.memory.dram_write_cycles += result.dram_cycles
        return result.complete_cycle

    def _update_compute_stats(self, tile: TileOp, cycles_info) -> None:
        self.stats.compute.total_mac_ops += tile.tile_m * tile.tile_n * tile.tile_k
        self.stats.compute.total_compute_cycles += cycles_info.total_cycles
        self.stats.compute.pipeline_fill_cycles += cycles_info.pipeline_fill_cycles
        self.stats.compute.pipeline_drain_cycles += cycles_info.pipeline_drain_cycles
        self.stats.compute.total_tiles_processed += 1
