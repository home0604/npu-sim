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

        weight_done, act_done = self._load_tiles_parallel(first_tile, slot_idx, cycle)
        data_ready_cycle = max(weight_done, act_done)

        # Compute first tile
        cycles_info = self.systolic.analytical_cycles(
            first_tile.tile_m, first_tile.tile_n, first_tile.tile_k,
            dataflow=self.dataflow,
        )
        compute_done_cycle = data_ready_cycle + cycles_info.total_cycles
        self._update_compute_stats(first_tile, cycles_info)

        if self.dataflow != "OS" and (first_tile.accumulate or not first_tile.store_output):
            compute_done_cycle = self._accumulate_output(first_tile, compute_done_cycle)

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

            weight_done, act_done = self._load_tiles_parallel(
                current_tile, next_slot_idx, prefetch_start)
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

            if self.dataflow != "OS" and (current_tile.accumulate or not current_tile.store_output):
                compute_done = self._accumulate_output(current_tile, compute_done)

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
            weight_done, act_done = self._load_tiles_parallel(tile, 0, cycle)
            data_ready_cycle = max(weight_done, act_done)

            cycles_info = self.systolic.analytical_cycles(
                tile.tile_m, tile.tile_n, tile.tile_k, dataflow=self.dataflow,
            )
            compute_done = data_ready_cycle + cycles_info.total_cycles
            self._update_compute_stats(tile, cycles_info)

            if self.dataflow != "OS" and (tile.accumulate or not tile.store_output):
                compute_done = self._accumulate_output(tile, compute_done)

            if tile.store_output:
                compute_done = self._store_output(tile, compute_done)

            cycle = compute_done

        self.stats.total_cycles = max(self.stats.total_cycles, cycle)
        return cycle

    def _load_tiles_parallel(
        self, tile: TileOp, slot_idx: int, cycle: int
    ) -> tuple[int, int]:
        """Issue weight+activation DRAM reads concurrently, return (weight_done, act_done).

        Uses begin/end pattern so DRAMSim3 can exploit channel parallelism.
        """
        w_size = tile.tile_m * tile.tile_k * self.systolic.dtype.num_bytes
        a_size = tile.tile_k * tile.tile_n * self.systolic.dtype.num_bytes

        # Begin: issue needed loads at the same cycle (no tick between them)
        w_token = None
        a_token = None
        if tile.load_weight:
            w_addr = self.weight_base_dram + tile.weight_dram_offset
            w_token = self.mem_ctrl.begin_load_from_dram(w_addr, w_size, cycle, "weight")
        if tile.load_activation:
            a_addr = self.act_base_dram + tile.activation_dram_offset
            a_token = self.mem_ctrl.begin_load_from_dram(a_addr, a_size, cycle, "activation")

        # End: tick until both complete
        weight_done = cycle
        if w_token is not None:
            w_result = self.mem_ctrl.end_load_from_dram(
                w_token, self.weight_bufs[slot_idx], w_size, cycle)
            self.stats.memory.dram_read_bytes += w_size
            self.stats.memory.dram_read_count += 1
            self.stats.memory.dram_read_cycles += w_result.dram_cycles
            self.stats.memory.sram_write_cycles += w_result.sram_cycles
            weight_done = w_result.complete_cycle

        act_done = cycle
        if a_token is not None:
            a_result = self.mem_ctrl.end_load_from_dram(
                a_token, self.act_bufs[slot_idx], a_size, cycle)
            self.stats.memory.dram_read_bytes += a_size
            self.stats.memory.dram_read_count += 1
            self.stats.memory.dram_read_cycles += a_result.dram_cycles
            self.stats.memory.sram_write_cycles += a_result.sram_cycles
            act_done = a_result.complete_cycle

        return weight_done, act_done

    def _store_output(self, tile: TileOp, cycle: int) -> int:
        """Store output tile from SRAM to DRAM."""
        dram_addr = self.output_base_dram + tile.output_dram_offset
        size_bytes = tile.tile_m * tile.tile_n * self.systolic.acc_dtype.num_bytes

        result = self.mem_ctrl.store_to_dram(self.output_buf, dram_addr, size_bytes, cycle, "output")
        self.stats.memory.dram_write_bytes += size_bytes
        self.stats.memory.dram_write_count += 1
        self.stats.memory.sram_read_cycles += result.sram_cycles
        self.stats.memory.dram_write_cycles += result.dram_cycles
        return result.complete_cycle

    def _accumulate_output(self, tile: TileOp, cycle: int) -> int:
        """Partial-sum SRAM traffic for WS/IS accumulation between k-tiles."""
        size = tile.tile_m * tile.tile_n * self.systolic.acc_dtype.num_bytes
        done = cycle
        if tile.accumulate:
            done = self.output_buf.read(0, size, done)
        if not tile.store_output:
            done = self.output_buf.write(0, size, done)
        self.stats.memory.accumulation_count += 1
        self.stats.memory.accumulation_cycles += done - cycle
        return done

    def _update_compute_stats(self, tile: TileOp, cycles_info) -> None:
        self.stats.compute.total_mac_ops += tile.tile_m * tile.tile_n * tile.tile_k
        self.stats.compute.total_compute_cycles += cycles_info.total_cycles
        self.stats.compute.pipeline_fill_cycles += cycles_info.pipeline_fill_cycles
        self.stats.compute.pipeline_drain_cycles += cycles_info.pipeline_drain_cycles
        self.stats.compute.total_tiles_processed += 1
