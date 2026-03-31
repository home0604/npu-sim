from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from memory.dram_interface import SimpleDRAMModel
from memory.sram_buffers import SRAMBuffer

if TYPE_CHECKING:
    from core.clock import SimulationEngine


@dataclass
class MemXferResult:
    """Cycle breakdown for a DRAM<->SRAM transfer."""
    complete_cycle: int
    dram_cycles: int
    sram_cycles: int


class MemoryController:
    """Coordinates data movement between DRAM and SRAM buffers.

    Each SRAMBuffer owns its own BankedSRAM instance, so there are
    no cross-buffer bank conflicts.
    """

    def __init__(
        self,
        dram: SimpleDRAMModel,
        engine: SimulationEngine,
    ):
        self.dram = dram
        self.engine = engine

    def load_from_dram(
        self, dram_addr: int, sram_buf: SRAMBuffer, size_bytes: int, cycle: int, tag: str = ""
    ) -> MemXferResult:
        """Load data from DRAM to SRAM buffer. Returns MemXferResult with cycle breakdown.

        Steps:
        1. Issue DRAM read (latency + bandwidth)
        2. When DRAM data arrives, write to SRAM buffer (bank conflict modeling)
        """
        dram_resp = self.dram.issue_read(dram_addr, size_bytes, cycle, tag)
        sram_done = sram_buf.write(0, size_bytes, dram_resp.complete_cycle)
        dram_cycles = dram_resp.complete_cycle - cycle
        sram_cycles = sram_done - dram_resp.complete_cycle
        return MemXferResult(
            complete_cycle=sram_done,
            dram_cycles=dram_cycles,
            sram_cycles=sram_cycles,
        )

    def begin_load_from_dram(
        self, dram_addr: int, size_bytes: int, cycle: int, tag: str = ""
    ):
        """Issue DRAM read without waiting. Returns opaque token."""
        return self.dram.begin_read(dram_addr, size_bytes, cycle, tag)

    def end_load_from_dram(
        self, token, sram_buf: SRAMBuffer, size_bytes: int, issue_cycle: int
    ) -> MemXferResult:
        """Wait for DRAM read completion and write to SRAM buffer."""
        dram_resp = self.dram.end_read(token)
        sram_done = sram_buf.write(0, size_bytes, dram_resp.complete_cycle)
        dram_cycles = dram_resp.complete_cycle - issue_cycle
        sram_cycles = sram_done - dram_resp.complete_cycle
        return MemXferResult(
            complete_cycle=sram_done,
            dram_cycles=dram_cycles,
            sram_cycles=sram_cycles,
        )

    def store_to_dram(
        self, sram_buf: SRAMBuffer, dram_addr: int, size_bytes: int, cycle: int, tag: str = ""
    ) -> MemXferResult:
        """Store data from SRAM buffer to DRAM. Returns MemXferResult with cycle breakdown.

        Steps:
        1. Read from SRAM buffer (bank conflict modeling)
        2. Issue DRAM write
        """
        sram_done = sram_buf.read(0, size_bytes, cycle)
        dram_resp = self.dram.issue_write(dram_addr, size_bytes, sram_done, tag)
        sram_cycles = sram_done - cycle
        dram_cycles = dram_resp.complete_cycle - sram_done
        return MemXferResult(
            complete_cycle=dram_resp.complete_cycle,
            dram_cycles=dram_cycles,
            sram_cycles=sram_cycles,
        )
