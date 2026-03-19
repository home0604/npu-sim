from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from memory.dram_interface import SimpleDRAMModel
from memory.sram import BankedSRAM

if TYPE_CHECKING:
    from core.clock import SimulationEngine


@dataclass
class MemXferResult:
    """Cycle breakdown for a DRAM<->SRAM transfer."""
    complete_cycle: int
    dram_cycles: int
    sram_cycles: int


class MemoryController:
    """Coordinates data movement between DRAM and SRAM.

    Uses SimpleDRAMModel for latency estimation and BankedSRAM for
    on-chip access timing with bank conflict modeling.
    """

    def __init__(
        self,
        dram: SimpleDRAMModel,
        sram: BankedSRAM,
        engine: SimulationEngine,
    ):
        self.dram = dram
        self.sram = sram
        self.engine = engine

    def load_from_dram(
        self, dram_addr: int, sram_addr: int, size_bytes: int, cycle: int, tag: str = ""
    ) -> MemXferResult:
        """Load data from DRAM to SRAM. Returns MemXferResult with cycle breakdown.

        Steps:
        1. Issue DRAM read (latency + bandwidth)
        2. When DRAM data arrives, write to SRAM (bank conflict modeling)
        """
        dram_resp = self.dram.issue_read(dram_addr, size_bytes, cycle, tag)
        sram_done = self.sram.write(sram_addr, size_bytes, dram_resp.complete_cycle)
        dram_cycles = dram_resp.complete_cycle - cycle
        sram_cycles = sram_done - dram_resp.complete_cycle
        return MemXferResult(
            complete_cycle=sram_done,
            dram_cycles=dram_cycles,
            sram_cycles=sram_cycles,
        )

    def store_to_dram(
        self, sram_addr: int, dram_addr: int, size_bytes: int, cycle: int, tag: str = ""
    ) -> MemXferResult:
        """Store data from SRAM to DRAM. Returns MemXferResult with cycle breakdown.

        Steps:
        1. Read from SRAM (bank conflict modeling)
        2. Issue DRAM write
        """
        sram_done = self.sram.read(sram_addr, size_bytes, cycle)
        dram_resp = self.dram.issue_write(dram_addr, size_bytes, sram_done, tag)
        sram_cycles = sram_done - cycle
        dram_cycles = dram_resp.complete_cycle - sram_done
        return MemXferResult(
            complete_cycle=dram_resp.complete_cycle,
            dram_cycles=dram_cycles,
            sram_cycles=sram_cycles,
        )
