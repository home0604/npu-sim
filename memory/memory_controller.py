from __future__ import annotations

from typing import TYPE_CHECKING

from .dram_interface import SimpleDRAMModel
from .sram import BankedSRAM

if TYPE_CHECKING:
    from ..core.clock import SimulationEngine


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
    ) -> int:
        """Load data from DRAM to SRAM. Returns completion cycle.

        Steps:
        1. Issue DRAM read (latency + bandwidth)
        2. When DRAM data arrives, write to SRAM (bank conflict modeling)
        """
        dram_resp = self.dram.issue_read(dram_addr, size_bytes, cycle, tag)
        sram_done = self.sram.write(sram_addr, size_bytes, dram_resp.complete_cycle)
        return sram_done

    def store_to_dram(
        self, sram_addr: int, dram_addr: int, size_bytes: int, cycle: int, tag: str = ""
    ) -> int:
        """Store data from SRAM to DRAM. Returns completion cycle.

        Steps:
        1. Read from SRAM (bank conflict modeling)
        2. Issue DRAM write
        """
        sram_done = self.sram.read(sram_addr, size_bytes, cycle)
        dram_resp = self.dram.issue_write(dram_addr, size_bytes, sram_done, tag)
        return dram_resp.complete_cycle
