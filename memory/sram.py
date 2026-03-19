from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class PortType(Enum):
    SINGLE = auto()  # one read OR one write per cycle per bank
    DUAL = auto()  # one read AND one write per cycle per bank


@dataclass
class SRAMStats:
    total_reads: int = 0
    total_writes: int = 0
    total_read_bytes: int = 0
    total_write_bytes: int = 0
    bank_conflicts: int = 0
    port_conflicts: int = 0  # dual-port same-type conflicts


class BankedSRAM:
    """Banked SRAM model with configurable port type and banking conflict modeling.

    Banking scheme: interleaved
        bank_id = (address / bank_width_bytes) % num_banks

    Single-port: each bank can do ONE access (read or write) per cycle.
    Dual-port:   each bank can do ONE read AND ONE write per cycle.
                 Two reads or two writes to the same bank in the same cycle conflict.
    """

    def __init__(
        self,
        size_bytes: int,
        num_banks: int,
        bank_width_bytes: int,
        port_type: PortType,
        read_latency: int = 1,
        write_latency: int = 1,
    ):
        self.size_bytes = size_bytes
        self.num_banks = num_banks
        self.bank_width_bytes = bank_width_bytes
        self.port_type = port_type
        self.read_latency = read_latency
        self.write_latency = write_latency
        self.bank_size = size_bytes // num_banks

        # Per-bank busy tracking
        # bank busy는 해당 뱅크가 언제 사용 가능한지 cycle을 저장.
        if port_type == PortType.SINGLE:
            self._bank_busy: list[int] = [0] * num_banks
        else:
            self._bank_read_busy: list[int] = [0] * num_banks
            self._bank_write_busy: list[int] = [0] * num_banks

        self.stats = SRAMStats()

    def _addr_to_bank(self, address: int) -> int:
        return (address // self.bank_width_bytes) % self.num_banks

    def _get_banks_for_access(self, address: int, size_bytes: int) -> list[int]:
        """Determine which banks are accessed for a given address range."""
        banks = []
        offset = 0
        while offset < size_bytes:
            bank = self._addr_to_bank(address + offset)
            banks.append(bank)
            offset += self.bank_width_bytes
        return banks

    def read(self, address: int, size_bytes: int, cycle: int) -> int:
        """Issue a read. Returns the cycle when the read completes.

        For multi-bank accesses, banks are accessed in parallel where possible.
        Bank conflicts cause serialization.
        """
        banks = self._get_banks_for_access(address, size_bytes)
        max_completion = cycle

        for bank_id in banks:
            if self.port_type == PortType.SINGLE:
                start = max(cycle, self._bank_busy[bank_id])
                if start > cycle:
                    self.stats.bank_conflicts += 1
                end = start + self.read_latency
                self._bank_busy[bank_id] = end
            else:
                start = max(cycle, self._bank_read_busy[bank_id])
                if start > cycle:
                    self.stats.port_conflicts += 1
                end = start + self.read_latency
                self._bank_read_busy[bank_id] = end

            max_completion = max(max_completion, end)

        self.stats.total_reads += 1
        self.stats.total_read_bytes += size_bytes
        return max_completion

    def write(self, address: int, size_bytes: int, cycle: int) -> int:
        """Issue a write. Returns the cycle when the write completes."""
        banks = self._get_banks_for_access(address, size_bytes)
        max_completion = cycle

        for bank_id in banks:
            if self.port_type == PortType.SINGLE:
                start = max(cycle, self._bank_busy[bank_id])
                if start > cycle:
                    self.stats.bank_conflicts += 1
                end = start + self.write_latency
                self._bank_busy[bank_id] = end
            else:
                start = max(cycle, self._bank_write_busy[bank_id])
                if start > cycle:
                    self.stats.port_conflicts += 1
                end = start + self.write_latency
                self._bank_write_busy[bank_id] = end

            max_completion = max(max_completion, end)

        self.stats.total_writes += 1
        self.stats.total_write_bytes += size_bytes
        return max_completion

    def reset_stats(self) -> None:
        self.stats = SRAMStats()

    def reset_busy(self) -> None:
        """Reset all bank busy trackers."""
        if self.port_type == PortType.SINGLE:
            self._bank_busy = [0] * self.num_banks
        else:
            self._bank_read_busy = [0] * self.num_banks
            self._bank_write_busy = [0] * self.num_banks
