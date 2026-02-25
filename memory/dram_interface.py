from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.clock import SimulationEngine


CACHELINE_SIZE = 64


@dataclass
class DRAMRequest:
    address: int
    size_bytes: int
    is_write: bool
    issue_cycle: int
    tag: str = ""
    request_id: int = 0


@dataclass
class DRAMResponse:
    request: DRAMRequest
    complete_cycle: int


class SimpleDRAMModel:
    """Simple analytical DRAM model (fallback when DRAMSim3 not available).

    Models DRAM with a fixed latency + bandwidth-limited transfer time.
    This provides reasonable estimates for initial development and can be
    replaced with DRAMSim3 for cycle-accurate modeling.

    Transfer time = latency + ceil(size / cacheline) * (cacheline / bandwidth)
    """

    def __init__(
        self,
        bandwidth_gbps: float = 25.6,
        latency_ns: float = 50.0,
        clock_freq_mhz: int = 1000,
    ):
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_ns = latency_ns
        self.clock_freq_mhz = clock_freq_mhz

        # Pre-compute cycles
        self.latency_cycles = int(latency_ns * clock_freq_mhz / 1000)
        # Bytes per cycle: bandwidth_gbps * 1e9 / 8 / (clock_freq_mhz * 1e6)
        self.bytes_per_cycle = (bandwidth_gbps * 1e9 / 8) / (clock_freq_mhz * 1e6)

        self._request_counter = 0

        # Stats
        self.total_reads = 0
        self.total_writes = 0
        self.total_read_bytes = 0
        self.total_write_bytes = 0

        # Simple contention model: track when the DRAM bus is free
        self._bus_free_cycle = 0

    def issue_read(self, address: int, size_bytes: int, cycle: int, tag: str = "") -> DRAMResponse:
        """Issue a read request. Returns response with completion cycle."""
        self._request_counter += 1
        req = DRAMRequest(
            address=address,
            size_bytes=size_bytes,
            is_write=False,
            issue_cycle=cycle,
            tag=tag,
            request_id=self._request_counter,
        )

        start_cycle = max(cycle, self._bus_free_cycle)
        transfer_cycles = max(1, int(size_bytes / self.bytes_per_cycle))
        complete_cycle = start_cycle + self.latency_cycles + transfer_cycles
        self._bus_free_cycle = complete_cycle

        self.total_reads += 1
        self.total_read_bytes += size_bytes
        return DRAMResponse(request=req, complete_cycle=complete_cycle)

    def issue_write(self, address: int, size_bytes: int, cycle: int, tag: str = "") -> DRAMResponse:
        """Issue a write request. Returns response with completion cycle."""
        self._request_counter += 1
        req = DRAMRequest(
            address=address,
            size_bytes=size_bytes,
            is_write=True,
            issue_cycle=cycle,
            tag=tag,
            request_id=self._request_counter,
        )

        start_cycle = max(cycle, self._bus_free_cycle)
        transfer_cycles = max(1, int(size_bytes / self.bytes_per_cycle))
        complete_cycle = start_cycle + self.latency_cycles + transfer_cycles
        self._bus_free_cycle = complete_cycle

        self.total_writes += 1
        self.total_write_bytes += size_bytes
        return DRAMResponse(request=req, complete_cycle=complete_cycle)

    def reset(self) -> None:
        self._bus_free_cycle = 0
        self._request_counter = 0
        self.total_reads = 0
        self.total_writes = 0
        self.total_read_bytes = 0
        self.total_write_bytes = 0


class DRAMSim3Interface:
    """DRAMSim3 integration via pybind11.

    Requires DRAMSim3 to be built as a shared library and the pybind11
    wrapper to be compiled. See scripts/setup_dramsim3.sh.

    Falls back to SimpleDRAMModel if DRAMSim3 is not available.
    """

    def __init__(
        self,
        config_file: str,
        output_dir: str,
        sim_engine: SimulationEngine,
    ):
        self.sim_engine = sim_engine
        self._dramsim_cycle = 0
        self._request_counter = 0
        self._pending_reads: dict[int, DRAMRequest] = {}
        self._pending_writes: dict[int, DRAMRequest] = {}

        self.total_reads = 0
        self.total_writes = 0
        self.total_read_bytes = 0
        self.total_write_bytes = 0

        try:
            import dramsim3_py  # type: ignore

            self._mem_system = dramsim3_py.MemorySystem(
                config_file,
                output_dir,
                self._read_callback,
                self._write_callback,
            )
            self._available = True
        except ImportError:
            raise ImportError(
                "DRAMSim3 pybind11 module not found. "
                "Build it with scripts/setup_dramsim3.sh or use SimpleDRAMModel."
            )

    def _read_callback(self, address: int) -> None:
        from ..core.events import EventType

        if address in self._pending_reads:
            req = self._pending_reads.pop(address)
            resp = DRAMResponse(request=req, complete_cycle=self._dramsim_cycle)
            self.sim_engine.schedule_event_at(
                cycle=self._dramsim_cycle,
                event_type=EventType.DRAM_READ_COMPLETE,
                data={"request": req, "response": resp},
            )

    def _write_callback(self, address: int) -> None:
        from ..core.events import EventType

        if address in self._pending_writes:
            req = self._pending_writes.pop(address)
            resp = DRAMResponse(request=req, complete_cycle=self._dramsim_cycle)
            self.sim_engine.schedule_event_at(
                cycle=self._dramsim_cycle,
                event_type=EventType.DRAM_WRITE_COMPLETE,
                data={"request": req, "response": resp},
            )

    def _advance_to_cycle(self, target_cycle: int) -> None:
        while self._dramsim_cycle < target_cycle:
            self._mem_system.ClockTick()
            self._dramsim_cycle += 1

    def issue_read(self, address: int, size_bytes: int, cycle: int, tag: str = "") -> int:
        self._request_counter += 1
        req = DRAMRequest(
            address=address,
            size_bytes=size_bytes,
            is_write=False,
            issue_cycle=cycle,
            tag=tag,
            request_id=self._request_counter,
        )
        self._advance_to_cycle(cycle)

        for offset in range(0, size_bytes, CACHELINE_SIZE):
            addr = address + offset
            self._mem_system.AddTransaction(addr, False)
            self._pending_reads[addr] = req

        self.total_reads += 1
        self.total_read_bytes += size_bytes
        return self._request_counter

    def issue_write(self, address: int, size_bytes: int, cycle: int, tag: str = "") -> int:
        self._request_counter += 1
        req = DRAMRequest(
            address=address,
            size_bytes=size_bytes,
            is_write=True,
            issue_cycle=cycle,
            tag=tag,
            request_id=self._request_counter,
        )
        self._advance_to_cycle(cycle)

        for offset in range(0, size_bytes, CACHELINE_SIZE):
            addr = address + offset
            self._mem_system.AddTransaction(addr, True)
            self._pending_writes[addr] = req

        self.total_writes += 1
        self.total_write_bytes += size_bytes
        return self._request_counter
