from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
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
            # .so is built in package root (npu_sim/); ensure it is on path
            _pkg_root = Path(__file__).resolve().parent.parent
            if str(_pkg_root) not in sys.path:
                sys.path.insert(0, str(_pkg_root))
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

    def tick(self) -> None:
        """Advance DRAMSim3 by one cycle. Callbacks may schedule completion events."""
        self._mem_system.ClockTick()
        self._dramsim_cycle += 1

    def tick_n(self, n: int) -> None:
        """Advance DRAMSim3 by n cycles. Callbacks may schedule completion events."""
        for _ in range(n):
            self._mem_system.ClockTick()
            self._dramsim_cycle += 1

    @property
    def current_cycle(self) -> int:
        """Current DRAMSim3 cycle (for adapter batch drain)."""
        return self._dramsim_cycle

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
            while not self._mem_system.AddTransaction(addr, False):
                self._mem_system.ClockTick()
                self._dramsim_cycle += 1
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
            while not self._mem_system.AddTransaction(addr, True):
                self._mem_system.ClockTick()
                self._dramsim_cycle += 1
            self._pending_writes[addr] = req

        self.total_writes += 1
        self.total_write_bytes += size_bytes
        return self._request_counter


# Batch size for DRAMSim3Adapter: tick this many cycles before draining events.
# Larger = fewer Python loop iterations, faster; too large may delay completion handling.
_DRAMSIM3_TICK_BATCH = 2048


class DRAMSim3Adapter:
    """Wraps DRAMSim3Interface to provide synchronous issue_read/issue_write returning DRAMResponse.

    The scheduler and MemoryController expect immediate completion cycle; DRAMSim3
    completes asynchronously via callbacks. This adapter runs the engine and ticks
    DRAMSim3 until the request completes, then returns DRAMResponse.
    Uses batch ticking to reduce Python overhead (tick many cycles, then drain events).
    """

    def __init__(
        self,
        dramsim: DRAMSim3Interface,
        engine: "SimulationEngine",
    ):
        self._dramsim = dramsim
        self._engine = engine
        self.total_reads = dramsim.total_reads
        self.total_writes = dramsim.total_writes
        self.total_read_bytes = dramsim.total_read_bytes
        self.total_write_bytes = dramsim.total_write_bytes

    def _drain_events_until(self, max_cycle: int) -> None:
        """Process all events with cycle <= max_cycle."""
        while not self._engine.event_queue.empty:
            next_ev = self._engine.event_queue.peek()
            if next_ev is None or next_ev.cycle > max_cycle:
                break
            self._engine.run_one_event()

    def _sync_read_completion(
        self, request_id: int, expected_callbacks: int
    ) -> DRAMResponse:
        from ..core.events import EventType

        state = {
            "count": 0,
            "max_cycle": 0,
            "request": None,
        }

        def on_read_complete(event) -> None:
            req = event.data.get("request")
            resp = event.data.get("response")
            if req is None or resp is None or req.request_id != request_id:
                return
            state["count"] += 1
            state["max_cycle"] = max(state["max_cycle"], resp.complete_cycle)
            if state["request"] is None:
                state["request"] = req

        self._engine.register_handler(EventType.DRAM_READ_COMPLETE, on_read_complete)
        try:
            while state["count"] < expected_callbacks:
                self._dramsim.tick_n(_DRAMSIM3_TICK_BATCH)
                self._drain_events_until(self._dramsim.current_cycle)
            self.total_reads = self._dramsim.total_reads
            self.total_read_bytes = self._dramsim.total_read_bytes
            return DRAMResponse(
                request=state["request"],
                complete_cycle=state["max_cycle"],
            )
        finally:
            self._engine.unregister_handler(EventType.DRAM_READ_COMPLETE, on_read_complete)

    def _sync_write_completion(
        self, request_id: int, expected_callbacks: int
    ) -> DRAMResponse:
        from ..core.events import EventType

        state = {
            "count": 0,
            "max_cycle": 0,
            "request": None,
        }

        def on_write_complete(event) -> None:
            req = event.data.get("request")
            resp = event.data.get("response")
            if req is None or resp is None or req.request_id != request_id:
                return
            state["count"] += 1
            state["max_cycle"] = max(state["max_cycle"], resp.complete_cycle)
            if state["request"] is None:
                state["request"] = req

        self._engine.register_handler(EventType.DRAM_WRITE_COMPLETE, on_write_complete)
        try:
            while state["count"] < expected_callbacks:
                self._dramsim.tick_n(_DRAMSIM3_TICK_BATCH)
                self._drain_events_until(self._dramsim.current_cycle)
            self.total_writes = self._dramsim.total_writes
            self.total_write_bytes = self._dramsim.total_write_bytes
            return DRAMResponse(
                request=state["request"],
                complete_cycle=state["max_cycle"],
            )
        finally:
            self._engine.unregister_handler(
                EventType.DRAM_WRITE_COMPLETE, on_write_complete
            )

    def issue_read(
        self, address: int, size_bytes: int, cycle: int, tag: str = ""
    ) -> DRAMResponse:
        expected = (size_bytes + CACHELINE_SIZE - 1) // CACHELINE_SIZE
        request_id = self._dramsim.issue_read(address, size_bytes, cycle, tag)
        return self._sync_read_completion(request_id, expected)

    def issue_write(
        self, address: int, size_bytes: int, cycle: int, tag: str = ""
    ) -> DRAMResponse:
        expected = (size_bytes + CACHELINE_SIZE - 1) // CACHELINE_SIZE
        request_id = self._dramsim.issue_write(address, size_bytes, cycle, tag)
        return self._sync_write_completion(request_id, expected)

    def reset(self) -> None:
        """DRAMSim3 state is not reset by this adapter; config-driven if needed."""
        pass
