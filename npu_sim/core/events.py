from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class EventType(Enum):
    # DRAM events
    DRAM_READ_REQUEST = auto()
    DRAM_READ_COMPLETE = auto()
    DRAM_WRITE_REQUEST = auto()
    DRAM_WRITE_COMPLETE = auto()

    # SRAM events
    SRAM_READ_COMPLETE = auto()
    SRAM_WRITE_COMPLETE = auto()

    # Compute events
    COMPUTE_START = auto()
    COMPUTE_DONE = auto()

    # Tile orchestration events
    TILE_WEIGHT_READY = auto()
    TILE_ACTIVATION_READY = auto()
    TILE_BOTH_READY = auto()
    TILE_OUTPUT_READY = auto()

    # Double buffer events
    BUFFER_SWAP = auto()
    PREFETCH_START = auto()
    PREFETCH_DONE = auto()

    # Layer-level events
    LAYER_START = auto()
    LAYER_DONE = auto()


@dataclass(order=True)
class Event:
    cycle: int
    priority: int = 0  # lower = higher priority (tie-breaking)
    seq: int = field(default=0, compare=True)  # insertion order for stable sort
    event_type: EventType = field(compare=False, default=EventType.COMPUTE_START)
    data: dict[str, Any] = field(compare=False, default_factory=dict)
    callback: Callable[[Event], None] | None = field(compare=False, default=None)
    tag: str = field(compare=False, default="")


class EventQueue:
    """Heap-based priority queue for simulation events."""

    def __init__(self):
        self._heap: list[Event] = []
        self._counter: int = 0

    def push(self, event: Event) -> None:
        event.seq = self._counter
        self._counter += 1
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek(self) -> Event | None:
        return self._heap[0] if self._heap else None

    @property
    def empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._counter = 0
