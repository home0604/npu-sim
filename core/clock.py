from __future__ import annotations

from collections import defaultdict
from typing import Callable

from core.events import Event, EventQueue, EventType


class SimulationEngine:
    """Event-driven simulation engine.

    Uses jump-to-next-event instead of per-cycle tick to avoid wasting
    time on empty cycles. Critical for hybrid simulation where compute
    blocks span thousands of cycles.
    """

    def __init__(self):
        self.current_cycle: int = 0
        self.event_queue = EventQueue()
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = defaultdict(list)

    def register_handler(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        self._handlers[event_type].append(handler)

    def unregister_handler(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def schedule_event(
        self,
        delay: int,
        event_type: EventType,
        data: dict | None = None,
        priority: int = 0,
        tag: str = "",
        callback: Callable[[Event], None] | None = None,
    ) -> None:
        """Schedule an event at current_cycle + delay."""
        event = Event(
            cycle=self.current_cycle + delay,
            priority=priority,
            event_type=event_type,
            data=data or {},
            tag=tag,
            callback=callback,
        )
        self.event_queue.push(event)

    def schedule_event_at(
        self,
        cycle: int,
        event_type: EventType,
        data: dict | None = None,
        priority: int = 0,
        tag: str = "",
        callback: Callable[[Event], None] | None = None,
    ) -> None:
        """Schedule an event at an absolute cycle."""
        event = Event(
            cycle=cycle,
            priority=priority,
            event_type=event_type,
            data=data or {},
            tag=tag,
            callback=callback,
        )
        self.event_queue.push(event)

    def run(self, max_cycles: int = 0) -> None:
        """Main simulation loop -- jumps to next event."""
        while not self.event_queue.empty:
            event = self.event_queue.pop()

            if max_cycles > 0 and event.cycle > max_cycles:
                break

            self.current_cycle = event.cycle

            for handler in self._handlers.get(event.event_type, []):
                handler(event)

            if event.callback:
                event.callback(event)

    def run_one_event(self) -> bool:
        """Pop and process exactly one event. Returns False if queue empty, True otherwise."""
        if self.event_queue.empty:
            return False
        event = self.event_queue.pop()
        self.current_cycle = event.cycle
        for handler in self._handlers.get(event.event_type, []):
            handler(event)
        if event.callback:
            event.callback(event)
        return True

    def run_until(self, target_cycle: int) -> None:
        """Run simulation until a specific cycle."""
        while not self.event_queue.empty:
            next_event = self.event_queue.peek()
            if next_event is None or next_event.cycle > target_cycle:
                break
            event = self.event_queue.pop()
            self.current_cycle = event.cycle

            for handler in self._handlers.get(event.event_type, []):
                handler(event)

            if event.callback:
                event.callback(event)

        self.current_cycle = max(self.current_cycle, target_cycle)

    def reset(self) -> None:
        self.current_cycle = 0
        self.event_queue.clear()
