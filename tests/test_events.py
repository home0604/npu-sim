"""Tests for the event system and simulation engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.clock import SimulationEngine
from core.events import Event, EventQueue, EventType


def test_event_queue_ordering():
    """Events should be popped in cycle order."""
    q = EventQueue()
    q.push(Event(cycle=10, event_type=EventType.COMPUTE_DONE))
    q.push(Event(cycle=5, event_type=EventType.COMPUTE_START))
    q.push(Event(cycle=15, event_type=EventType.DRAM_READ_COMPLETE))

    assert q.pop().cycle == 5
    assert q.pop().cycle == 10
    assert q.pop().cycle == 15
    assert q.empty


def test_event_queue_priority_tiebreak():
    """Events at the same cycle should be ordered by priority."""
    q = EventQueue()
    q.push(Event(cycle=10, priority=2, event_type=EventType.COMPUTE_DONE))
    q.push(Event(cycle=10, priority=0, event_type=EventType.COMPUTE_START))
    q.push(Event(cycle=10, priority=1, event_type=EventType.DRAM_READ_COMPLETE))

    assert q.pop().priority == 0
    assert q.pop().priority == 1
    assert q.pop().priority == 2


def test_simulation_engine_run():
    """Engine should dispatch events to registered handlers."""
    engine = SimulationEngine()
    results = []

    def handler(event: Event):
        results.append((event.cycle, event.event_type))

    engine.register_handler(EventType.COMPUTE_START, handler)
    engine.register_handler(EventType.COMPUTE_DONE, handler)

    engine.schedule_event(delay=10, event_type=EventType.COMPUTE_START)
    engine.schedule_event(delay=20, event_type=EventType.COMPUTE_DONE)

    engine.run()

    assert len(results) == 2
    assert results[0] == (10, EventType.COMPUTE_START)
    assert results[1] == (20, EventType.COMPUTE_DONE)
    assert engine.current_cycle == 20


def test_simulation_engine_max_cycles():
    """Engine should stop at max_cycles."""
    engine = SimulationEngine()
    results = []

    def handler(event: Event):
        results.append(event.cycle)

    engine.register_handler(EventType.COMPUTE_START, handler)

    engine.schedule_event(delay=5, event_type=EventType.COMPUTE_START)
    engine.schedule_event(delay=15, event_type=EventType.COMPUTE_START)
    engine.schedule_event(delay=25, event_type=EventType.COMPUTE_START)

    engine.run(max_cycles=20)

    assert len(results) == 2
    assert results == [5, 15]


def test_simulation_engine_callback():
    """Events with callbacks should have them invoked."""
    engine = SimulationEngine()
    callback_called = []

    def cb(event: Event):
        callback_called.append(event.cycle)

    engine.schedule_event(delay=10, event_type=EventType.COMPUTE_START, callback=cb)
    engine.run()

    assert callback_called == [10]


def test_simulation_engine_schedule_during_run():
    """Handlers can schedule new events during simulation."""
    engine = SimulationEngine()
    results = []

    def handler(event: Event):
        results.append(event.cycle)
        if event.cycle < 30:
            engine.schedule_event(delay=10, event_type=EventType.COMPUTE_START)

    engine.register_handler(EventType.COMPUTE_START, handler)
    engine.schedule_event(delay=10, event_type=EventType.COMPUTE_START)

    engine.run(max_cycles=50)

    assert results == [10, 20, 30]
