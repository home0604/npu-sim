"""Tests for the double buffer state machine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.double_buffer import BufferSlot, DoubleBufferController, SlotState


def test_initial_state():
    """Both slots should start empty."""
    db = DoubleBufferController()
    assert db.compute_state == SlotState.EMPTY
    assert db.prefetch_state == SlotState.EMPTY


def test_loading_to_ready():
    """Slot should transition from LOADING to READY when both data ready."""
    db = DoubleBufferController()

    db.start_loading(BufferSlot.SLOT_A)
    assert db._states[BufferSlot.SLOT_A] == SlotState.LOADING

    # Mark weight ready (not both ready yet)
    both_ready = db.mark_weight_ready(BufferSlot.SLOT_A)
    assert not both_ready
    assert db._states[BufferSlot.SLOT_A] == SlotState.LOADING

    # Mark activation ready (now both ready)
    both_ready = db.mark_activation_ready(BufferSlot.SLOT_A)
    assert both_ready
    assert db._states[BufferSlot.SLOT_A] == SlotState.READY


def test_compute_flow():
    """Test the full compute flow: READY -> COMPUTING -> DRAINING -> EMPTY."""
    db = DoubleBufferController()

    db.start_loading(BufferSlot.SLOT_A)
    db.mark_weight_ready(BufferSlot.SLOT_A)
    db.mark_activation_ready(BufferSlot.SLOT_A)

    # Start compute
    assert db.can_start_compute()
    started = db.start_compute()
    assert started
    assert db.compute_state == SlotState.COMPUTING

    # Finish compute
    db.finish_compute()
    assert db.compute_state == SlotState.DRAINING

    # Finish drain
    db.finish_drain()
    assert db._states[BufferSlot.SLOT_A] == SlotState.EMPTY


def test_swap():
    """Swap should switch compute and prefetch slots."""
    db = DoubleBufferController()
    assert db.compute_slot == BufferSlot.SLOT_A
    assert db.prefetch_slot == BufferSlot.SLOT_B

    db.swap()
    assert db.compute_slot == BufferSlot.SLOT_B
    assert db.prefetch_slot == BufferSlot.SLOT_A


def test_prefetch_miss():
    """Trying to start compute when not ready should record a miss."""
    db = DoubleBufferController()

    # Slot A is still EMPTY, not READY
    started = db.start_compute()
    assert not started
    assert db.prefetch_misses == 1


def test_double_buffer_alternation():
    """Full alternation between two slots."""
    db = DoubleBufferController()

    # Load slot A (compute slot)
    db.start_loading(BufferSlot.SLOT_A)
    db.mark_weight_ready(BufferSlot.SLOT_A)
    db.mark_activation_ready(BufferSlot.SLOT_A)

    # Start loading slot B (prefetch slot) in parallel
    db.start_loading(BufferSlot.SLOT_B)

    # Start compute on A
    db.start_compute()
    assert db.compute_state == SlotState.COMPUTING

    # While A computes, B finishes loading
    db.mark_weight_ready(BufferSlot.SLOT_B)
    db.mark_activation_ready(BufferSlot.SLOT_B)
    assert db._states[BufferSlot.SLOT_B] == SlotState.READY

    # A finishes compute
    db.finish_compute()
    db.finish_drain()

    # Swap: B becomes compute, A becomes prefetch
    db.swap()
    assert db.compute_slot == BufferSlot.SLOT_B

    # Start compute on B
    started = db.start_compute()
    assert started
    assert db.prefetch_hits == 2
