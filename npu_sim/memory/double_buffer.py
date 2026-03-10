from __future__ import annotations

from enum import Enum, auto


class BufferSlot(Enum):
    SLOT_A = 0
    SLOT_B = 1


class SlotState(Enum):
    EMPTY = auto()
    LOADING = auto()
    READY = auto()
    COMPUTING = auto()
    DRAINING = auto()


class DoubleBufferController:
    """Double buffering state machine.

    Manages two buffer slots that alternate between loading and computing:
    - While slot A is COMPUTING, slot B is LOADING (prefetch next tile)
    - When compute finishes on A, swap: B becomes COMPUTING, A starts LOADING

    State transitions per slot:
        EMPTY -> LOADING      (prefetch initiated)
        LOADING -> READY      (DMA transfer complete)
        READY -> COMPUTING    (systolic array starts using data)
        COMPUTING -> DRAINING (compute done, output writeback)
        DRAINING -> EMPTY     (writeback complete)

    Stall: compute finishes but next slot is still LOADING.
    """

    def __init__(self):
        self._states = {
            BufferSlot.SLOT_A: SlotState.EMPTY,
            BufferSlot.SLOT_B: SlotState.EMPTY,
        }
        self._tile_info: dict[BufferSlot, dict | None] = {
            BufferSlot.SLOT_A: None,
            BufferSlot.SLOT_B: None,
        }
        self.compute_slot = BufferSlot.SLOT_A
        self.prefetch_slot = BufferSlot.SLOT_B

        # Tracking partial readiness (weight + activation)
        self._weight_ready = {BufferSlot.SLOT_A: False, BufferSlot.SLOT_B: False}
        self._activation_ready = {BufferSlot.SLOT_A: False, BufferSlot.SLOT_B: False}

        # Statistics
        self.prefetch_hits: int = 0
        self.prefetch_misses: int = 0
        self.stall_cycles: int = 0

    @property
    def compute_state(self) -> SlotState:
        return self._states[self.compute_slot]

    @property
    def prefetch_state(self) -> SlotState:
        return self._states[self.prefetch_slot]

    def start_loading(self, slot: BufferSlot, tile_info: dict | None = None) -> None:
        self._states[slot] = SlotState.LOADING
        self._tile_info[slot] = tile_info
        self._weight_ready[slot] = False
        self._activation_ready[slot] = False

    def mark_weight_ready(self, slot: BufferSlot) -> bool:
        """Mark weight as loaded. Returns True if both weight+activation ready."""
        self._weight_ready[slot] = True
        return self._check_both_ready(slot)

    def mark_activation_ready(self, slot: BufferSlot) -> bool:
        """Mark activation as loaded. Returns True if both weight+activation ready."""
        self._activation_ready[slot] = True
        return self._check_both_ready(slot)

    def _check_both_ready(self, slot: BufferSlot) -> bool:
        if self._weight_ready[slot] and self._activation_ready[slot]:
            self._states[slot] = SlotState.READY
            return True
        return False

    def can_start_compute(self) -> bool:
        return self._states[self.compute_slot] == SlotState.READY

    def start_compute(self) -> bool:
        """Try to start computation on the compute slot. Returns True if started."""
        if self._states[self.compute_slot] == SlotState.READY:
            self._states[self.compute_slot] = SlotState.COMPUTING
            self.prefetch_hits += 1
            return True
        self.prefetch_misses += 1
        return False

    def finish_compute(self) -> None:
        """Mark compute as done, start draining."""
        self._states[self.compute_slot] = SlotState.DRAINING

    def finish_drain(self) -> None:
        """Mark drain as done, slot is now empty."""
        self._states[self.compute_slot] = SlotState.EMPTY

    def swap(self) -> None:
        """Swap compute and prefetch slots."""
        self.compute_slot, self.prefetch_slot = self.prefetch_slot, self.compute_slot

    def add_stall_cycles(self, cycles: int) -> None:
        self.stall_cycles += cycles

    def reset(self) -> None:
        for slot in BufferSlot:
            self._states[slot] = SlotState.EMPTY
            self._tile_info[slot] = None
            self._weight_ready[slot] = False
            self._activation_ready[slot] = False
        self.compute_slot = BufferSlot.SLOT_A
        self.prefetch_slot = BufferSlot.SLOT_B
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.stall_cycles = 0
