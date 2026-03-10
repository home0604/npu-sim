from __future__ import annotations

from dataclasses import dataclass

from .sram import BankedSRAM


@dataclass
class BufferRegion:
    """A logical region within the banked SRAM."""

    base_addr: int
    size_bytes: int


class SRAMBuffer:
    """Manages a logical buffer region within the banked SRAM.

    Provides load/store operations that map to the underlying BankedSRAM
    with proper address offsets.
    """

    def __init__(self, sram: BankedSRAM, base_addr: int, size_bytes: int, name: str = ""):
        self.sram = sram
        self.base_addr = base_addr
        self.size_bytes = size_bytes
        self.name = name
        self._used_bytes: int = 0

    def can_fit(self, data_bytes: int) -> bool:
        return data_bytes <= self.size_bytes

    def read(self, offset: int, size_bytes: int, cycle: int) -> int:
        """Read from this buffer region. Returns completion cycle."""
        addr = self.base_addr + offset
        return self.sram.read(addr, size_bytes, cycle)

    def write(self, offset: int, size_bytes: int, cycle: int) -> int:
        """Write to this buffer region. Returns completion cycle."""
        addr = self.base_addr + offset
        return self.sram.write(addr, size_bytes, cycle)


def create_buffer_partitions(
    sram: BankedSRAM,
    weight_fraction: float = 0.4,
    activation_fraction: float = 0.3,
    output_fraction: float = 0.3,
    double_buffer: bool = True,
) -> dict[str, list[SRAMBuffer]]:
    """Create buffer partitions within the SRAM.

    With double buffering, weight and activation buffers get 2 slots each.
    Output buffer is single (partial sums accumulate in place).

    Returns:
        dict with keys "weight", "activation", "output", each mapping
        to a list of SRAMBuffer instances (2 for double-buffered, 1 otherwise).
    """
    total = sram.size_bytes
    weight_size = int(total * weight_fraction)
    act_size = int(total * activation_fraction)
    output_size = total - weight_size - act_size  # remainder to avoid rounding loss

    buffers: dict[str, list[SRAMBuffer]] = {}
    addr = 0

    if double_buffer:
        # Two weight buffer slots
        slot_w = weight_size // 2
        buffers["weight"] = [
            SRAMBuffer(sram, addr, slot_w, "weight_A"),
            SRAMBuffer(sram, addr + slot_w, slot_w, "weight_B"),
        ]
        addr += weight_size

        # Two activation buffer slots
        slot_a = act_size // 2
        buffers["activation"] = [
            SRAMBuffer(sram, addr, slot_a, "activation_A"),
            SRAMBuffer(sram, addr + slot_a, slot_a, "activation_B"),
        ]
        addr += act_size
    else:
        buffers["weight"] = [SRAMBuffer(sram, addr, weight_size, "weight")]
        addr += weight_size

        buffers["activation"] = [SRAMBuffer(sram, addr, act_size, "activation")]
        addr += act_size

    buffers["output"] = [SRAMBuffer(sram, addr, output_size, "output")]
    return buffers
