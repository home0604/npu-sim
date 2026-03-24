from __future__ import annotations

from dataclasses import dataclass

from memory.sram import BankedSRAM


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


def create_gptvq_buffer_partitions(
    sram: BankedSRAM,
    codebook_fraction: float = 0.10,
    index_fraction: float = 0.05,
    scale_fraction: float = 0.05,
    dequant_weight_fraction: float = 0.25,
    activation_fraction: float = 0.35,
    output_fraction: float = 0.20,
    double_buffer: bool = True,
    use_scaling: bool = False,
) -> dict[str, list[SRAMBuffer]]:
    """Create GPTVQ buffer partitions within the SRAM.

    Partitions into 6 regions: codebook, index, scale, dequant_weight, activation, output.
    - dequant_weight: holds dequantized weight tile (tile_m × tile_k × entry_bytes)
    - When scaling is disabled, scale buffer space is redistributed to index and activation.

    With double buffering: index, scale, dequant_weight, activation get 2 slots each.
    Codebook and output are single-slot.
    """
    total = sram.size_bytes

    if not use_scaling:
        extra = scale_fraction / 2
        index_fraction += extra
        activation_fraction += extra
        scale_fraction = 0.0

    codebook_size = int(total * codebook_fraction)
    index_size = int(total * index_fraction)
    scale_size = int(total * scale_fraction)
    dequant_weight_size = int(total * dequant_weight_fraction)
    act_size = int(total * activation_fraction)
    output_size = total - codebook_size - index_size - scale_size - dequant_weight_size - act_size

    buffers: dict[str, list[SRAMBuffer]] = {}
    addr = 0

    buffers["codebook"] = [SRAMBuffer(sram, addr, codebook_size, "codebook")]
    addr += codebook_size

    if double_buffer:
        slot_idx = index_size // 2
        buffers["index"] = [
            SRAMBuffer(sram, addr, slot_idx, "index_A"),
            SRAMBuffer(sram, addr + slot_idx, slot_idx, "index_B"),
        ]
        addr += index_size

        if use_scaling and scale_size > 0:
            slot_sc = scale_size // 2
            buffers["scale"] = [
                SRAMBuffer(sram, addr, slot_sc, "scale_A"),
                SRAMBuffer(sram, addr + slot_sc, slot_sc, "scale_B"),
            ]
        else:
            buffers["scale"] = [SRAMBuffer(sram, addr, 0, "scale_none")]
        addr += scale_size

        slot_dq = dequant_weight_size // 2
        buffers["dequant_weight"] = [
            SRAMBuffer(sram, addr, slot_dq, "dequant_weight_A"),
            SRAMBuffer(sram, addr + slot_dq, slot_dq, "dequant_weight_B"),
        ]
        addr += dequant_weight_size

        slot_act = act_size // 2
        buffers["activation"] = [
            SRAMBuffer(sram, addr, slot_act, "activation_A"),
            SRAMBuffer(sram, addr + slot_act, slot_act, "activation_B"),
        ]
        addr += act_size
    else:
        buffers["index"] = [SRAMBuffer(sram, addr, index_size, "index")]
        addr += index_size

        if use_scaling and scale_size > 0:
            buffers["scale"] = [SRAMBuffer(sram, addr, scale_size, "scale")]
        else:
            buffers["scale"] = [SRAMBuffer(sram, addr, 0, "scale_none")]
        addr += scale_size

        buffers["dequant_weight"] = [SRAMBuffer(sram, addr, dequant_weight_size, "dequant_weight")]
        addr += dequant_weight_size

        buffers["activation"] = [SRAMBuffer(sram, addr, act_size, "activation")]
        addr += act_size

    buffers["output"] = [SRAMBuffer(sram, addr, output_size, "output")]
    return buffers
