from __future__ import annotations

from dataclasses import dataclass

from memory.sram import BankedSRAM, PortType


@dataclass
class BufferRegion:
    """A logical region within the banked SRAM."""

    base_addr: int
    size_bytes: int


class SRAMBuffer:
    """Manages a logical buffer region within a BankedSRAM.

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


def _make_sram(
    size_bytes: int,
    num_banks: int,
    bank_width_bytes: int,
    port_type: PortType,
    read_latency: int = 1,
    write_latency: int = 1,
) -> BankedSRAM:
    """Create a BankedSRAM instance, clamping num_banks to avoid zero bank_size."""
    if size_bytes <= 0:
        size_bytes = 1  # dummy for zero-sized buffers (e.g. scale when disabled)
    effective_banks = min(num_banks, max(1, size_bytes // bank_width_bytes))
    return BankedSRAM(
        size_bytes=size_bytes,
        num_banks=effective_banks,
        bank_width_bytes=bank_width_bytes,
        port_type=port_type,
        read_latency=read_latency,
        write_latency=write_latency,
    )


def create_buffer_partitions(
    total_size_bytes: int,
    num_banks: int,
    bank_width_bytes: int,
    port_type: PortType,
    read_latency: int = 1,
    write_latency: int = 1,
    weight_fraction: float = 0.4,
    activation_fraction: float = 0.3,
    output_fraction: float = 0.3,
    double_buffer: bool = True,
) -> dict[str, list[SRAMBuffer]]:
    """Create buffer partitions with separate BankedSRAM per buffer group.

    Each buffer group (weight, activation, output) gets its own BankedSRAM
    instance so there are no cross-buffer bank conflicts.

    With double buffering, weight and activation buffers get 2 slots each.
    Output buffer is single (partial sums accumulate in place).

    Returns:
        dict with keys "weight", "activation", "output", each mapping
        to a list of SRAMBuffer instances (2 for double-buffered, 1 otherwise).
    """
    weight_size = int(total_size_bytes * weight_fraction)
    act_size = int(total_size_bytes * activation_fraction)
    output_size = total_size_bytes - weight_size - act_size

    sram_args = dict(num_banks=num_banks, bank_width_bytes=bank_width_bytes,
                     port_type=port_type, read_latency=read_latency,
                     write_latency=write_latency)

    weight_sram = _make_sram(weight_size, **sram_args)
    act_sram = _make_sram(act_size, **sram_args)
    output_sram = _make_sram(output_size, **sram_args)

    buffers: dict[str, list[SRAMBuffer]] = {}

    if double_buffer:
        slot_w = weight_size // 2
        buffers["weight"] = [
            SRAMBuffer(weight_sram, 0, slot_w, "weight_A"),
            SRAMBuffer(weight_sram, slot_w, slot_w, "weight_B"),
        ]
        slot_a = act_size // 2
        buffers["activation"] = [
            SRAMBuffer(act_sram, 0, slot_a, "activation_A"),
            SRAMBuffer(act_sram, slot_a, slot_a, "activation_B"),
        ]
    else:
        buffers["weight"] = [SRAMBuffer(weight_sram, 0, weight_size, "weight")]
        buffers["activation"] = [SRAMBuffer(act_sram, 0, act_size, "activation")]

    buffers["output"] = [SRAMBuffer(output_sram, 0, output_size, "output")]
    return buffers


def create_gptvq_buffer_partitions(
    total_size_bytes: int,
    num_banks: int,
    bank_width_bytes: int,
    port_type: PortType,
    read_latency: int = 1,
    write_latency: int = 1,
    codebook_fraction: float = 0.10,
    index_fraction: float = 0.05,
    scale_fraction: float = 0.05,
    dequant_weight_fraction: float = 0.25,
    activation_fraction: float = 0.35,
    output_fraction: float = 0.20,
    double_buffer: bool = True,
    use_scaling: bool = False,
    codebook_num_banks: int = 0,
    codebook_port_type: PortType | None = None,
) -> dict[str, list[SRAMBuffer]]:
    """Create GPTVQ buffer partitions with separate BankedSRAM per buffer group.

    5 BankedSRAM groups:
    - codebook: dedicated SRAM with configurable num_banks (random access)
    - index+scale: shared SRAM (sequential access, loaded together)
    - dequant_weight: same role as weight buffer in non-VQ (systolic input)
    - activation: activation tiles
    - output: output accumulation

    With double buffering: index, scale, dequant_weight, activation get 2 slots each.
    Codebook and output are single-slot.
    """
    if not use_scaling:
        extra = scale_fraction / 2
        index_fraction += extra
        activation_fraction += extra
        scale_fraction = 0.0

    codebook_size = int(total_size_bytes * codebook_fraction)
    index_size = int(total_size_bytes * index_fraction)
    scale_size = int(total_size_bytes * scale_fraction)
    dequant_weight_size = int(total_size_bytes * dequant_weight_fraction)
    act_size = int(total_size_bytes * activation_fraction)
    output_size = total_size_bytes - codebook_size - index_size - scale_size - dequant_weight_size - act_size

    sram_args = dict(bank_width_bytes=bank_width_bytes, port_type=port_type,
                     read_latency=read_latency, write_latency=write_latency)

    # Codebook: separate config (random access, bank conflicts matter)
    cb_banks = codebook_num_banks if codebook_num_banks > 0 else num_banks
    cb_port = codebook_port_type if codebook_port_type is not None else port_type
    codebook_sram = _make_sram(codebook_size, num_banks=cb_banks,
                               bank_width_bytes=bank_width_bytes, port_type=cb_port,
                               read_latency=read_latency, write_latency=write_latency)

    # Index + scale: share one BankedSRAM (both sequential, loaded together)
    idx_scale_sram = _make_sram(index_size + scale_size, num_banks=num_banks, **sram_args)

    # Dequant weight, activation, output: each gets own BankedSRAM
    dq_sram = _make_sram(dequant_weight_size, num_banks=num_banks, **sram_args)
    act_sram = _make_sram(act_size, num_banks=num_banks, **sram_args)
    output_sram = _make_sram(output_size, num_banks=num_banks, **sram_args)

    buffers: dict[str, list[SRAMBuffer]] = {}

    buffers["codebook"] = [SRAMBuffer(codebook_sram, 0, codebook_size, "codebook")]

    if double_buffer:
        slot_idx = index_size // 2
        buffers["index"] = [
            SRAMBuffer(idx_scale_sram, 0, slot_idx, "index_A"),
            SRAMBuffer(idx_scale_sram, slot_idx, slot_idx, "index_B"),
        ]

        if use_scaling and scale_size > 0:
            slot_sc = scale_size // 2
            buffers["scale"] = [
                SRAMBuffer(idx_scale_sram, index_size, slot_sc, "scale_A"),
                SRAMBuffer(idx_scale_sram, index_size + slot_sc, slot_sc, "scale_B"),
            ]
        else:
            buffers["scale"] = [SRAMBuffer(idx_scale_sram, index_size, 0, "scale_none")]

        slot_dq = dequant_weight_size // 2
        buffers["dequant_weight"] = [
            SRAMBuffer(dq_sram, 0, slot_dq, "dequant_weight_A"),
            SRAMBuffer(dq_sram, slot_dq, slot_dq, "dequant_weight_B"),
        ]

        slot_act = act_size // 2
        buffers["activation"] = [
            SRAMBuffer(act_sram, 0, slot_act, "activation_A"),
            SRAMBuffer(act_sram, slot_act, slot_act, "activation_B"),
        ]
    else:
        buffers["index"] = [SRAMBuffer(idx_scale_sram, 0, index_size, "index")]

        if use_scaling and scale_size > 0:
            buffers["scale"] = [SRAMBuffer(idx_scale_sram, index_size, scale_size, "scale")]
        else:
            buffers["scale"] = [SRAMBuffer(idx_scale_sram, index_size, 0, "scale_none")]

        buffers["dequant_weight"] = [SRAMBuffer(dq_sram, 0, dequant_weight_size, "dequant_weight")]
        buffers["activation"] = [SRAMBuffer(act_sram, 0, act_size, "activation")]

    buffers["output"] = [SRAMBuffer(output_sram, 0, output_size, "output")]
    return buffers


def collect_srams(buffers: dict[str, list[SRAMBuffer]]) -> list[BankedSRAM]:
    """Collect unique BankedSRAM instances from a buffer dict."""
    seen: set[int] = set()
    srams: list[BankedSRAM] = []
    for buf_list in buffers.values():
        for buf in buf_list:
            sid = id(buf.sram)
            if sid not in seen:
                seen.add(sid)
                srams.append(buf.sram)
    return srams
