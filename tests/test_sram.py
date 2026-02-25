"""Tests for the banked SRAM model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from npu_sim.memory.sram import BankedSRAM, PortType


def test_single_port_no_conflict():
    """Reads to different banks should complete in parallel."""
    sram = BankedSRAM(
        size_bytes=4096,
        num_banks=4,
        bank_width_bytes=64,
        port_type=PortType.SINGLE,
        read_latency=1,
    )

    # Read from bank 0 (addr 0) and bank 1 (addr 64) at same cycle
    done1 = sram.read(0, 64, cycle=0)
    done2 = sram.read(64, 64, cycle=0)

    assert done1 == 1  # 0 + 1
    assert done2 == 1  # 0 + 1 (different bank, no conflict)
    assert sram.stats.bank_conflicts == 0


def test_single_port_bank_conflict():
    """Two accesses to the same bank should serialize."""
    sram = BankedSRAM(
        size_bytes=4096,
        num_banks=4,
        bank_width_bytes=64,
        port_type=PortType.SINGLE,
        read_latency=1,
    )

    # Two reads to bank 0
    done1 = sram.read(0, 64, cycle=0)  # bank 0
    done2 = sram.read(256, 64, cycle=0)  # bank 0 again (256/64=4, 4%4=0)

    assert done1 == 1
    assert done2 == 2  # delayed by conflict
    assert sram.stats.bank_conflicts == 1


def test_dual_port_read_write_no_conflict():
    """Dual-port: one read and one write to the same bank should not conflict."""
    sram = BankedSRAM(
        size_bytes=4096,
        num_banks=4,
        bank_width_bytes=64,
        port_type=PortType.DUAL,
        read_latency=1,
        write_latency=1,
    )

    # Read and write to bank 0 simultaneously
    done_r = sram.read(0, 64, cycle=0)
    done_w = sram.write(0, 64, cycle=0)

    assert done_r == 1
    assert done_w == 1
    assert sram.stats.port_conflicts == 0


def test_dual_port_read_conflict():
    """Dual-port: two reads to the same bank should conflict."""
    sram = BankedSRAM(
        size_bytes=4096,
        num_banks=4,
        bank_width_bytes=64,
        port_type=PortType.DUAL,
        read_latency=1,
    )

    done1 = sram.read(0, 64, cycle=0)
    done2 = sram.read(256, 64, cycle=0)  # same bank

    assert done1 == 1
    assert done2 == 2
    assert sram.stats.port_conflicts == 1


def test_multi_bank_access():
    """Access spanning multiple banks should touch all relevant banks."""
    sram = BankedSRAM(
        size_bytes=4096,
        num_banks=4,
        bank_width_bytes=64,
        port_type=PortType.SINGLE,
        read_latency=1,
    )

    # 128 bytes starting at addr 0: banks 0 and 1
    done = sram.read(0, 128, cycle=0)
    assert done == 1  # parallel access to banks 0 and 1

    assert sram.stats.total_reads == 1
    assert sram.stats.total_read_bytes == 128


def test_interleaved_banking():
    """Verify interleaved bank address mapping."""
    sram = BankedSRAM(
        size_bytes=4096,
        num_banks=8,
        bank_width_bytes=64,
        port_type=PortType.SINGLE,
    )

    assert sram._addr_to_bank(0) == 0
    assert sram._addr_to_bank(64) == 1
    assert sram._addr_to_bank(128) == 2
    assert sram._addr_to_bank(512) == 0  # wraps: 512/64=8, 8%8=0
