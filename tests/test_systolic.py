"""Tests for the systolic array model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from npu_sim.compute.systolic_array import SystolicArray
from npu_sim.core.datatypes import AccumulatorType, DataType


def test_analytical_cycles_basic():
    """Test basic cycle formula for a tile that fits in the array."""
    sa = SystolicArray(rows=32, cols=32)

    # tile fits exactly in array: 32x32, K=64
    cycles = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64)

    # fill = 32 + 32 - 1 = 63
    # compute = 64
    # drain = 63
    # total = 63 + 64 + 63 = 190
    assert cycles.pipeline_fill_cycles == 63
    assert cycles.compute_cycles == 64
    assert cycles.pipeline_drain_cycles == 63
    assert cycles.total_cycles == 190


def test_analytical_cycles_small_tile():
    """Test with a tile smaller than the array."""
    sa = SystolicArray(rows=32, cols=32)

    cycles = sa.analytical_cycles(tile_m=16, tile_n=8, tile_k=32)

    # fill = 16 + 8 - 1 = 23
    # compute = 32
    # drain = 23
    # total = 23 + 32 + 23 = 78
    assert cycles.pipeline_fill_cycles == 23
    assert cycles.compute_cycles == 32
    assert cycles.pipeline_drain_cycles == 23
    assert cycles.total_cycles == 78


def test_analytical_cycles_multi_pass():
    """Test with a tile larger than the array (requires multiple passes)."""
    sa = SystolicArray(rows=8, cols=8)

    # tile 16x16 on 8x8 array -> 2x2 = 4 passes
    cycles = sa.analytical_cycles(tile_m=16, tile_n=16, tile_k=32)

    # Each pass: 8x8, fill=15, compute=32, drain=15, total=62
    # 4 passes: fill=60, compute=128, drain=60, total=248
    assert cycles.pipeline_fill_cycles == 4 * 15
    assert cycles.compute_cycles == 4 * 32
    assert cycles.pipeline_drain_cycles == 4 * 15
    assert cycles.total_cycles == 4 * 62


def test_analytical_cycles_non_divisible():
    """Test with tile dimensions not divisible by array dimensions."""
    sa = SystolicArray(rows=8, cols=8)

    # tile 10x10 on 8x8 -> 2x2 passes
    # Pass (0,0): 8x8, fill=15, comp=16, drain=15 = 46
    # Pass (0,1): 8x2, fill=9, comp=16, drain=9 = 34
    # Pass (1,0): 2x8, fill=9, comp=16, drain=9 = 34
    # Pass (1,1): 2x2, fill=3, comp=16, drain=3 = 22
    cycles = sa.analytical_cycles(tile_m=10, tile_n=10, tile_k=16)

    expected_total = 46 + 34 + 34 + 22
    assert cycles.total_cycles == expected_total


def test_compute_tile_correctness():
    """Test that PyTorch computation produces correct results."""
    sa = SystolicArray(rows=32, cols=32, dtype=DataType.FP32, acc_dtype=AccumulatorType.FP32)

    weight = torch.randn(16, 32)
    activation = torch.randn(32, 8)

    result = sa.compute_tile(weight, activation)

    expected = torch.matmul(weight.float(), activation.float())
    assert torch.allclose(result.output, expected, atol=1e-5)
    assert result.tile_m == 16
    assert result.tile_n == 8
    assert result.tile_k == 32
    assert result.cycles.total_cycles > 0


def test_weight_load_cycles():
    """Test weight load cycle calculation."""
    sa = SystolicArray(rows=32, cols=32)

    # 32x64 weight tile, 1 pass
    assert sa.weight_load_cycles(32, 64) == 64

    # 64x64 weight tile on 32-row array, 2 passes
    assert sa.weight_load_cycles(64, 64) == 2 * 64
