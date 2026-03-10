"""Tests for the systolic array model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from npu_sim.compute.systolic_array import SystolicArray
from npu_sim.core.datatypes import AccumulatorType, DataType


def test_analytical_cycles_os_basic():
    """Test basic OS cycle formula for a tile that fits in the array."""
    sa = SystolicArray(rows=32, cols=32)

    # OS: SR=M=32, SC=N=32, T=K=64
    cycles = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64, dataflow="OS")

    # fill = 32 + 32 - 1 = 63
    # compute = 64
    # drain = 63
    # total = 63 + 64 + 63 = 190
    assert cycles.pipeline_fill_cycles == 63
    assert cycles.compute_cycles == 64
    assert cycles.pipeline_drain_cycles == 63
    assert cycles.total_cycles == 190


def test_analytical_cycles_os_small_tile():
    """Test OS with a tile smaller than the array."""
    sa = SystolicArray(rows=32, cols=32)

    cycles = sa.analytical_cycles(tile_m=16, tile_n=8, tile_k=32, dataflow="OS")

    # fill = 16 + 8 - 1 = 23
    # compute = 32
    # drain = 23
    # total = 23 + 32 + 23 = 78
    assert cycles.pipeline_fill_cycles == 23
    assert cycles.compute_cycles == 32
    assert cycles.pipeline_drain_cycles == 23
    assert cycles.total_cycles == 78


def test_analytical_cycles_os_multi_pass():
    """Test OS with a tile larger than the array (requires multiple passes)."""
    sa = SystolicArray(rows=8, cols=8)

    # tile 16x16 on 8x8 array -> SR=16,SC=16 -> 2x2 = 4 passes
    cycles = sa.analytical_cycles(tile_m=16, tile_n=16, tile_k=32, dataflow="OS")

    # Each pass: 8x8, fill=15, compute=32, drain=15, total=62
    assert cycles.pipeline_fill_cycles == 4 * 15
    assert cycles.compute_cycles == 4 * 32
    assert cycles.pipeline_drain_cycles == 4 * 15
    assert cycles.total_cycles == 4 * 62


def test_analytical_cycles_os_non_divisible():
    """Test OS with tile dimensions not divisible by array dimensions."""
    sa = SystolicArray(rows=8, cols=8)

    # tile 10x10 on 8x8 -> 2x2 passes
    # Pass (0,0): 8x8, fill=15, comp=16, drain=15 = 46
    # Pass (0,1): 8x2, fill=9, comp=16, drain=9 = 34
    # Pass (1,0): 2x8, fill=9, comp=16, drain=9 = 34
    # Pass (1,1): 2x2, fill=3, comp=16, drain=3 = 22
    cycles = sa.analytical_cycles(tile_m=10, tile_n=10, tile_k=16, dataflow="OS")

    expected_total = 46 + 34 + 34 + 22
    assert cycles.total_cycles == expected_total


def test_analytical_cycles_ws():
    """Test WS cycle formula: SR=K, SC=N, T=M."""
    sa = SystolicArray(rows=32, cols=32)

    # WS: SR=K=64, SC=N=32, T=M=32
    # SR passes = ceil(64/32) = 2, SC passes = 1
    # Pass 0: eff_sr=32, eff_sc=32 → fill=63, compute=32, drain=63 → 158
    # Pass 1: eff_sr=32, eff_sc=32 → fill=63, compute=32, drain=63 → 158
    # total = 316
    cycles = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64, dataflow="WS")

    assert cycles.pipeline_fill_cycles == 2 * 63
    assert cycles.compute_cycles == 2 * 32
    assert cycles.pipeline_drain_cycles == 2 * 63
    assert cycles.total_cycles == 2 * 158


def test_analytical_cycles_is():
    """Test IS cycle formula: SR=K, SC=M, T=N."""
    sa = SystolicArray(rows=32, cols=32)

    # IS: SR=K=64, SC=M=32, T=N=16
    # SR passes = ceil(64/32) = 2, SC passes = 1
    # Pass 0: eff_sr=32, eff_sc=32 → fill=63, compute=16, drain=63 → 142
    # Pass 1: eff_sr=32, eff_sc=32 → fill=63, compute=16, drain=63 → 142
    # total = 284
    cycles = sa.analytical_cycles(tile_m=32, tile_n=16, tile_k=64, dataflow="IS")

    assert cycles.pipeline_fill_cycles == 2 * 63
    assert cycles.compute_cycles == 2 * 16
    assert cycles.pipeline_drain_cycles == 2 * 63
    assert cycles.total_cycles == 2 * 142


def test_analytical_cycles_default_is_os():
    """Default dataflow should be OS for backward compatibility."""
    sa = SystolicArray(rows=32, cols=32)

    cycles_default = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64)
    cycles_os = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64, dataflow="OS")

    assert cycles_default.total_cycles == cycles_os.total_cycles


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
