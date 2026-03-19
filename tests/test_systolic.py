"""Tests for the systolic array model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from compute.systolic_array import SystolicArray
from core.datatypes import AccumulatorType, DataType


def test_analytical_cycles_os_basic():
    """Test OS cycle formula: SR=M, SC=N, T=K → SR+SC+T-2."""
    sa = SystolicArray(rows=32, cols=32)

    # OS: SR=M=32, SC=N=32, T=K=64
    # preload=0, fill=32+32-2=62, compute=64, drain=0
    # total = 0 + 62 + 64 = 126
    cycles = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64, dataflow="OS")

    assert cycles.preload_cycles == 0
    assert cycles.pipeline_fill_cycles == 62
    assert cycles.compute_cycles == 64
    assert cycles.pipeline_drain_cycles == 0
    assert cycles.total_cycles == 126


def test_analytical_cycles_os_small_tile():
    """Test OS with a tile smaller than the array."""
    sa = SystolicArray(rows=32, cols=32)

    # OS: SR=M=16, SC=N=8, T=K=32
    # preload=0, fill=16+8-2=22, compute=32, drain=0 → 54
    cycles = sa.analytical_cycles(tile_m=16, tile_n=8, tile_k=32, dataflow="OS")

    assert cycles.preload_cycles == 0
    assert cycles.pipeline_fill_cycles == 22
    assert cycles.compute_cycles == 32
    assert cycles.pipeline_drain_cycles == 0
    assert cycles.total_cycles == 54


def test_analytical_cycles_os_multi_pass():
    """Test OS with a tile larger than the array (requires multiple passes)."""
    sa = SystolicArray(rows=8, cols=8)

    # tile 16x16 on 8x8 array → SR=16, SC=16 → 2x2 = 4 passes
    # each pass (eff_sr=8, eff_sc=8): preload=0, fill=14, compute=32, drain=0 → 46
    cycles = sa.analytical_cycles(tile_m=16, tile_n=16, tile_k=32, dataflow="OS")

    assert cycles.preload_cycles == 0
    assert cycles.pipeline_fill_cycles == 4 * 14
    assert cycles.compute_cycles == 4 * 32
    assert cycles.pipeline_drain_cycles == 0
    assert cycles.total_cycles == 4 * 46


def test_analytical_cycles_os_non_divisible():
    """Test OS with tile dimensions not divisible by array dimensions."""
    sa = SystolicArray(rows=8, cols=8)

    # tile 10x10 on 8x8 → 2x2 passes
    # Pass (0,0): eff_sr=8, eff_sc=8 → fill=14, compute=16 → 30
    # Pass (0,1): eff_sr=8, eff_sc=2 → fill=8, compute=16 → 24
    # Pass (1,0): eff_sr=2, eff_sc=8 → fill=8, compute=16 → 24
    # Pass (1,1): eff_sr=2, eff_sc=2 → fill=2, compute=16 → 18
    cycles = sa.analytical_cycles(tile_m=10, tile_n=10, tile_k=16, dataflow="OS")

    expected_total = 30 + 24 + 24 + 18
    assert cycles.total_cycles == expected_total


def test_analytical_cycles_ws():
    """Test WS cycle formula: SR=K, SC=N, T=M → 2*SR+SC+T-2."""
    sa = SystolicArray(rows=32, cols=32)

    # WS: SR=K=64, SC=N=32, T=M=32
    # sr_passes = ceil(64/32) = 2, sc_passes = 1
    # each pass (eff_sr=32, eff_sc=32): preload=32, fill=62, compute=32 → 126
    # total = 2 * 126 = 252
    cycles = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64, dataflow="WS")

    assert cycles.preload_cycles == 2 * 32
    assert cycles.pipeline_fill_cycles == 2 * 62
    assert cycles.compute_cycles == 2 * 32
    assert cycles.pipeline_drain_cycles == 0
    assert cycles.total_cycles == 2 * 126


def test_analytical_cycles_is():
    """Test IS cycle formula: SR=K, SC=M, T=N → 2*SR+SC+T-2."""
    sa = SystolicArray(rows=32, cols=32)

    # IS: SR=K=64, SC=M=32, T=N=16
    # sr_passes = ceil(64/32) = 2, sc_passes = 1
    # each pass (eff_sr=32, eff_sc=32): preload=32, fill=62, compute=16 → 110
    # total = 2 * 110 = 220
    cycles = sa.analytical_cycles(tile_m=32, tile_n=16, tile_k=64, dataflow="IS")

    assert cycles.preload_cycles == 2 * 32
    assert cycles.pipeline_fill_cycles == 2 * 62
    assert cycles.compute_cycles == 2 * 16
    assert cycles.pipeline_drain_cycles == 0
    assert cycles.total_cycles == 2 * 110


def test_analytical_cycles_default_is_os():
    """Default dataflow should be OS for backward compatibility."""
    sa = SystolicArray(rows=32, cols=32)

    cycles_default = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64)
    cycles_os = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64, dataflow="OS")

    assert cycles_default.total_cycles == cycles_os.total_cycles


def test_ws_more_cycles_than_os():
    """WS should have more cycles than OS due to preload cost."""
    sa = SystolicArray(rows=32, cols=32)

    cycles_os = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64, dataflow="OS")
    cycles_ws = sa.analytical_cycles(tile_m=32, tile_n=32, tile_k=64, dataflow="WS")
    cycles_is = sa.analytical_cycles(tile_m=32, tile_n=16, tile_k=64, dataflow="IS")

    assert cycles_ws.total_cycles > cycles_os.total_cycles
    assert cycles_is.total_cycles > cycles_os.total_cycles


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
