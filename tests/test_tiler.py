"""Tests for the tiler module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import DataTypeConfig, SRAMConfig, SystolicArrayConfig
from src.dataflow.tiler import Tiler


def make_tiler(
    array_rows=32,
    array_cols=32,
    sram_kb=512,
    num_banks=32,
    compute_dtype="INT8",
    double_buffer=True,
) -> Tiler:
    return Tiler(
        sram_config=SRAMConfig(total_size_kb=sram_kb, num_banks=num_banks),
        array_config=SystolicArrayConfig(rows=array_rows, cols=array_cols),
        dtype_config=DataTypeConfig(compute_dtype=compute_dtype),
        double_buffer=double_buffer,
    )


def test_tile_fits_in_array():
    """When matrix <= array size, tile should equal matrix size."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=512)
    tc = tiler.compute_tiles(M=16, N=16, K=64)

    assert tc.tile_m == 16
    assert tc.tile_n == 16
    assert tc.num_m_tiles == 1
    assert tc.num_n_tiles == 1


def test_tile_clips_to_array():
    """When matrix > array, tile_m/n should clip to array dims."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=512)
    tc = tiler.compute_tiles(M=128, N=64, K=256)

    assert tc.tile_m == 32
    assert tc.tile_n == 32
    assert tc.num_m_tiles == 4
    assert tc.num_n_tiles == 2


def test_tile_k_maximized():
    """tile_k should be as large as SRAM budget allows."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=512, double_buffer=True)
    tc = tiler.compute_tiles(M=256, N=256, K=1024)

    # With 512KB, double buffer:
    # weight_budget = 512*1024 * 0.4 / 2 = 52428 bytes per slot
    # act_budget = 512*1024 * 0.3 / 2 = 39321 bytes per slot
    # tile_m=32, tile_n=32, bpe=1 (INT8)
    # max_k_weight = 52428 / (32*1) = 1638
    # max_k_act = 39321 / (32*1) = 1228
    # tile_k = min(1024, 1638, 1228) = 1024
    assert tc.tile_k == 1024
    assert tc.num_k_tiles == 1


def test_tile_k_limited_by_sram():
    """When K is very large, tile_k should be limited by SRAM."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=64, double_buffer=True)
    tc = tiler.compute_tiles(M=256, N=256, K=4096)

    # With 64KB and double buffer, budget is much tighter
    assert tc.tile_k < 4096
    assert tc.num_k_tiles > 1

    # Verify tile fits in SRAM
    db_factor = 2
    assert tc.weight_tile_bytes <= (64 * 1024 * 0.4 / db_factor)
    assert tc.activation_tile_bytes <= (64 * 1024 * 0.3 / db_factor)


def test_tiles_cover_full_matrix():
    """All tiles should cover the entire matrix."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=128)
    M, N, K = 100, 80, 200
    tc = tiler.compute_tiles(M, N, K)

    assert tc.num_m_tiles * tc.tile_m >= M
    assert tc.num_n_tiles * tc.tile_n >= N
    assert tc.num_k_tiles * tc.tile_k >= K


def test_dram_traffic_estimate():
    """Verify DRAM traffic estimation."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=512)
    traffic = tiler.estimate_dram_traffic(M=256, N=256, K=256)

    assert traffic["total_read_bytes"] > 0
    assert traffic["total_write_bytes"] > 0
    assert traffic["total_bytes"] == traffic["total_read_bytes"] + traffic["total_write_bytes"]


# --- Dataflow-specific tiling tests ---

def test_ws_tile_k_maps_to_rows():
    """In WS, tile_k should be clipped to array_rows."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=512)
    tc = tiler.compute_tiles(M=256, N=256, K=256, dataflow="WS")

    assert tc.tile_k <= 32
    assert tc.tile_n <= 32


def test_is_tile_k_maps_to_rows():
    """In IS, tile_k should be clipped to array_rows."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=512)
    tc = tiler.compute_tiles(M=256, N=256, K=256, dataflow="IS")

    assert tc.tile_k <= 32
    assert tc.tile_m <= 32


def test_os_tile_m_n_maps_to_array():
    """In OS, tile_m/n should be clipped to array rows/cols."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=512)
    tc = tiler.compute_tiles(M=256, N=256, K=256, dataflow="OS")

    assert tc.tile_m <= 32
    assert tc.tile_n <= 32


def test_ws_tiles_cover_full_matrix():
    """WS tiles should cover the entire matrix."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=128)
    M, N, K = 100, 80, 200
    tc = tiler.compute_tiles(M, N, K, dataflow="WS")

    assert tc.num_m_tiles * tc.tile_m >= M
    assert tc.num_n_tiles * tc.tile_n >= N
    assert tc.num_k_tiles * tc.tile_k >= K


def test_is_tiles_cover_full_matrix():
    """IS tiles should cover the entire matrix."""
    tiler = make_tiler(array_rows=32, array_cols=32, sram_kb=128)
    M, N, K = 100, 80, 200
    tc = tiler.compute_tiles(M, N, K, dataflow="IS")

    assert tc.num_m_tiles * tc.tile_m >= M
    assert tc.num_n_tiles * tc.tile_n >= N
    assert tc.num_k_tiles * tc.tile_k >= K
