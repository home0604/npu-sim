"""End-to-end tests for the NPU simulator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from npu_sim.core.config import NPUConfig
from npu_sim.sim.simulator import NPUSimulator


def make_simulator(rows=32, cols=32, sram_kb=512, dtype="INT8") -> NPUSimulator:
    config = NPUConfig()
    config.systolic.rows = rows
    config.systolic.cols = cols
    config.sram.total_size_kb = sram_kb
    config.dtype.compute_dtype = dtype
    return NPUSimulator(config)


def test_small_matmul():
    """Small MatMul that fits in a single tile."""
    sim = make_simulator(rows=32, cols=32, sram_kb=512)
    result = sim.run_matmul(M=32, N=32, K=64)

    assert result["total_cycles"] > 0
    assert result["compute"]["total_mac_ops"] == 32 * 32 * 64
    assert result["compute"]["tiles_processed"] >= 1


def test_large_matmul():
    """Large MatMul requiring multiple tiles."""
    sim = make_simulator(rows=32, cols=32, sram_kb=512)
    result = sim.run_matmul(M=256, N=256, K=256)

    assert result["total_cycles"] > 0
    assert result["compute"]["total_mac_ops"] == 256 * 256 * 256
    assert result["compute"]["tiles_processed"] > 1


def test_matmul_cycle_sanity():
    """Verify cycle count is in a reasonable range."""
    sim = make_simulator(rows=32, cols=32, sram_kb=512)
    result = sim.run_matmul(M=32, N=32, K=64)

    # Minimum: just compute cycles = 64 + 2*(32+32-1) = 190
    # With DRAM loading, should be more
    assert result["total_cycles"] >= 190


def test_matmul_larger_array_is_faster():
    """A larger array should complete in fewer cycles (compute-bound case)."""
    sim_small = make_simulator(rows=16, cols=16, sram_kb=512)
    sim_large = make_simulator(rows=64, cols=64, sram_kb=512)

    r_small = sim_small.run_matmul(M=128, N=128, K=128)
    r_large = sim_large.run_matmul(M=128, N=128, K=128)

    # Larger array should have fewer compute cycles
    assert r_large["compute"]["total_compute_cycles"] <= r_small["compute"]["total_compute_cycles"]


def test_double_buffer_reduces_stalls():
    """Double buffering should reduce or eliminate stall cycles."""
    # With double buffer
    config_db = NPUConfig()
    config_db.double_buffer.enabled = True
    sim_db = NPUSimulator(config_db)
    r_db = sim_db.run_matmul(M=256, N=256, K=256)

    # Without double buffer
    config_no = NPUConfig()
    config_no.double_buffer.enabled = False
    sim_no = NPUSimulator(config_no)
    r_no = sim_no.run_matmul(M=256, N=256, K=256)

    # Double buffered should have fewer or equal total cycles
    # (or at least some double buffer hits)
    assert r_db["total_cycles"] <= r_no["total_cycles"]


def test_config_from_yaml():
    """Test loading config from YAML and running simulation."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if config_path.exists():
        sim = NPUSimulator(config_path)
        result = sim.run_matmul(M=64, N=64, K=128)
        assert result["total_cycles"] > 0


def test_attention_workload():
    """Test running attention workload end-to-end."""
    sim = make_simulator(rows=32, cols=32, sram_kb=512)
    result = sim.run_attention(seq_len=64, hidden_dim=256, num_heads=4)

    assert result["total_cycles"] > 0
    assert result["compute"]["tiles_processed"] > 0


def test_transformer_layer():
    """Test running a full transformer layer."""
    sim = make_simulator(rows=32, cols=32, sram_kb=512)
    result = sim.run_transformer_layer(
        seq_len=32, hidden_dim=128, num_heads=4
    )

    assert result["total_cycles"] > 0
    assert result["compute"]["tiles_processed"] > 0


def test_different_dtypes():
    """INT8 should process more data per byte than FP16."""
    sim_int8 = make_simulator(dtype="INT8", sram_kb=256)
    sim_fp16 = make_simulator(dtype="FP16", sram_kb=256)

    r_int8 = sim_int8.run_matmul(M=128, N=128, K=128)
    r_fp16 = sim_fp16.run_matmul(M=128, N=128, K=128)

    # INT8 has smaller tiles in bytes -> larger tile_k -> less DRAM traffic
    assert r_int8["memory"]["dram_read_bytes"] <= r_fp16["memory"]["dram_read_bytes"]
