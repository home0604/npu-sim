"""VQ-specific tests: dequant unit, tiler, dataflow, and E2E simulation."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from core.config import NPUConfig, load_config
from compute.dequant_unit import DequantizationUnit
from dataflow.vq_tiler import VQTiler
from dataflow.vq_dataflow import VQDataflow
from sim.simulator import NPUSimulator

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vq_config(**overrides) -> NPUConfig:
    """Load vq.yaml (inherits default.yaml) and apply optional overrides."""
    cfg = load_config(_CONFIGS_DIR / "vq.yaml")
    for k, v in overrides.items():
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def _make_vq_sim(**overrides) -> NPUSimulator:
    return NPUSimulator(_make_vq_config(**overrides))


# ===========================================================================
# 1. DequantizationUnit tests
# ===========================================================================

class TestDequantUnit:
    """Dequant cycle calculation and functional correctness."""

    def test_basic_cycle_count(self):
        """pipeline_stages + num_lookups * max(read_lat, write_lat)."""
        cfg = _make_vq_config()
        unit = DequantizationUnit(cfg.vq, cfg.sram)
        d = cfg.vq.vector_dim
        cycles = unit.dequant_cycles(32, 64)
        num_lookups = 32 * math.ceil(64 / d)
        latency = unit._latency_per_lookup()
        expected = cfg.vq.dequant_pipeline_stages + num_lookups * latency
        assert cycles == expected

    def test_latency_per_lookup_uses_max(self):
        """cyc/lookup = max(read_lat, write_lat)."""
        cfg = _make_vq_config()
        cfg.sram.write_latency_cycles = 5  # force write > read
        unit = DequantizationUnit(cfg.vq, cfg.sram)
        assert unit._latency_per_lookup() == 5
        assert unit._read_latency_per_lookup() < 5

    def test_codebook_bank_override(self):
        """codebook_sram.num_banks overrides global num_banks."""
        cfg = _make_vq_config()
        cfg.sram.num_banks = 32
        cfg.sram.codebook_sram.num_banks = 4
        unit = DequantizationUnit(cfg.vq, cfg.sram)
        assert unit.sram_n_banks == 4

    def test_codebook_bank_fallback(self):
        """codebook_sram.num_banks=0 falls back to global."""
        cfg = _make_vq_config()
        cfg.sram.num_banks = 32
        cfg.sram.codebook_sram.num_banks = 0
        unit = DequantizationUnit(cfg.vq, cfg.sram)
        assert unit.sram_n_banks == 32

    def test_read_latency_increases_with_stages(self):
        """More RVQ stages -> higher read latency when bandwidth is tight."""
        cfg = _make_vq_config()
        cfg.sram.codebook_sram.num_banks = 4
        cfg.sram.bank_width_bytes = 8
        cfg.vq.vector_dim = 8

        cfg.vq.num_stages = 1
        unit1 = DequantizationUnit(cfg.vq, cfg.sram)
        lat1 = unit1._read_latency_per_lookup()

        cfg.vq.num_stages = 4
        unit4 = DequantizationUnit(cfg.vq, cfg.sram)
        lat4 = unit4._read_latency_per_lookup()

        assert lat4 >= lat1

    def test_dequantize_functional(self):
        """Functional dequantization matches manual lookup."""
        cfg = _make_vq_config(**{"vq.vector_dim": 2, "vq.codebook_size": 16})
        unit = DequantizationUnit(cfg.vq, cfg.sram)

        R, d = 16, 2
        codebook = torch.randn(R, d)
        tile_m, tile_k = 4, 8
        num_groups = tile_k // d
        indices = torch.randint(0, R, (tile_m, num_groups))

        result = unit.dequantize(indices, codebook, tile_m, tile_k)

        # Manual
        expected = torch.zeros(tile_m, tile_k)
        for i in range(tile_m):
            for j in range(num_groups):
                expected[i, j * d:(j + 1) * d] = codebook[indices[i, j]]

        assert torch.allclose(result.output, expected)
        assert result.num_vectors == tile_m * num_groups

    def test_dequantize_with_scaling(self):
        """Functional dequantization with scale and zero_point."""
        cfg = _make_vq_config(**{"vq.use_scaling": True, "vq.vector_dim": 2, "vq.codebook_size": 16})
        unit = DequantizationUnit(cfg.vq, cfg.sram)

        R, d = 16, 2
        codebook = torch.randn(R, d)
        tile_m, tile_k = 4, 8
        num_groups = tile_k // d
        indices = torch.randint(0, R, (tile_m, num_groups))
        scales = torch.rand(num_groups) + 0.5
        zero_points = torch.randn(num_groups) * 0.01

        result = unit.dequantize(indices, codebook, tile_m, tile_k,
                                 scales=scales, zero_points=zero_points)

        # Manual
        expected = torch.zeros(tile_m, tile_k)
        for i in range(tile_m):
            for j in range(num_groups):
                v = codebook[indices[i, j]]
                expected[i, j * d:(j + 1) * d] = scales[j] * v + zero_points[j]

        assert torch.allclose(result.output, expected, atol=1e-5)


# ===========================================================================
# 2. VQTiler tests
# ===========================================================================

class TestVQTiler:
    """Tile sizing constraints for VQ."""

    def test_tile_k_is_multiple_of_vector_dim(self):
        """tile_k must be a multiple of d."""
        cfg = _make_vq_config()
        tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
        tc = tiler.compute_tiles(128, 128, 256)
        assert tc.tile_k % cfg.vq.vector_dim == 0

    def test_tile_k_multiple_of_d_various(self):
        """tile_k divisible by d for different vector_dim values."""
        for d in [2, 4, 8]:
            cfg = _make_vq_config(**{"vq.vector_dim": d})
            tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
            tc = tiler.compute_tiles(64, 64, 128)
            assert tc.tile_k % d == 0, f"d={d}: tile_k={tc.tile_k}"

    def test_single_tile(self):
        """Small problem fits in one tile."""
        cfg = _make_vq_config()
        tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
        tc = tiler.compute_tiles(32, 32, 64)
        assert tc.num_m_tiles == 1
        assert tc.num_n_tiles == 1
        assert tc.total_tiles >= 1

    def test_multi_tile(self):
        """Large problem requires multiple tiles."""
        cfg = _make_vq_config()
        tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
        tc = tiler.compute_tiles(256, 256, 512)
        assert tc.total_tiles > 1

    def test_codebook_tile_bytes(self):
        """codebook_tile_bytes = R * d * entry_bytes."""
        cfg = _make_vq_config()
        tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
        tc = tiler.compute_tiles(64, 64, 128)
        R = cfg.vq.codebook_size
        d = cfg.vq.vector_dim
        entry_bytes = cfg.vq.codebook_entry_bytes
        assert tc.codebook_tile_bytes == R * d * entry_bytes

    def test_double_buffer_reduces_tile_k(self):
        """Double buffering halves per-slot budget -> smaller or equal tile_k."""
        cfg_db = _make_vq_config()
        tiler_db = VQTiler(cfg_db.sram, cfg_db.systolic, cfg_db.dtype, cfg_db.vq, double_buffer=True)
        tc_db = tiler_db.compute_tiles(128, 128, 512)

        cfg_sb = _make_vq_config()
        tiler_sb = VQTiler(cfg_sb.sram, cfg_sb.systolic, cfg_sb.dtype, cfg_sb.vq, double_buffer=False)
        tc_sb = tiler_sb.compute_tiles(128, 128, 512)

        assert tc_db.tile_k <= tc_sb.tile_k


# ===========================================================================
# 3. VQDataflow tests
# ===========================================================================

class TestVQDataflow:
    """Schedule generation and load flag correctness for OS/WS/IS."""

    def _make_dataflow(self, dataflow="OS", **overrides):
        cfg = _make_vq_config(**overrides)
        cfg.systolic.dataflow = dataflow
        tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
        return VQDataflow(tiler, cfg.vq), cfg

    def test_schedule_length(self):
        """Schedule has correct number of tile operations."""
        df, cfg = self._make_dataflow()
        tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
        tc = tiler.compute_tiles(64, 64, 128)
        _, schedule = df.generate_schedule(64, 64, 128, dataflow="OS")
        assert len(schedule) == tc.total_tiles

    def test_os_load_codebook_on_first_k(self):
        """OS: codebook loaded on first k-tile only."""
        df, _ = self._make_dataflow("OS")
        _, schedule = df.generate_schedule(64, 64, 128, dataflow="OS")
        for op in schedule:
            if op.k_idx == 0:
                assert op.load_codebook is True
            else:
                assert op.load_codebook is False

    def test_os_store_on_last_k(self):
        """OS: output stored on last k-tile only."""
        df, cfg = self._make_dataflow("OS")
        tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
        tc = tiler.compute_tiles(64, 64, 128)
        _, schedule = df.generate_schedule(64, 64, 128, dataflow="OS")
        for op in schedule:
            if op.k_idx == tc.num_k_tiles - 1:
                assert op.store_output is True
            else:
                assert op.store_output is False

    def test_ws_indices_loaded_once_per_mk(self):
        """WS: indices loaded only on n==0 (once per (m,k) pair).
        Need num_n_tiles > 1 to observe reuse across N."""
        df, cfg = self._make_dataflow("WS")
        tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
        tc = tiler.compute_tiles(64, 128, 64)
        assert tc.num_n_tiles > 1, "Need multiple n-tiles to verify WS reuse"
        _, schedule = df.generate_schedule(64, 128, 64, dataflow="WS")
        for op in schedule:
            expected = op.n_idx == 0
            assert op.load_indices == expected, \
                f"WS: m={op.m_idx},n={op.n_idx},k={op.k_idx}: load_indices={op.load_indices}, expected={expected}"

    def test_is_activation_loaded_once_per_kn(self):
        """IS: activation loaded only on m==0 (once per (k,n) pair).
        Need num_m_tiles > 1 to observe reuse across M."""
        df, cfg = self._make_dataflow("IS")
        tiler = VQTiler(cfg.sram, cfg.systolic, cfg.dtype, cfg.vq)
        tc = tiler.compute_tiles(128, 64, 64)
        assert tc.num_m_tiles > 1, "Need multiple m-tiles to verify IS reuse"
        _, schedule = df.generate_schedule(128, 64, 64, dataflow="IS")
        for op in schedule:
            expected = op.m_idx == 0
            assert op.load_activation == expected, \
                f"IS: m={op.m_idx},n={op.n_idx},k={op.k_idx}: load_activation={op.load_activation}"

    def test_accumulate_flag(self):
        """accumulate=True when k_idx > 0."""
        for dataflow in ["OS", "WS", "IS"]:
            df, _ = self._make_dataflow(dataflow)
            _, schedule = df.generate_schedule(64, 64, 128, dataflow=dataflow)
            for op in schedule:
                assert op.accumulate == (op.k_idx > 0), \
                    f"{dataflow}: k_idx={op.k_idx}, accumulate={op.accumulate}"

    def test_all_dataflows_same_tile_count(self):
        """OS/WS/IS produce same number of tile ops for same problem."""
        counts = []
        for dataflow in ["OS", "WS", "IS"]:
            df, _ = self._make_dataflow(dataflow)
            _, schedule = df.generate_schedule(64, 64, 128, dataflow=dataflow)
            counts.append(len(schedule))
        assert counts[0] == counts[1] == counts[2]


# ===========================================================================
# 4. E2E VQ simulation tests
# ===========================================================================

class TestVQE2E:
    """End-to-end VQ simulation through NPUSimulator."""

    def test_vq_matmul_basic(self):
        """VQ matmul produces valid cycle count and stats."""
        sim = _make_vq_sim()
        result = sim.run_matmul(M=64, N=64, K=128)

        assert result["total_cycles"] > 0
        assert result["compute"]["total_mac_ops"] == 64 * 64 * 128
        assert result["compute"]["tiles_processed"] >= 1
        assert "vq" in result

    def test_vq_stats_present(self):
        """VQ-specific stats are reported."""
        sim = _make_vq_sim()
        result = sim.run_matmul(M=64, N=64, K=128)

        vq = result["vq"]
        assert vq["total_dequant_cycles"] > 0
        assert vq["total_vectors_dequantized"] > 0
        assert vq["compression_ratio"] > 0

    def test_vq_correctness(self):
        """VQ correctness check passes."""
        sim = _make_vq_sim()
        result = sim.run_matmul(M=32, N=32, K=64, check_correctness=True)
        assert result["correctness"] == "pass"

    def test_vq_correctness_multi_tile(self):
        """VQ correctness with multiple tiles."""
        sim = _make_vq_sim()
        result = sim.run_matmul(M=128, N=128, K=256, check_correctness=True)
        assert result["correctness"] == "pass"
        assert result["compute"]["tiles_processed"] > 1

    def test_vq_correctness_with_scaling(self):
        """VQ correctness with scaling enabled."""
        sim = _make_vq_sim(**{"vq.use_scaling": True})
        result = sim.run_matmul(M=32, N=32, K=64, check_correctness=True)
        assert result["correctness"] == "pass"

    def test_vq_dequant_adds_cycles(self):
        """VQ total_cycles > compute_cycles (dequant overhead exists)."""
        sim = _make_vq_sim()
        result = sim.run_matmul(M=64, N=64, K=128)
        assert result["total_cycles"] > result["compute"]["total_compute_cycles"]

    def test_vq_compression_ratio(self):
        """Compression ratio is reasonable for 4-bit index, d=2, FP16 codebook."""
        sim = _make_vq_sim()
        result = sim.run_matmul(M=128, N=128, K=256)
        ratio = result["vq"]["compression_ratio"]
        # 4-bit VQ with d=2 should compress relative to INT8 weights
        assert ratio > 1.0

    def test_vq_all_dataflows(self):
        """VQ works with OS, WS, IS dataflows."""
        for df in ["OS", "WS", "IS"]:
            sim = _make_vq_sim(**{"systolic.dataflow": df})
            result = sim.run_matmul(M=64, N=64, K=128)
            assert result["total_cycles"] > 0, f"Dataflow {df} failed"
            assert "vq" in result

    def test_vq_vs_non_vq_has_dequant_overhead(self):
        """VQ simulation includes dequant overhead not present in non-VQ."""
        cfg_vq = _make_vq_config()
        sim_vq = NPUSimulator(cfg_vq)
        result_vq = sim_vq.run_matmul(M=64, N=64, K=128)

        cfg_novq = NPUConfig()
        cfg_novq.systolic.rows = 32
        cfg_novq.systolic.cols = 32
        cfg_novq.sram.total_size_kb = 512
        sim_novq = NPUSimulator(cfg_novq)
        result_novq = sim_novq.run_matmul(M=64, N=64, K=128)

        assert result_vq["vq"]["total_dequant_cycles"] > 0
        assert "vq" not in result_novq
