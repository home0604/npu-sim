"""Tests for GPTVQ integration: dequant unit, OS systolic array, GPTVQ tiler,
OS dataflow, GPTVQ scheduler, and end-to-end GPTVQ matmul."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

from npu_sim.core.config import (
    DataTypeConfig,
    DoubleBufferConfig,
    GPTVQConfig,
    NPUConfig,
    SRAMConfig,
    SystolicArrayConfig,
)
from npu_sim.core.datatypes import AccumulatorType, DataType
from npu_sim.compute.dequant_unit import DequantizationUnit
from npu_sim.compute.systolic_array import OSSystolicArray
from npu_sim.dataflow.gptvq_tiler import GPTVQTiler
from npu_sim.dataflow.os_dataflow import OSDataflow


class TestDequantUnit:
    """Test the dequantization unit."""

    def test_dequant_cycles_basic(self):
        cfg = GPTVQConfig(codebook_size=16, vector_dim=2, index_bits=4,
                          dequant_pipeline_stages=3, dequant_throughput=1)
        unit = DequantizationUnit(cfg)
        # tile_m=4, tile_k=8 → 4 groups per row → 4*4=16 vectors
        cycles = unit.dequant_cycles(4, 8)
        assert cycles == 3 + 16  # pipeline + num_vectors

    def test_dequant_cycles_throughput(self):
        cfg = GPTVQConfig(codebook_size=16, vector_dim=2, index_bits=4,
                          dequant_pipeline_stages=3, dequant_throughput=4)
        unit = DequantizationUnit(cfg)
        cycles = unit.dequant_cycles(4, 8)
        # 16 vectors / 4 throughput = 4 cycles + 3 pipeline
        assert cycles == 3 + 4

    def test_dequant_functional_basic(self):
        cfg = GPTVQConfig(codebook_size=4, vector_dim=2, index_bits=2, use_scaling=False)
        unit = DequantizationUnit(cfg)

        codebook = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        indices = torch.tensor([[0, 2], [1, 3]])  # [2, 2] → tile_m=2, K_groups=2

        result = unit.dequantize(indices, codebook, tile_m=2, tile_k=4)
        # Row 0: codebook[0]=[1,2], codebook[2]=[5,6] → [1,2,5,6]
        # Row 1: codebook[1]=[3,4], codebook[3]=[7,8] → [3,4,7,8]
        expected = torch.tensor([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]])
        assert torch.equal(result.output, expected)
        assert result.num_vectors == 4

    def test_dequant_functional_with_scaling(self):
        cfg = GPTVQConfig(codebook_size=4, vector_dim=2, index_bits=2, use_scaling=True)
        unit = DequantizationUnit(cfg)

        codebook = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        indices = torch.tensor([[0, 1]])  # [1, 2]
        scales = torch.tensor([2.0, 3.0])  # per-group scaling
        zero_points = torch.tensor([0.5, 1.0])

        result = unit.dequantize(indices, codebook, tile_m=1, tile_k=4,
                                 scales=scales, zero_points=zero_points)
        # Group 0: 2.0 * [1,1] + 0.5 = [2.5, 2.5]
        # Group 1: 3.0 * [2,2] + 1.0 = [7.0, 7.0]
        expected = torch.tensor([[2.5, 2.5, 7.0, 7.0]])
        assert torch.allclose(result.output, expected)


class TestOSSystolicArray:
    """Test the Output-Stationary systolic array."""

    def test_os_analytical_cycles(self):
        sa = OSSystolicArray(rows=8, cols=8, dtype=DataType.FP16, acc_dtype=AccumulatorType.FP32)
        cycles = sa.analytical_cycles(8, 8, 16)
        # fill=15, compute=16, drain=15 → total=46
        assert cycles.pipeline_fill_cycles == 15
        assert cycles.compute_cycles == 16
        assert cycles.pipeline_drain_cycles == 15
        assert cycles.total_cycles == 46
        assert cycles.weight_load_cycles == 0  # OS has no separate weight load

    def test_os_compute_tile(self):
        sa = OSSystolicArray(rows=8, cols=8, dtype=DataType.FP16, acc_dtype=AccumulatorType.FP32)
        W = torch.randn(4, 8)
        A = torch.randn(8, 4)
        result = sa.compute_tile(W, A)
        expected = torch.matmul(W.float(), A.float())
        assert torch.allclose(result.output.float(), expected, atol=1e-5)

    def test_os_multi_pass(self):
        sa = OSSystolicArray(rows=4, cols=4, dtype=DataType.FP16, acc_dtype=AccumulatorType.FP32)
        # tile 8x8 on 4x4 array → 4 passes
        cycles = sa.analytical_cycles(8, 8, 16)
        # 4 passes: each fill=7, compute=16, drain=7 → 30 per pass → 120 total
        assert cycles.total_cycles == 4 * (7 + 16 + 7)


class TestGPTVQTiler:
    """Test the GPTVQ-aware tiler."""

    def test_tile_fits_in_sram(self):
        sram_cfg = SRAMConfig(total_size_kb=512, codebook_buffer_fraction=0.05,
                              index_buffer_fraction=0.20, scale_buffer_fraction=0.05)
        array_cfg = SystolicArrayConfig(rows=32, cols=32)
        dtype_cfg = DataTypeConfig(compute_dtype="FP16", accumulator_dtype="FP32")
        gptvq_cfg = GPTVQConfig(codebook_size=16, vector_dim=2, index_bits=4, use_scaling=False)

        tiler = GPTVQTiler(sram_cfg, array_cfg, dtype_cfg, gptvq_cfg, double_buffer=True)
        tc = tiler.compute_tiles(32, 32, 32)

        assert tc.tile_m == 32
        assert tc.tile_n == 32
        assert tc.tile_k >= 2  # at least one vector group
        assert tc.tile_k % 2 == 0  # multiple of vector_dim
        assert tc.num_m_tiles >= 1
        assert tc.num_n_tiles >= 1

    def test_tile_k_multiple_of_vector_dim(self):
        sram_cfg = SRAMConfig(total_size_kb=4)  # very small SRAM
        array_cfg = SystolicArrayConfig(rows=8, cols=8)
        dtype_cfg = DataTypeConfig(compute_dtype="FP16", accumulator_dtype="FP32")
        gptvq_cfg = GPTVQConfig(codebook_size=16, vector_dim=4, index_bits=4, use_scaling=False)

        tiler = GPTVQTiler(sram_cfg, array_cfg, dtype_cfg, gptvq_cfg, double_buffer=False)
        tc = tiler.compute_tiles(64, 64, 128)

        assert tc.tile_k % 4 == 0  # multiple of vector_dim=4

    def test_codebook_tile_bytes(self):
        gptvq_cfg = GPTVQConfig(codebook_size=16, vector_dim=2, codebook_dtype="FP16")
        sram_cfg = SRAMConfig(total_size_kb=512)
        array_cfg = SystolicArrayConfig(rows=32, cols=32)
        dtype_cfg = DataTypeConfig(compute_dtype="FP16", accumulator_dtype="FP32")

        tiler = GPTVQTiler(sram_cfg, array_cfg, dtype_cfg, gptvq_cfg, double_buffer=False)
        tc = tiler.compute_tiles(32, 32, 32)

        # codebook: 16 entries * 2 dims * 2 bytes = 64 bytes
        assert tc.codebook_tile_bytes == 16 * 2 * 2

    def test_dram_traffic_estimate(self):
        sram_cfg = SRAMConfig(total_size_kb=512)
        array_cfg = SystolicArrayConfig(rows=32, cols=32)
        dtype_cfg = DataTypeConfig(compute_dtype="FP16", accumulator_dtype="FP32")
        gptvq_cfg = GPTVQConfig(codebook_size=16, vector_dim=2, index_bits=4, use_scaling=False)

        tiler = GPTVQTiler(sram_cfg, array_cfg, dtype_cfg, gptvq_cfg, double_buffer=True)
        traffic = tiler.estimate_dram_traffic(256, 256, 256)

        assert traffic["codebook_read_bytes"] > 0
        assert traffic["index_read_bytes"] > 0
        assert traffic["activation_read_bytes"] > 0
        assert traffic["output_write_bytes"] > 0
        assert traffic["total_bytes"] == traffic["total_read_bytes"] + traffic["total_write_bytes"]


class TestOSDataflow:
    """Test the Output-Stationary dataflow schedule generation."""

    def test_basic_schedule(self):
        sram_cfg = SRAMConfig(total_size_kb=512)
        array_cfg = SystolicArrayConfig(rows=32, cols=32)
        dtype_cfg = DataTypeConfig(compute_dtype="FP16", accumulator_dtype="FP32")
        gptvq_cfg = GPTVQConfig(codebook_size=16, vector_dim=2, index_bits=4, use_scaling=False)

        tiler = GPTVQTiler(sram_cfg, array_cfg, dtype_cfg, gptvq_cfg, double_buffer=True)
        dataflow = OSDataflow(tiler, gptvq_cfg)
        tc, schedule = dataflow.generate_schedule(32, 32, 32)

        assert len(schedule) > 0
        # First tile should load codebook
        assert schedule[0].load_codebook is True
        assert schedule[0].load_indices is True
        assert schedule[0].load_activation is True

    def test_os_output_stored_on_last_k(self):
        sram_cfg = SRAMConfig(total_size_kb=8)  # small SRAM forces multiple k-tiles
        array_cfg = SystolicArrayConfig(rows=8, cols=8)
        dtype_cfg = DataTypeConfig(compute_dtype="FP16", accumulator_dtype="FP32")
        gptvq_cfg = GPTVQConfig(codebook_size=16, vector_dim=2, index_bits=4, use_scaling=False)

        tiler = GPTVQTiler(sram_cfg, array_cfg, dtype_cfg, gptvq_cfg, double_buffer=False)
        dataflow = OSDataflow(tiler, gptvq_cfg)
        tc, schedule = dataflow.generate_schedule(8, 8, 64)

        if tc.num_k_tiles > 1:
            # Only the last k-tile per (m,n) should store output
            for op in schedule:
                if op.k_idx < tc.num_k_tiles - 1:
                    assert op.store_output is False
                    if op.k_idx > 0:
                        assert op.accumulate is True

    def test_scaling_flags(self):
        sram_cfg = SRAMConfig(total_size_kb=512)
        array_cfg = SystolicArrayConfig(rows=32, cols=32)
        dtype_cfg = DataTypeConfig(compute_dtype="FP16", accumulator_dtype="FP32")
        gptvq_cfg = GPTVQConfig(codebook_size=16, vector_dim=2, index_bits=4, use_scaling=True)

        tiler = GPTVQTiler(sram_cfg, array_cfg, dtype_cfg, gptvq_cfg, double_buffer=True)
        dataflow = OSDataflow(tiler, gptvq_cfg)
        _, schedule = dataflow.generate_schedule(32, 32, 32)

        # All ops should have load_scales=True when scaling is enabled
        for op in schedule:
            assert op.load_scales is True

    def test_no_scaling_flags(self):
        sram_cfg = SRAMConfig(total_size_kb=512)
        array_cfg = SystolicArrayConfig(rows=32, cols=32)
        dtype_cfg = DataTypeConfig(compute_dtype="FP16", accumulator_dtype="FP32")
        gptvq_cfg = GPTVQConfig(codebook_size=16, vector_dim=2, index_bits=4, use_scaling=False)

        tiler = GPTVQTiler(sram_cfg, array_cfg, dtype_cfg, gptvq_cfg, double_buffer=True)
        dataflow = OSDataflow(tiler, gptvq_cfg)
        _, schedule = dataflow.generate_schedule(32, 32, 32)

        for op in schedule:
            assert op.load_scales is False


class TestGPTVQE2E:
    """End-to-end tests for GPTVQ simulation."""

    def _make_gptvq_config(self, use_scaling=False, sram_kb=512):
        return NPUConfig(
            name="test-gptvq",
            systolic=SystolicArrayConfig(rows=8, cols=8, clock_freq_mhz=1000),
            dtype=DataTypeConfig(compute_dtype="FP16", accumulator_dtype="FP32"),
            sram=SRAMConfig(
                total_size_kb=sram_kb,
                num_banks=16,
                bank_width_bytes=64,
                port_type="dual",
                codebook_buffer_fraction=0.05,
                index_buffer_fraction=0.20,
                scale_buffer_fraction=0.05,
            ),
            double_buffer=DoubleBufferConfig(enabled=True),
            gptvq=GPTVQConfig(
                enabled=True,
                codebook_size=16,
                vector_dim=2,
                index_bits=4,
                use_scaling=use_scaling,
                codebook_dtype="FP16",
                scale_dtype="FP16",
                zero_point_dtype="FP16",
                dequant_pipeline_stages=3,
                dequant_throughput=1,
            ),
        )

    def test_gptvq_matmul_basic(self):
        from npu_sim.sim.simulator import NPUSimulator
        config = self._make_gptvq_config()
        sim = NPUSimulator(config)
        result = sim.run_matmul(16, 16, 16)
        assert result["total_cycles"] > 0
        assert result["compute"]["tiles_processed"] > 0

    def test_gptvq_matmul_larger(self):
        from npu_sim.sim.simulator import NPUSimulator
        config = self._make_gptvq_config()
        sim = NPUSimulator(config)
        result = sim.run_matmul(64, 64, 64)
        assert result["total_cycles"] > 0
        assert "gptvq" in result
        assert result["gptvq"]["total_dequant_cycles"] > 0
        assert result["gptvq"]["index_load_bytes"] > 0

    def test_gptvq_matmul_correctness(self):
        from npu_sim.sim.simulator import NPUSimulator
        config = self._make_gptvq_config()
        sim = NPUSimulator(config)
        result = sim.run_matmul(16, 16, 16, check_correctness=True)
        assert result["correctness"] == "pass"

    def test_gptvq_matmul_correctness_larger(self):
        from npu_sim.sim.simulator import NPUSimulator
        config = self._make_gptvq_config()
        sim = NPUSimulator(config)
        result = sim.run_matmul(32, 32, 64, check_correctness=True)
        assert result["correctness"] == "pass"

    def test_gptvq_with_scaling(self):
        from npu_sim.sim.simulator import NPUSimulator
        config = self._make_gptvq_config(use_scaling=True)
        sim = NPUSimulator(config)
        result = sim.run_matmul(16, 16, 16, check_correctness=True)
        assert result["correctness"] == "pass"
        assert "gptvq" in result

    def test_gptvq_compression_ratio(self):
        from npu_sim.sim.simulator import NPUSimulator
        config = self._make_gptvq_config()
        sim = NPUSimulator(config)
        result = sim.run_matmul(64, 64, 64)
        # With 4-bit indices and vector_dim=2 for FP16 weights,
        # compression should be > 1
        assert result["gptvq"]["compression_ratio"] > 1.0

    def test_gptvq_stats_in_summary(self):
        from npu_sim.sim.simulator import NPUSimulator
        config = self._make_gptvq_config()
        sim = NPUSimulator(config)
        result = sim.run_matmul(32, 32, 32)
        assert "gptvq" in result
        assert "total_dequant_cycles" in result["gptvq"]
        assert "total_vectors_dequantized" in result["gptvq"]
        assert "codebook_load_bytes" in result["gptvq"]
        assert "index_load_bytes" in result["gptvq"]
        assert "compression_ratio" in result["gptvq"]

    def test_gptvq_double_buffer_overlap(self):
        """GPTVQ with double buffer should have fewer total cycles than without."""
        from npu_sim.sim.simulator import NPUSimulator
        config_db = self._make_gptvq_config()
        config_no_db = self._make_gptvq_config()
        config_no_db.double_buffer.enabled = False

        sim_db = NPUSimulator(config_db)
        sim_no_db = NPUSimulator(config_no_db)

        result_db = sim_db.run_matmul(64, 64, 64)
        result_no_db = sim_no_db.run_matmul(64, 64, 64)

        # Double buffer should help (or at least not be worse)
        assert result_db["total_cycles"] <= result_no_db["total_cycles"]

    def test_gptvq_from_yaml_config(self):
        from npu_sim.core.config import load_config
        from npu_sim.sim.simulator import NPUSimulator

        config_path = Path(__file__).parent.parent / "configs" / "gptvq.yaml"
        if not config_path.exists():
            pytest.skip("gptvq.yaml not found")

        config = load_config(config_path)
        assert config.gptvq.enabled is True
        sim = NPUSimulator(config)
        result = sim.run_matmul(32, 32, 32)
        assert result["total_cycles"] > 0
