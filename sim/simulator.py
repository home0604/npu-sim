from __future__ import annotations

from pathlib import Path

import torch

from compute.dequant_unit import DequantizationUnit
from compute.special_functions import SpecialFunctionUnit
from compute.systolic_array import SystolicArray
from core.clock import SimulationEngine
from core.config import DataTypeConfig, NPUConfig, load_config
from core.datatypes import AccumulatorType, DataType
from core.stats import LayerStats, SimStats
from dataflow.gptvq_dataflow import GPTVQDataflow
from dataflow.gptvq_scheduler import GPTVQScheduler
from dataflow.gptvq_tiler import GPTVQTiler
from dataflow.scheduler import TileScheduler
from dataflow.tiler import Tiler
from dataflow.stationary import StationaryDataflow
from memory.dram_interface import (
    DRAMSim3Adapter,
    DRAMSim3Interface,
    SimpleDRAMModel,
)
from memory.double_buffer import DoubleBufferController
from memory.memory_controller import MemoryController
from memory.sram import BankedSRAM, PortType
from memory.sram_buffers import SRAMBuffer, create_buffer_partitions, create_gptvq_buffer_partitions
from sim.functional_check import (
    compare_outputs,
    compute_diff_report,
    format_diff_report,
    reference_attention,
    reference_matmul,
    reference_transformer_layer,
    run_attention_functional,
    run_gptvq_schedule_functional,
    run_schedule_functional,
    run_transformer_layer_functional,
)


class NPUSimulator:
    """Top-level NPU Simulator.

    Orchestrates all components:
    - SimulationEngine (event-driven clock)
    - SystolicArray (PyTorch compute + analytical cycles)
    - BankedSRAM (with banking conflict modeling)
    - DRAM (SimpleDRAMModel or DRAMSim3)
    - Tiler + StationaryDataflow (tile schedule generation)
    - TileScheduler + DoubleBufferController (execution with overlap)
    """

    def __init__(self, config: NPUConfig | str | Path):
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config

        # Core engine
        self.engine = SimulationEngine()

        # Data types
        self.dtype = DataType[config.dtype.compute_dtype]
        self.acc_dtype = AccumulatorType[config.dtype.accumulator_dtype]

        # Systolic array
        self.systolic = SystolicArray(
            rows=config.systolic.rows,
            cols=config.systolic.cols,
            dtype=self.dtype,
            acc_dtype=self.acc_dtype,
        )

        # SRAM
        port_type = PortType.DUAL if config.sram.port_type.lower() == "dual" else PortType.SINGLE
        self.sram = BankedSRAM(
            size_bytes=config.sram.total_size_kb * 1024,
            num_banks=config.sram.num_banks,
            bank_width_bytes=config.sram.bank_width_bytes,
            port_type=port_type,
            read_latency=config.sram.read_latency_cycles,
            write_latency=config.sram.write_latency_cycles,
        )

        # Buffer partitions
        self.buffers = create_buffer_partitions(
            self.sram,
            weight_fraction=config.sram.weight_buffer_fraction,
            activation_fraction=config.sram.activation_buffer_fraction,
            output_fraction=config.sram.output_buffer_fraction,
            double_buffer=config.double_buffer.enabled,
        )

        # DRAM: SimpleDRAMModel (analytical) or DRAMSim3 (cycle-accurate) via adapter
        if getattr(config.dram, "use_dramsim3", False):
            try:
                _pkg_root = Path(__file__).resolve().parent.parent
                _config_file = config.dram.config_file
                if not Path(_config_file).is_absolute():
                    _config_file = str(_pkg_root / _config_file)
                _output_dir = config.dram.output_dir
                if not Path(_output_dir).is_absolute():
                    _output_dir = str(_pkg_root / _output_dir)
                Path(_output_dir).mkdir(parents=True, exist_ok=True)
                _dramsim = DRAMSim3Interface(
                    config_file=_config_file,
                    output_dir=_output_dir,
                    sim_engine=self.engine,
                    npu_freq_mhz=config.systolic.clock_freq_mhz,
                )
                self.dram = DRAMSim3Adapter(_dramsim, self.engine)
            except ImportError as e:
                import warnings

                warnings.warn(
                    f"DRAMSim3 requested but not available: {e}. Using SimpleDRAMModel."
                )
                self.dram = SimpleDRAMModel(
                    bandwidth_gbps=config.dram.bandwidth_gbps,
                    latency_ns=config.dram.latency_ns,
                    clock_freq_mhz=config.systolic.clock_freq_mhz,
                )
        else:
            self.dram = SimpleDRAMModel(
                bandwidth_gbps=config.dram.bandwidth_gbps,
                latency_ns=config.dram.latency_ns,
                clock_freq_mhz=config.systolic.clock_freq_mhz,
            )

        # Memory controller
        self.mem_ctrl = MemoryController(self.dram, self.sram, self.engine)

        # Tiler
        self.tiler = Tiler(
            config.sram,
            config.systolic,
            config.dtype,
            double_buffer=config.double_buffer.enabled,
        )

        # Dataflow
        self.dataflow = StationaryDataflow(self.tiler, dataflow=config.systolic.dataflow)

        # Double buffer controller
        self.double_buf = DoubleBufferController()

        # Stats
        self.stats = SimStats(
            array_rows=config.systolic.rows,
            array_cols=config.systolic.cols,
            clock_freq_mhz=config.systolic.clock_freq_mhz,
        )

        # GPTVQ components (initialized only when enabled)
        self.gptvq_enabled = getattr(config.gptvq, "enabled", False)
        if self.gptvq_enabled:
            self._init_gptvq()

    def _create_scheduler(
        self,
        weight_base_dram: int = 0,
        act_base_dram: int = 0,
        output_base_dram: int = 0,
    ) -> TileScheduler:
        return TileScheduler(
            engine=self.engine,
            systolic=self.systolic,
            mem_ctrl=self.mem_ctrl,
            double_buf=self.double_buf,
            weight_bufs=self.buffers["weight"],
            act_bufs=self.buffers["activation"],
            output_buf=self.buffers["output"][0],
            stats=self.stats,
            weight_base_dram=weight_base_dram,
            act_base_dram=act_base_dram,
            output_base_dram=output_base_dram,
            dataflow=self.config.systolic.dataflow,
        )

    def _init_gptvq(self) -> None:
        """Initialize GPTVQ-specific components."""
        cfg = self.config

        self.dequant_unit = DequantizationUnit(cfg.gptvq)

        self.gptvq_buffers = create_gptvq_buffer_partitions(
            self.sram,
            codebook_fraction=cfg.sram.codebook_buffer_fraction,
            index_fraction=cfg.sram.index_buffer_fraction,
            scale_fraction=cfg.sram.scale_buffer_fraction,
            activation_fraction=cfg.sram.activation_buffer_fraction,
            output_fraction=cfg.sram.output_buffer_fraction,
            double_buffer=cfg.double_buffer.enabled,
            use_scaling=cfg.gptvq.use_scaling,
        )

        self.gptvq_tiler = GPTVQTiler(
            cfg.sram,
            cfg.systolic,
            cfg.dtype,
            cfg.gptvq,
            double_buffer=cfg.double_buffer.enabled,
        )

        self.gptvq_dataflow = GPTVQDataflow(self.gptvq_tiler, cfg.gptvq)

    def _create_gptvq_scheduler(
        self,
        codebook_base_dram: int = 0,
        index_base_dram: int = 0,
        scale_base_dram: int = 0,
        act_base_dram: int = 0,
        output_base_dram: int = 0,
    ) -> GPTVQScheduler:
        return GPTVQScheduler(
            engine=self.engine,
            systolic=self.systolic,
            dequant_unit=self.dequant_unit,
            mem_ctrl=self.mem_ctrl,
            double_buf=self.double_buf,
            codebook_bufs=self.gptvq_buffers["codebook"],
            index_bufs=self.gptvq_buffers["index"],
            scale_bufs=self.gptvq_buffers["scale"],
            act_bufs=self.gptvq_buffers["activation"],
            output_buf=self.gptvq_buffers["output"][0],
            stats=self.stats,
            codebook_base_dram=codebook_base_dram,
            index_base_dram=index_base_dram,
            scale_base_dram=scale_base_dram,
            act_base_dram=act_base_dram,
            output_base_dram=output_base_dram,
        )

    def run_matmul(
        self,
        M: int,
        N: int,
        K: int,
        name: str = "MatMul",
        weight_base: int = 0,
        act_base: int = 0,
        output_base: int = 0,
        check_correctness: bool = False,
        weight: torch.Tensor | None = None,
        activation: torch.Tensor | None = None,
        seed: int | None = 42,
    ) -> dict:
        """Simulate a single MatMul: C[M,N] = A[M,K] * B[K,N].

        If check_correctness is True, runs the same tile schedule with real tensors
        and compares output to a reference (PyTorch matmul). Optionally pass
        weight (A) and activation (B); if None, random tensors are generated (seed).

        Returns summary statistics dict; when check_correctness is True, includes
        "correctness": "pass" | "fail" and "correctness_message".
        """
        if self.gptvq_enabled:
            return self._run_matmul_gptvq(
                M, N, K, name=name,
                act_base=act_base, output_base=output_base,
                check_correctness=check_correctness,
                activation=activation, seed=seed,
            )

        # Generate tile schedule
        tile_config, schedule = self.dataflow.generate_schedule(M, N, K)

        # Functionality check (before cycle sim): verify mul/accum per data type vs reference
        correctness_pass: bool | None = None
        correctness_message: str | None = None
        correctness_report: dict | None = None
        if check_correctness:
            if weight is None or activation is None:
                gen = torch.Generator()
                if seed is not None:
                    gen.manual_seed(seed)
                if self.dtype == DataType.INT8:
                    weight = torch.randint(
                        -128, 128, (M, K), dtype=self.dtype.torch_dtype, generator=gen
                    )
                    activation = torch.randint(
                        -128, 128, (K, N), dtype=self.dtype.torch_dtype, generator=gen
                    )
                else:
                    weight = torch.randn(M, K, generator=gen).to(self.dtype.torch_dtype)
                    activation = torch.randn(K, N, generator=gen).to(self.dtype.torch_dtype)
            C_ref = reference_matmul(weight, activation, self.acc_dtype)
            C_sim = run_schedule_functional(
                schedule, tile_config, weight, activation, self.systolic, M, N, K
            )
            # FP32: ref = one-shot matmul, sim = tile accumulate → small diff (tiling+acc error).
            # Use relaxed tol so we pass but report shows the actual NPU-side error.
            rtol, atol = 1e-5, 1e-5
            if self.acc_dtype == AccumulatorType.FP32:
                rtol, atol = 1e-3, 1e-2
            correctness_pass, correctness_message = compare_outputs(
                C_sim, C_ref, self.acc_dtype, rtol=rtol, atol=atol
            )
            correctness_report = compute_diff_report(
                C_sim, C_ref, self.acc_dtype, rtol=rtol, atol=atol, max_sample=10
            )
            if not correctness_pass:
                raise AssertionError(
                    f"Functionality check failed: {correctness_message}\n"
                    f"{format_diff_report(correctness_report)}"
                )

        # Update tile stats
        self.stats.tile.tile_m = tile_config.tile_m
        self.stats.tile.tile_n = tile_config.tile_n
        self.stats.tile.tile_k = tile_config.tile_k
        self.stats.tile.num_m_tiles = tile_config.num_m_tiles
        self.stats.tile.num_n_tiles = tile_config.num_n_tiles
        self.stats.tile.num_k_tiles = tile_config.num_k_tiles
        self.stats.tile.total_tiles = tile_config.total_tiles

        # Execute
        scheduler = self._create_scheduler(weight_base, act_base, output_base)
        end_cycle = scheduler.execute_schedule(schedule, self.engine.current_cycle)
        self.engine.current_cycle = end_cycle
        self.stats.total_cycles = end_cycle

        # Record layer stats
        self.stats.layer_stats.append(
            LayerStats(
                name=name,
                op_type="matmul",
                cycles=end_cycle,
                mac_ops=M * N * K,
                dram_bytes=self.stats.memory.dram_read_bytes + self.stats.memory.dram_write_bytes,
            )
        )

        summary = self.stats.summary()
        if check_correctness and correctness_pass is not None:
            summary["correctness"] = "pass" if correctness_pass else "fail"
            summary["correctness_message"] = correctness_message or ""
            if correctness_report is not None:
                summary["correctness_report"] = correctness_report
        return summary

    def _run_matmul_gptvq(
        self,
        M: int,
        N: int,
        K: int,
        name: str = "MatMul_GPTVQ",
        codebook_base: int = 0,
        index_base: int = 0,
        scale_base: int = 0,
        act_base: int = 0,
        output_base: int = 0,
        check_correctness: bool = False,
        codebook: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        activation: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        zero_points: torch.Tensor | None = None,
        seed: int | None = 42,
    ) -> dict:
        """Simulate a GPTVQ MatMul: C[M,N] = dequant(codebook, indices, scales)[M,K] * A[K,N].

        Uses Output-Stationary dataflow with dequantization.
        """
        import math

        gptvq_cfg = self.config.gptvq
        d = gptvq_cfg.vector_dim
        R = gptvq_cfg.codebook_size

        tile_config, schedule = self.gptvq_dataflow.generate_schedule(
            M, N, K,
            codebook_base_addr=codebook_base,
            index_base_addr=index_base,
            scale_base_addr=scale_base,
            activation_base_addr=act_base,
            output_base_addr=output_base,
        )

        correctness_pass: bool | None = None
        correctness_message: str | None = None
        correctness_report: dict | None = None
        if check_correctness:
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(seed)
            K_groups = math.ceil(K / d)
            if codebook is None:
                codebook = torch.randn(R, d, generator=gen).to(torch.float32)
            if indices is None:
                indices = torch.randint(0, R, (M, K_groups), generator=gen)
            if activation is None:
                activation = torch.randn(K, N, generator=gen).to(
                    self.dtype.torch_dtype if self.dtype != DataType.INT8 else torch.float32
                )
            if gptvq_cfg.use_scaling and scales is None:
                scales = torch.randn(K_groups, generator=gen).abs() + 0.1
            if gptvq_cfg.use_scaling and zero_points is None:
                zero_points = torch.randn(K_groups, generator=gen) * 0.01

            dequant_full = self.dequant_unit.dequantize(
                indices, codebook, M, K,
                scales=scales, zero_points=zero_points,
            )
            W_deq = dequant_full.output.float()
            C_ref = reference_matmul(
                W_deq.to(self.acc_dtype.torch_dtype),
                activation.float().to(self.acc_dtype.torch_dtype),
                self.acc_dtype,
            )

            C_sim = run_gptvq_schedule_functional(
                schedule, tile_config, codebook, indices, activation,
                self.dequant_unit, self.systolic, M, N, K,
                scales=scales, zero_points=zero_points,
            )

            rtol, atol = 1e-3, 1e-2
            correctness_pass, correctness_message = compare_outputs(
                C_sim, C_ref, self.acc_dtype, rtol=rtol, atol=atol
            )
            correctness_report = compute_diff_report(
                C_sim, C_ref, self.acc_dtype, rtol=rtol, atol=atol, max_sample=10
            )
            if not correctness_pass:
                raise AssertionError(
                    f"GPTVQ functionality check failed: {correctness_message}\n"
                    f"{format_diff_report(correctness_report)}"
                )

        self.stats.tile.tile_m = tile_config.tile_m
        self.stats.tile.tile_n = tile_config.tile_n
        self.stats.tile.tile_k = tile_config.tile_k
        self.stats.tile.num_m_tiles = tile_config.num_m_tiles
        self.stats.tile.num_n_tiles = tile_config.num_n_tiles
        self.stats.tile.num_k_tiles = tile_config.num_k_tiles
        self.stats.tile.total_tiles = tile_config.total_tiles

        scheduler = self._create_gptvq_scheduler(
            codebook_base, index_base, scale_base, act_base, output_base
        )
        end_cycle = scheduler.execute_schedule(schedule, self.engine.current_cycle)
        self.engine.current_cycle = end_cycle
        self.stats.total_cycles = end_cycle

        original_weight_bytes = M * K * self.dtype.num_bytes
        K_groups = math.ceil(K / d)
        compressed_bytes = (
            R * d * gptvq_cfg.codebook_entry_bytes
            + M * K_groups * gptvq_cfg.index_elem_bytes
        )
        if gptvq_cfg.use_scaling:
            compressed_bytes += K_groups * (gptvq_cfg.scale_bytes + gptvq_cfg.zero_point_bytes)
        if compressed_bytes > 0:
            self.stats.gptvq.compression_ratio = original_weight_bytes / compressed_bytes

        self.stats.layer_stats.append(
            LayerStats(
                name=name,
                op_type="matmul_gptvq",
                cycles=end_cycle,
                mac_ops=M * N * K,
                dram_bytes=self.stats.memory.dram_read_bytes + self.stats.memory.dram_write_bytes,
            )
        )

        summary = self.stats.summary()
        if check_correctness and correctness_pass is not None:
            summary["correctness"] = "pass" if correctness_pass else "fail"
            summary["correctness_message"] = correctness_message or ""
            if correctness_report is not None:
                summary["correctness_report"] = correctness_report
        return summary

    def run_attention(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int | None = None,
        check_correctness: bool = False,
        x: torch.Tensor | None = None,
        W_qkv: torch.Tensor | None = None,
        W_o: torch.Tensor | None = None,
        seed: int | None = 42,
    ) -> dict:
        """Simulate multi-head attention.

        Operations:
        1. QKV projection: MatMul(seq_len, 3*hidden_dim, hidden_dim)
        2. Attention scores: H x MatMul(seq_len, seq_len, head_dim)
        3. Softmax
        4. Attention output: H x MatMul(seq_len, head_dim, seq_len)
        5. Output projection: MatMul(seq_len, hidden_dim, hidden_dim)

        If check_correctness is True, runs reference and functional attention and
        compares outputs; optionally pass x, W_qkv, W_o or use random (seed).
        """
        if head_dim is None:
            head_dim = hidden_dim // num_heads

        correctness_pass: bool | None = None
        correctness_report: dict | None = None
        if check_correctness:
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(seed)
            if x is None:
                x = torch.randn(seq_len, hidden_dim, generator=gen)
            if W_qkv is None:
                W_qkv = torch.randn(hidden_dim, 3 * hidden_dim, generator=gen)
            if W_o is None:
                W_o = torch.randn(hidden_dim, hidden_dim, generator=gen)
            y_ref = reference_attention(x, W_qkv, W_o, num_heads, head_dim)
            df_fp32, tiler_fp32, systolic_fp32 = self._get_fp32_functional_components()
            y_sim = run_attention_functional(
                x, W_qkv, W_o, num_heads, head_dim,
                df_fp32, tiler_fp32, systolic_fp32,
            )
            # Ref = one-shot, sim = tile accumulate → report shows tiling+acc error.
            correctness_report = compute_diff_report(
                y_sim, y_ref, AccumulatorType.FP32, rtol=1e-2, atol=0.25, max_sample=10
            )
            correctness_pass = correctness_report["match"]
            if not correctness_pass:
                raise AssertionError(
                    "Attention functionality check failed.\n"
                    f"{format_diff_report(correctness_report)}"
                )

        dram_addr = 0
        bpe = self.dtype.num_bytes

        # 1. QKV projection
        self._reset_for_layer()
        qkv_weight_size = hidden_dim * 3 * hidden_dim * bpe
        self.run_matmul(
            seq_len, 3 * hidden_dim, hidden_dim,
            name="QKV_proj",
            weight_base=dram_addr,
            act_base=dram_addr + qkv_weight_size,
        )
        dram_addr += qkv_weight_size + seq_len * hidden_dim * bpe

        # 2. Attention scores per head
        for h in range(num_heads):
            self._reset_for_layer()
            self.run_matmul(
                seq_len, seq_len, head_dim,
                name=f"attn_score_h{h}",
                weight_base=dram_addr,
            )
            dram_addr += seq_len * head_dim * bpe

        # 3. Softmax (analytical)
        sf = SpecialFunctionUnit.softmax_cycles(seq_len, num_heads, self.dtype)
        self.engine.current_cycle += sf.cycles
        self.stats.total_cycles = self.engine.current_cycle
        self.stats.layer_stats.append(
            LayerStats(name="softmax", op_type="softmax", cycles=sf.cycles)
        )

        # 4. Attention output per head
        for h in range(num_heads):
            self._reset_for_layer()
            self.run_matmul(
                seq_len, head_dim, seq_len,
                name=f"attn_out_h{h}",
                weight_base=dram_addr,
            )
            dram_addr += seq_len * seq_len * bpe

        # 5. Output projection
        self._reset_for_layer()
        self.run_matmul(
            seq_len, hidden_dim, hidden_dim,
            name="output_proj",
            weight_base=dram_addr,
        )

        summary = self.stats.summary()
        if check_correctness and correctness_report is not None:
            summary["correctness"] = "pass" if correctness_pass else "fail"
            summary["correctness_message"] = (
                "Output matches reference (attention)." if correctness_pass else "Mismatch."
            )
            summary["correctness_report"] = correctness_report
        return summary

    def run_transformer_layer(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int | None = None,
        head_dim: int | None = None,
        check_correctness: bool = False,
        seed: int | None = 42,
    ) -> dict:
        """Simulate a full transformer layer.

        Operations:
        1. LayerNorm
        2. Multi-head Attention
        3. Residual Add
        4. LayerNorm
        5. FFN (2x MatMul with GELU)
        6. Residual Add

        If check_correctness is True, runs reference and functional transformer layer
        with random weights (seed) and compares outputs.

        Precision (cycle simulation, config default INT8):
        - Activations / weights in DRAM/SRAM: config.dtype (e.g. INT8).
        - All MatMuls (QKV, attn scores, attn out, output proj, ffn_up, ffn_down):
          input A/B in dtype (INT8), internal multiply in float, output cast to
          config.dtype.accumulator_dtype (INT32). Stored output tile = INT32.
        - LayerNorm: input/output buffer size = dtype; gamma/beta params = FP32 (4B each).
        - Softmax / GELU / Residual add: memory and cycles modeled with dtype for
          element size (no explicit type change in model).

        Precision (functional check only):
        - Entire layer in FP32: x, all weights, LN, attention, FFN matmuls, GELU, adds.
        - Uses FP32 systolic/tiler/dataflow so tile matmuls and ref both stay float.
        """
        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim
        if head_dim is None:
            head_dim = hidden_dim // num_heads

        correctness_pass: bool | None = None
        correctness_report: dict | None = None
        if check_correctness:
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(seed)
            x = torch.randn(seq_len, hidden_dim, generator=gen)
            W_ln1_g = torch.ones(hidden_dim)
            W_ln1_b = torch.zeros(hidden_dim)
            W_qkv = torch.randn(hidden_dim, 3 * hidden_dim, generator=gen)
            W_o = torch.randn(hidden_dim, hidden_dim, generator=gen)
            W_ln2_g = torch.ones(hidden_dim)
            W_ln2_b = torch.zeros(hidden_dim)
            W_ffn_up = torch.randn(hidden_dim, ffn_dim, generator=gen)
            W_ffn_down = torch.randn(ffn_dim, hidden_dim, generator=gen)
            y_ref = reference_transformer_layer(
                x, W_ln1_g, W_ln1_b, W_qkv, W_o, W_ln2_g, W_ln2_b,
                W_ffn_up, W_ffn_down, num_heads, head_dim, ffn_dim,
            )
            df_fp32, tiler_fp32, systolic_fp32 = self._get_fp32_functional_components()
            y_sim = run_transformer_layer_functional(
                x, W_ln1_g, W_ln1_b, W_qkv, W_o, W_ln2_g, W_ln2_b,
                W_ffn_up, W_ffn_down, num_heads, head_dim, ffn_dim,
                df_fp32, tiler_fp32, systolic_fp32,
            )
            # Ref = one-shot, sim = tile accumulate → report shows tiling+acc error.
            correctness_report = compute_diff_report(
                y_sim, y_ref, AccumulatorType.FP32, rtol=1e-2, atol=0.25, max_sample=10
            )
            correctness_pass = correctness_report["match"]
            if not correctness_pass:
                raise AssertionError(
                    "Transformer layer functionality check failed.\n"
                    f"{format_diff_report(correctness_report)}"
                )

        # 1. LayerNorm
        ln1 = SpecialFunctionUnit.layernorm_cycles(seq_len, hidden_dim, self.dtype)
        self.engine.current_cycle += ln1.cycles

        # 2. Attention
        self.run_attention(seq_len, hidden_dim, num_heads, head_dim)

        # 3. Residual add
        res1 = SpecialFunctionUnit.add_residual_cycles(seq_len, hidden_dim, self.dtype)
        self.engine.current_cycle += res1.cycles

        # 4. LayerNorm
        ln2 = SpecialFunctionUnit.layernorm_cycles(seq_len, hidden_dim, self.dtype)
        self.engine.current_cycle += ln2.cycles

        # 5. FFN: up projection
        self._reset_for_layer()
        self.run_matmul(seq_len, ffn_dim, hidden_dim, name="ffn_up")

        # GELU
        gelu = SpecialFunctionUnit.gelu_cycles(seq_len, ffn_dim, self.dtype)
        self.engine.current_cycle += gelu.cycles

        # FFN: down projection
        self._reset_for_layer()
        self.run_matmul(seq_len, hidden_dim, ffn_dim, name="ffn_down")

        # 6. Residual add
        res2 = SpecialFunctionUnit.add_residual_cycles(seq_len, hidden_dim, self.dtype)
        self.engine.current_cycle += res2.cycles

        self.stats.total_cycles = self.engine.current_cycle
        summary = self.stats.summary()
        if check_correctness and correctness_report is not None:
            summary["correctness"] = "pass" if correctness_pass else "fail"
            summary["correctness_message"] = (
                "Output matches reference (transformer layer)." if correctness_pass else "Mismatch."
            )
            summary["correctness_report"] = correctness_report
        return summary

    def _get_fp32_functional_components(
        self,
    ) -> tuple[StationaryDataflow, Tiler, SystolicArray]:
        """Build FP32 dataflow/tiler/systolic for correctness check (float compare)."""
        dtype_fp32 = DataTypeConfig(compute_dtype="FP32", accumulator_dtype="FP32")
        systolic_fp32 = SystolicArray(
            rows=self.config.systolic.rows,
            cols=self.config.systolic.cols,
            dtype=DataType.FP32,
            acc_dtype=AccumulatorType.FP32,
        )
        tiler_fp32 = Tiler(
            self.config.sram,
            self.config.systolic,
            dtype_fp32,
            double_buffer=self.config.double_buffer.enabled,
        )
        dataflow_fp32 = StationaryDataflow(tiler_fp32, dataflow=self.config.systolic.dataflow)
        return dataflow_fp32, tiler_fp32, systolic_fp32

    def _reset_for_layer(self) -> None:
        """Reset SRAM bank tracking between layers (new data layout)."""
        self.sram.reset_busy()
        self.double_buf.reset()
        if hasattr(self.dram, "_bus_free_cycle"):
            self.dram._bus_free_cycle = self.engine.current_cycle

    def reset(self) -> None:
        """Full reset of the simulator."""
        self.engine.reset()
        self.sram.reset_busy()
        self.sram.reset_stats()
        self.dram.reset()
        self.double_buf.reset()
        self.stats = SimStats(
            array_rows=self.config.systolic.rows,
            array_cols=self.config.systolic.cols,
            clock_freq_mhz=self.config.systolic.clock_freq_mhz,
        )
