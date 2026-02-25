from __future__ import annotations

from pathlib import Path

import torch

from ..compute.special_functions import SpecialFunctionUnit
from ..compute.systolic_array import SystolicArray
from ..core.clock import SimulationEngine
from ..core.config import NPUConfig, load_config
from ..core.datatypes import AccumulatorType, DataType
from ..core.stats import LayerStats, SimStats
from ..dataflow.scheduler import TileScheduler
from ..dataflow.tiler import Tiler
from ..dataflow.ws_dataflow import WSDataflow
from ..memory.dram_interface import SimpleDRAMModel
from ..memory.double_buffer import DoubleBufferController
from ..memory.memory_controller import MemoryController
from ..memory.sram import BankedSRAM, PortType
from ..memory.sram_buffers import SRAMBuffer, create_buffer_partitions


class NPUSimulator:
    """Top-level NPU Simulator.

    Orchestrates all components:
    - SimulationEngine (event-driven clock)
    - SystolicArray (PyTorch compute + analytical cycles)
    - BankedSRAM (with banking conflict modeling)
    - DRAM (SimpleDRAMModel or DRAMSim3)
    - Tiler + WSDataflow (tile schedule generation)
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

        # DRAM
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
        self.dataflow = WSDataflow(self.tiler)

        # Double buffer controller
        self.double_buf = DoubleBufferController()

        # Stats
        self.stats = SimStats(
            array_rows=config.systolic.rows,
            array_cols=config.systolic.cols,
            clock_freq_mhz=config.systolic.clock_freq_mhz,
        )

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
    ) -> dict:
        """Simulate a single MatMul: C[M,N] = A[M,K] * B[K,N].

        Returns summary statistics dict.
        """
        # Generate tile schedule
        tile_config, schedule = self.dataflow.generate_schedule(
            M, N, K,
            weight_base_addr=weight_base,
            activation_base_addr=act_base,
            output_base_addr=output_base,
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

        return self.stats.summary()

    def run_attention(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int | None = None,
    ) -> dict:
        """Simulate multi-head attention.

        Operations:
        1. QKV projection: MatMul(seq_len, 3*hidden_dim, hidden_dim)
        2. Attention scores: H x MatMul(seq_len, seq_len, head_dim)
        3. Softmax
        4. Attention output: H x MatMul(seq_len, head_dim, seq_len)
        5. Output projection: MatMul(seq_len, hidden_dim, hidden_dim)
        """
        if head_dim is None:
            head_dim = hidden_dim // num_heads

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

        return self.stats.summary()

    def run_transformer_layer(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int | None = None,
        head_dim: int | None = None,
    ) -> dict:
        """Simulate a full transformer layer.

        Operations:
        1. LayerNorm
        2. Multi-head Attention
        3. Residual Add
        4. LayerNorm
        5. FFN (2x MatMul with GELU)
        6. Residual Add
        """
        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim
        if head_dim is None:
            head_dim = hidden_dim // num_heads

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
        return self.stats.summary()

    def _reset_for_layer(self) -> None:
        """Reset SRAM bank tracking between layers (new data layout)."""
        self.sram.reset_busy()
        self.double_buf.reset()
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
