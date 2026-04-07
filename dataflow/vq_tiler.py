"""VQ-aware tiler for computing tile sizes with compressed weight data.

Instead of raw weight tiles, accounts for codebook, index, and optional scale
data sizes when determining how large tiles can be within SRAM budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.config import DataTypeConfig, VQConfig, SRAMConfig, SystolicArrayConfig


@dataclass
class VQTileConfig:
    """Computed tile sizes and counts for a VQ MatMul operation."""

    tile_m: int
    tile_n: int
    tile_k: int  # always a multiple of vector_dim

    num_m_tiles: int
    num_n_tiles: int
    num_k_tiles: int

    # Byte sizes per tile
    codebook_tile_bytes: int  # R * d * codebook_entry_bytes (per codebook)
    index_tile_bytes: int  # tile_m * (tile_k/d) * index_elem_bytes
    scale_tile_bytes: int  # (tile_k/d) * scale_bytes (0 if scaling disabled)
    zero_point_tile_bytes: int  # (tile_k/d) * zp_bytes (0 if scaling disabled)
    activation_tile_bytes: int  # tile_k * tile_n * activation_bytes
    output_tile_bytes: int  # tile_m * tile_n * accumulator_bytes

    @property
    def total_tiles(self) -> int:
        return self.num_m_tiles * self.num_n_tiles * self.num_k_tiles


class VQTiler:
    """Computes optimal tiling for VQ-compressed matmul operations.

    Tile constraints differ from standard tiler:
    - Codebook buffer holds R * d * entry_bytes (usually small, fits easily)
    - Index buffer holds tile_m * (tile_k / d) * index_elem_bytes
    - Scale buffer (optional) holds (tile_k / d) * scale_bytes
    - Activation buffer holds tile_k * tile_n * activation_bytes
    - Output buffer holds tile_m * tile_n * accumulator_bytes
    - tile_k must be a multiple of vector_dim
    """

    def __init__(
        self,
        sram_config: SRAMConfig,
        array_config: SystolicArrayConfig,
        dtype_config: DataTypeConfig,
        vq_config: VQConfig,
        double_buffer: bool = True,
    ):
        self.sram_size = sram_config.total_size_kb * 1024
        self.array_rows = array_config.rows
        self.array_cols = array_config.cols
        self.act_bytes_per_elem = dtype_config.bytes_per_element
        self.acc_bytes = dtype_config.accumulator_bytes
        self.double_buffer = double_buffer

        self.vq = vq_config
        self.vector_dim = vq_config.vector_dim
        self.codebook_size = vq_config.codebook_size
        self.index_elem_bytes = vq_config.index_elem_bytes
        self.codebook_entry_bytes = vq_config.codebook_entry_bytes
        self.use_scaling = vq_config.use_scaling
        self.scale_bytes = vq_config.scale_bytes if vq_config.use_scaling else 0
        self.zp_bytes = vq_config.zero_point_bytes if vq_config.use_scaling else 0

        # Buffer sizes from SRAM fractions
        if vq_config.use_scaling:
            self.codebook_buf_size = int(self.sram_size * sram_config.codebook_buffer_fraction)
            self.index_buf_size = int(self.sram_size * sram_config.index_buffer_fraction)
            self.scale_buf_size = int(self.sram_size * sram_config.scale_buffer_fraction)
            self.dequant_weight_buf_size = int(self.sram_size * sram_config.dequant_weight_buffer_fraction)
            self.act_buf_size = int(self.sram_size * sram_config.activation_buffer_fraction)
        else:
            extra = sram_config.scale_buffer_fraction / 2
            self.codebook_buf_size = int(self.sram_size * sram_config.codebook_buffer_fraction)
            self.index_buf_size = int(
                self.sram_size * (sram_config.index_buffer_fraction + extra)
            )
            self.scale_buf_size = 0
            self.dequant_weight_buf_size = int(self.sram_size * sram_config.dequant_weight_buffer_fraction)
            self.act_buf_size = int(
                self.sram_size * (sram_config.activation_buffer_fraction + extra)
            )

        # Fused mode: dequant_weight buffer eliminated, redistribute to act/output
        if vq_config.dequant_mode == "fused":
            freed = self.dequant_weight_buf_size
            self.dequant_weight_buf_size = 0
            self.act_buf_size += freed // 2
            # remaining goes to output via the subtraction below

        self.out_buf_size = (
            self.sram_size
            - self.codebook_buf_size
            - self.index_buf_size
            - self.scale_buf_size
            - self.dequant_weight_buf_size
            - self.act_buf_size
        )

    def compute_tiles(self, M: int, N: int, K: int) -> VQTileConfig:
        """Compute tile sizes for VQ MatMul C[M,N] = W_deq[M,K] * A[K,N].

        Strategy:
        1. tile_m = min(M, array_rows), tile_n = min(N, array_cols)
        2. Maximize tile_k (multiple of vector_dim) within SRAM budget
        3. Double buffering halves per-slot budget for index/scale/activation
        """
        d = self.vector_dim
        db_factor = 2 if self.double_buffer else 1

        tile_m = min(M, self.array_rows)
        tile_n = min(N, self.array_cols)

        # Per-slot budget
        index_budget = self.index_buf_size // db_factor
        act_budget = self.act_buf_size // db_factor
        out_budget = self.out_buf_size

        # Check output buffer
        max_mn_output = out_budget // self.acc_bytes if self.acc_bytes > 0 else tile_m * tile_n
        if tile_m * tile_n > max_mn_output:
            tile_n = max(1, max_mn_output // tile_m) if tile_m > 0 else 1

        # Max tile_k from index buffer
        if tile_m > 0 and self.index_elem_bytes > 0:
            max_groups_idx = index_budget // (tile_m * self.index_elem_bytes)
            max_k_index = max_groups_idx * d
        else:
            max_k_index = K

        # Max tile_k from activation buffer
        if tile_n > 0 and self.act_bytes_per_elem > 0:
            max_k_act = act_budget // (tile_n * self.act_bytes_per_elem)
        else:
            max_k_act = K

        # Max tile_k from scale buffer (if scaling enabled)
        if self.use_scaling and self.scale_buf_size > 0:
            scale_budget = self.scale_buf_size // db_factor
            bytes_per_group = self.scale_bytes + self.zp_bytes
            if bytes_per_group > 0:
                max_groups_scale = scale_budget // bytes_per_group
                max_k_scale = max_groups_scale * d
            else:
                max_k_scale = K
        else:
            max_k_scale = K

        # Max tile_k from dequant_weight buffer (tile_m × tile_k × entry_bytes)
        if self.dequant_weight_buf_size > 0:
            dq_budget = self.dequant_weight_buf_size // db_factor
            max_k_dequant = int(dq_budget // (tile_m * self.codebook_entry_bytes)) if tile_m > 0 and self.codebook_entry_bytes > 0 else K
        else:
            max_k_dequant = K  # fused: no dequant_weight buffer needed

        tile_k = min(K, max_k_index, max_k_act, max_k_scale, max_k_dequant)
        tile_k = max(d, (tile_k // d) * d)
        tile_k = min(tile_k, K)
        if tile_k < d:
            tile_k = min(d, K)

        num_m = (M + tile_m - 1) // tile_m
        num_n = (N + tile_n - 1) // tile_n
        num_k = (K + tile_k - 1) // tile_k

        num_groups = math.ceil(tile_k / d)

        codebook_bytes = int(self.codebook_size * d * self.codebook_entry_bytes)

        return VQTileConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            num_m_tiles=num_m,
            num_n_tiles=num_n,
            num_k_tiles=num_k,
            codebook_tile_bytes=codebook_bytes,
            index_tile_bytes=tile_m * num_groups * self.index_elem_bytes,
            scale_tile_bytes=num_groups * self.scale_bytes if self.use_scaling else 0,
            zero_point_tile_bytes=num_groups * self.zp_bytes if self.use_scaling else 0,
            activation_tile_bytes=tile_k * tile_n * self.act_bytes_per_elem,
            output_tile_bytes=tile_m * tile_n * self.acc_bytes,
        )

    def estimate_dram_traffic(self, M: int, N: int, K: int) -> dict[str, int]:
        """Estimate total DRAM traffic in bytes for a VQ MatMul."""
        tc = self.compute_tiles(M, N, K)

        codebook_loads = tc.num_m_tiles * tc.num_n_tiles
        codebook_bytes = codebook_loads * tc.codebook_tile_bytes

        total_tiles = tc.num_m_tiles * tc.num_n_tiles * tc.num_k_tiles
        index_bytes = total_tiles * tc.index_tile_bytes

        scale_bytes = total_tiles * tc.scale_tile_bytes
        zp_bytes = total_tiles * tc.zero_point_tile_bytes

        act_bytes = total_tiles * tc.activation_tile_bytes

        output_writes = tc.num_m_tiles * tc.num_n_tiles
        output_bytes = output_writes * tc.output_tile_bytes

        total_read = codebook_bytes + index_bytes + scale_bytes + zp_bytes + act_bytes
        return {
            "codebook_read_bytes": codebook_bytes,
            "index_read_bytes": index_bytes,
            "scale_read_bytes": scale_bytes + zp_bytes,
            "activation_read_bytes": act_bytes,
            "output_write_bytes": output_bytes,
            "total_read_bytes": total_read,
            "total_write_bytes": output_bytes,
            "total_bytes": total_read + output_bytes,
        }
