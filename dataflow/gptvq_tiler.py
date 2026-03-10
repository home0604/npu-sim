"""GPTVQ-aware tiler for computing tile sizes with compressed weight data.

Instead of raw weight tiles, accounts for codebook, index, and optional scale
data sizes when determining how large tiles can be within SRAM budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.config import DataTypeConfig, GPTVQConfig, SRAMConfig, SystolicArrayConfig


@dataclass
class GPTVQTileConfig:
    """Computed tile sizes and counts for a GPTVQ MatMul operation."""

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


class GPTVQTiler:
    """Computes optimal tiling for GPTVQ-compressed matmul operations.

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
        gptvq_config: GPTVQConfig,
        double_buffer: bool = True,
    ):
        self.sram_size = sram_config.total_size_kb * 1024
        self.array_rows = array_config.rows
        self.array_cols = array_config.cols
        self.act_bytes_per_elem = dtype_config.bytes_per_element
        self.acc_bytes = dtype_config.accumulator_bytes
        self.double_buffer = double_buffer

        self.gptvq = gptvq_config
        self.vector_dim = gptvq_config.vector_dim
        self.codebook_size = gptvq_config.codebook_size
        self.index_elem_bytes = gptvq_config.index_elem_bytes
        self.codebook_entry_bytes = gptvq_config.codebook_entry_bytes
        self.use_scaling = gptvq_config.use_scaling
        self.scale_bytes = gptvq_config.scale_bytes if gptvq_config.use_scaling else 0
        self.zp_bytes = gptvq_config.zero_point_bytes if gptvq_config.use_scaling else 0

        # Buffer sizes from SRAM fractions
        if gptvq_config.use_scaling:
            self.codebook_buf_size = int(self.sram_size * sram_config.codebook_buffer_fraction)
            self.index_buf_size = int(self.sram_size * sram_config.index_buffer_fraction)
            self.scale_buf_size = int(self.sram_size * sram_config.scale_buffer_fraction)
            self.act_buf_size = int(
                self.sram_size * sram_config.activation_buffer_fraction
            )
        else:
            # When scaling disabled, redistribute scale fraction
            extra = sram_config.scale_buffer_fraction / 2
            self.codebook_buf_size = int(self.sram_size * sram_config.codebook_buffer_fraction)
            self.index_buf_size = int(
                self.sram_size * (sram_config.index_buffer_fraction + extra)
            )
            self.scale_buf_size = 0
            self.act_buf_size = int(
                self.sram_size * (sram_config.activation_buffer_fraction + extra)
            )

        self.out_buf_size = (
            self.sram_size
            - self.codebook_buf_size
            - self.index_buf_size
            - self.scale_buf_size
            - self.act_buf_size
        )

    def compute_tiles(self, M: int, N: int, K: int) -> GPTVQTileConfig:
        """Compute tile sizes for GPTVQ MatMul C[M,N] = W_deq[M,K] * A[K,N].

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

        # Max tile_k from index buffer: tile_m * (tile_k/d) * index_bytes <= budget
        if tile_m > 0 and self.index_elem_bytes > 0:
            max_groups_idx = index_budget // (tile_m * self.index_elem_bytes)
            max_k_index = max_groups_idx * d
        else:
            max_k_index = K

        # Max tile_k from activation buffer: tile_k * tile_n * act_bytes <= budget
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

        tile_k = min(K, max_k_index, max_k_act, max_k_scale)
        # Round down to multiple of vector_dim
        tile_k = max(d, (tile_k // d) * d)
        # But cap at K
        tile_k = min(tile_k, K)
        # If K is not a multiple of d, the last tile handles the remainder
        # For tiling purposes, use the rounded value
        if tile_k < d:
            tile_k = min(d, K)

        num_m = (M + tile_m - 1) // tile_m
        num_n = (N + tile_n - 1) // tile_n
        num_k = (K + tile_k - 1) // tile_k

        num_groups = math.ceil(tile_k / d)

        # Codebook: same for all tiles (R * d * entry_bytes per codebook)
        codebook_bytes = self.codebook_size * d * self.codebook_entry_bytes

        return GPTVQTileConfig(
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
        """Estimate total DRAM traffic in bytes for a GPTVQ MatMul."""
        tc = self.compute_tiles(M, N, K)

        # In OS dataflow:
        # Codebook: loaded once per (m,n) pair (or once if shared)
        codebook_loads = tc.num_m_tiles * tc.num_n_tiles
        codebook_bytes = codebook_loads * tc.codebook_tile_bytes

        # Indices: loaded every tile
        total_tiles = tc.num_m_tiles * tc.num_n_tiles * tc.num_k_tiles
        index_bytes = total_tiles * tc.index_tile_bytes

        # Scales: loaded every tile (if enabled)
        scale_bytes = total_tiles * tc.scale_tile_bytes
        zp_bytes = total_tiles * tc.zero_point_tile_bytes

        # Activation: loaded every tile
        act_bytes = total_tiles * tc.activation_tile_bytes

        # Output: written once per (m, n) pair
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
