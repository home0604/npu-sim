from __future__ import annotations

from dataclasses import dataclass

from ..core.config import DataTypeConfig, SRAMConfig, SystolicArrayConfig


@dataclass
class TileConfig:
    """Computed tile sizes and counts for a MatMul operation."""

    tile_m: int
    tile_n: int
    tile_k: int

    num_m_tiles: int
    num_n_tiles: int
    num_k_tiles: int

    weight_tile_bytes: int  # tile_m * tile_k * bytes_per_element
    activation_tile_bytes: int  # tile_k * tile_n * bytes_per_element
    output_tile_bytes: int  # tile_m * tile_n * accumulator_bytes

    @property
    def total_tiles(self) -> int:
        return self.num_m_tiles * self.num_n_tiles * self.num_k_tiles


class Tiler:
    """Computes optimal tiling for operations given SRAM constraints.

    For MatMul C[M,N] = A[M,K] * B[K,N]:
    - tile_m, tile_n are set to array dimensions (or smaller)
    - tile_k is maximized within SRAM budget to minimize DRAM traffic
    - With double buffering, weight+activation buffers need 2x space
    """

    def __init__(
        self,
        sram_config: SRAMConfig,
        array_config: SystolicArrayConfig,
        dtype_config: DataTypeConfig,
        double_buffer: bool = True,
    ):
        self.sram_size = sram_config.total_size_kb * 1024
        self.array_rows = array_config.rows
        self.array_cols = array_config.cols
        self.bytes_per_elem = dtype_config.bytes_per_element
        self.acc_bytes = dtype_config.accumulator_bytes
        self.double_buffer = double_buffer

        self.weight_buf_size = int(self.sram_size * sram_config.weight_buffer_fraction)
        self.act_buf_size = int(self.sram_size * sram_config.activation_buffer_fraction)
        self.out_buf_size = int(self.sram_size * sram_config.output_buffer_fraction)

    def compute_tiles(self, M: int, N: int, K: int) -> TileConfig:
        """Compute tile sizes for MatMul C[M,N] = A[M,K] * B[K,N].

        Strategy:
        1. tile_m = min(M, array_rows), tile_n = min(N, array_cols)
        2. Maximize tile_k within SRAM budget
        3. Double buffering halves the per-slot budget for weight+activation
        """
        db_factor = 2 if self.double_buffer else 1

        tile_m = min(M, self.array_rows)
        tile_n = min(N, self.array_cols)

        # Per-slot budget
        weight_budget = self.weight_buf_size // db_factor
        act_budget = self.act_buf_size // db_factor
        out_budget = self.out_buf_size

        # Check output buffer first
        max_mn_output = out_budget // self.acc_bytes if self.acc_bytes > 0 else tile_m * tile_n
        if tile_m * tile_n > max_mn_output:
            # Reduce tile_n to fit output buffer
            tile_n = max(1, max_mn_output // tile_m) if tile_m > 0 else 1

        # Max tile_k from weight buffer: tile_m * tile_k * bpe <= weight_budget
        max_k_weight = weight_budget // (tile_m * self.bytes_per_elem) if tile_m > 0 else K
        # Max tile_k from activation buffer: tile_k * tile_n * bpe <= act_budget
        max_k_act = act_budget // (tile_n * self.bytes_per_elem) if tile_n > 0 else K

        tile_k = min(K, max_k_weight, max_k_act)
        tile_k = max(1, tile_k)

        num_m = (M + tile_m - 1) // tile_m
        num_n = (N + tile_n - 1) // tile_n
        num_k = (K + tile_k - 1) // tile_k

        return TileConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            num_m_tiles=num_m,
            num_n_tiles=num_n,
            num_k_tiles=num_k,
            weight_tile_bytes=tile_m * tile_k * self.bytes_per_elem,
            activation_tile_bytes=tile_k * tile_n * self.bytes_per_elem,
            output_tile_bytes=tile_m * tile_n * self.acc_bytes,
        )

    def estimate_dram_traffic(self, M: int, N: int, K: int) -> dict[str, int]:
        """Estimate total DRAM traffic in bytes for a MatMul."""
        tc = self.compute_tiles(M, N, K)

        # In WS dataflow: weight loaded once per (m, n) pair, activation loaded every K-tile
        # Weight: loaded once per (m_tile, n_tile) but reused across k_tiles
        weight_loads = tc.num_m_tiles * tc.num_n_tiles  # one load per (m,n) tile pair
        weight_bytes = weight_loads * tc.weight_tile_bytes

        # Activation: loaded for every tile
        act_loads = tc.num_m_tiles * tc.num_n_tiles * tc.num_k_tiles
        act_bytes = act_loads * tc.activation_tile_bytes

        # Output: written once per (m, n) pair
        output_writes = tc.num_m_tiles * tc.num_n_tiles
        output_bytes = output_writes * tc.output_tile_bytes

        return {
            "weight_read_bytes": weight_bytes,
            "activation_read_bytes": act_bytes,
            "output_write_bytes": output_bytes,
            "total_read_bytes": weight_bytes + act_bytes,
            "total_write_bytes": output_bytes,
            "total_bytes": weight_bytes + act_bytes + output_bytes,
        }
