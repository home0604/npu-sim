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

    def compute_tiles(self, M: int, N: int, K: int, dataflow: str = "OS") -> TileConfig:
        """Compute tile sizes for MatMul C[M,N] = A[M,K] * B[K,N].

        Dataflow determines which dimensions map to array rows/cols:
          OS: tile_m <= rows, tile_n <= cols, maximize tile_k
          WS: tile_k <= rows, tile_n <= cols, maximize tile_m
          IS: tile_k <= rows, tile_m <= cols, maximize tile_n

        Double buffering halves the per-slot budget for weight+activation.
        """
        db_factor = 2 if self.double_buffer else 1

        weight_budget = self.weight_buf_size // db_factor
        act_budget = self.act_buf_size // db_factor
        out_budget = self.out_buf_size

        if dataflow == "WS":
            tile_k = min(K, self.array_rows)
            tile_n = min(N, self.array_cols)
            max_mn_output = out_budget // self.acc_bytes if self.acc_bytes > 0 else 1
            # tile_m is the "free" dimension; maximize within SRAM
            max_m_weight = weight_budget // (tile_k * self.bytes_per_elem) if tile_k > 0 else M
            max_m_act = act_budget // (tile_n * self.bytes_per_elem) if tile_n > 0 else M
            max_m_out = max_mn_output // tile_n if tile_n > 0 else M
            tile_m = min(M, max_m_weight, max_m_act, max_m_out)
            tile_m = max(1, tile_m)
        elif dataflow == "IS":
            tile_k = min(K, self.array_rows)
            tile_m = min(M, self.array_cols)
            max_mn_output = out_budget // self.acc_bytes if self.acc_bytes > 0 else 1
            # tile_n is the "free" dimension; maximize within SRAM
            max_n_weight = weight_budget // (tile_k * self.bytes_per_elem) if tile_k > 0 else N
            max_n_act = act_budget // (tile_m * self.bytes_per_elem) if tile_m > 0 else N
            max_n_out = max_mn_output // tile_m if tile_m > 0 else N
            tile_n = min(N, max_n_weight, max_n_act, max_n_out)
            tile_n = max(1, tile_n)
        else:  # OS
            tile_m = min(M, self.array_rows)
            tile_n = min(N, self.array_cols)
            max_mn_output = out_budget // self.acc_bytes if self.acc_bytes > 0 else tile_m * tile_n
            if tile_m * tile_n > max_mn_output:
                tile_n = max(1, max_mn_output // tile_m) if tile_m > 0 else 1
            max_k_weight = weight_budget // (tile_m * self.bytes_per_elem) if tile_m > 0 else K
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

    def estimate_dram_traffic(self, M: int, N: int, K: int, dataflow: str = "OS") -> dict[str, int]:
        """Estimate total DRAM traffic in bytes for a MatMul."""
        tc = self.compute_tiles(M, N, K, dataflow=dataflow)
        total_tiles = tc.num_m_tiles * tc.num_n_tiles * tc.num_k_tiles

        if dataflow == "WS":
            # Weight loaded once per (k,n) pair (first m); activation every tile
            weight_loads = tc.num_k_tiles * tc.num_n_tiles
            act_loads = total_tiles
            output_writes = total_tiles
        elif dataflow == "IS":
            # Activation loaded once per (m,k) pair (first n); weight every tile
            weight_loads = total_tiles
            act_loads = tc.num_m_tiles * tc.num_k_tiles
            output_writes = total_tiles
        else:  # OS
            # Weight & activation loaded every tile; output once per (m,n)
            weight_loads = total_tiles
            act_loads = total_tiles
            output_writes = tc.num_m_tiles * tc.num_n_tiles

        weight_bytes = weight_loads * tc.weight_tile_bytes
        act_bytes = act_loads * tc.activation_tile_bytes
        output_bytes = output_writes * tc.output_tile_bytes

        return {
            "weight_read_bytes": weight_bytes,
            "activation_read_bytes": act_bytes,
            "output_write_bytes": output_bytes,
            "total_read_bytes": weight_bytes + act_bytes,
            "total_write_bytes": output_bytes,
            "total_bytes": weight_bytes + act_bytes + output_bytes,
        }
