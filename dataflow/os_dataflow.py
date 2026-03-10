"""Output-Stationary dataflow for GPTVQ tile schedule generation.

Loop order (output stays in PEs):
    for m in range(num_m):
        for n in range(num_n):
            for k in range(num_k):
                load_codebook(k)     # small, loaded on first k-tile
                load_indices(m, k)   # compressed weight indices
                load_scales(k)       # optional per-group scaling
                dequantize()         # codebook[idx] → dequantized weight
                load_activation(k, n)
                compute()            # C[m,n] += W_deq * A
            store_output(m, n)

Key differences from WS:
- Output stays in PE across K-tiles (stored once at end)
- Both weight (via codebook+indices) and activation loaded every K-tile
- Includes dequantization step before compute
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.config import GPTVQConfig
from .gptvq_tiler import GPTVQTileConfig, GPTVQTiler


@dataclass
class GPTVQTileOp:
    """A single tile operation in the GPTVQ OS execution schedule."""

    m_idx: int
    n_idx: int
    k_idx: int
    tile_m: int  # effective tile dimensions (may be smaller at edges)
    tile_n: int
    tile_k: int

    load_codebook: bool  # True on first k-tile of each (m, n) pair
    load_indices: bool  # True always (compressed weight)
    load_scales: bool  # True if scaling enabled
    load_activation: bool  # True always
    store_output: bool  # True on last k-tile
    accumulate: bool  # True if k > 0 (accumulating partial sums)

    # DRAM address offsets (relative to base)
    codebook_dram_offset: int = 0
    index_dram_offset: int = 0
    scale_dram_offset: int = 0
    activation_dram_offset: int = 0
    output_dram_offset: int = 0


class OSDataflow:
    """Output-Stationary dataflow tile schedule generation for GPTVQ.

    Generates an ordered list of GPTVQTileOp that defines:
    - What data to load (codebook, indices, scales, activation)
    - When to dequantize and compute
    - When to store output (last K-tile of each output tile)
    """

    def __init__(self, tiler: GPTVQTiler, gptvq_config: GPTVQConfig):
        self.tiler = tiler
        self.gptvq = gptvq_config

    def generate_schedule(
        self,
        M: int,
        N: int,
        K: int,
        codebook_base_addr: int = 0,
        index_base_addr: int = 0,
        scale_base_addr: int = 0,
        activation_base_addr: int = 0,
        output_base_addr: int = 0,
    ) -> tuple[GPTVQTileConfig, list[GPTVQTileOp]]:
        """Generate ordered list of tile operations for a GPTVQ MatMul.

        Args:
            M, N, K: MatMul dimensions C[M,N] = W_deq[M,K] * A[K,N]
            codebook_base_addr: DRAM base for codebook
            index_base_addr: DRAM base for weight index matrix
            scale_base_addr: DRAM base for scaling factors
            activation_base_addr: DRAM base for activation matrix
            output_base_addr: DRAM base for output matrix

        Returns:
            (tile_config, schedule) tuple
        """
        tc = self.tiler.compute_tiles(M, N, K)
        d = self.gptvq.vector_dim
        idx_bytes = self.gptvq.index_elem_bytes
        act_bpe = self.tiler.act_bytes_per_elem
        acc_bytes = self.tiler.acc_bytes
        use_scaling = self.gptvq.use_scaling
        scale_bytes = self.gptvq.scale_bytes if use_scaling else 0
        zp_bytes = self.gptvq.zero_point_bytes if use_scaling else 0

        # Number of vector groups along K dimension (full matrix)
        K_groups = math.ceil(K / d)

        schedule: list[GPTVQTileOp] = []

        for m in range(tc.num_m_tiles):
            eff_m = min(tc.tile_m, M - m * tc.tile_m)

            for n in range(tc.num_n_tiles):
                eff_n = min(tc.tile_n, N - n * tc.tile_n)

                for k in range(tc.num_k_tiles):
                    eff_k = min(tc.tile_k, K - k * tc.tile_k)

                    is_first_k = k == 0
                    is_last_k = k == tc.num_k_tiles - 1

                    # Number of vector groups in this k-tile
                    k_groups = math.ceil(eff_k / d)
                    k_group_start = (k * tc.tile_k) // d

                    # Codebook offset: codebook is shared, always at base
                    # (same codebook for all tiles)
                    cb_offset = 0

                    # Index offset: indices stored row-major [M, K_groups]
                    # Row i, group j → index at (i * K_groups + j)
                    idx_offset = (
                        m * tc.tile_m * K_groups + k_group_start
                    ) * idx_bytes

                    # Scale offset: scales stored per-group [K_groups]
                    sc_offset = k_group_start * (scale_bytes + zp_bytes)

                    # Activation offset: A stored row-major [K, N]
                    act_offset = (k * tc.tile_k * N + n * tc.tile_n) * act_bpe

                    # Output offset: C stored row-major [M, N]
                    out_offset = (m * tc.tile_m * N + n * tc.tile_n) * acc_bytes

                    schedule.append(
                        GPTVQTileOp(
                            m_idx=m,
                            n_idx=n,
                            k_idx=k,
                            tile_m=eff_m,
                            tile_n=eff_n,
                            tile_k=eff_k,
                            load_codebook=is_first_k,
                            load_indices=True,
                            load_scales=use_scaling,
                            load_activation=True,
                            store_output=is_last_k,
                            accumulate=not is_first_k,
                            codebook_dram_offset=cb_offset,
                            index_dram_offset=idx_offset,
                            scale_dram_offset=sc_offset,
                            activation_dram_offset=act_offset,
                            output_dram_offset=out_offset,
                        )
                    )

        return tc, schedule
