"""VQ dataflow tile schedule generation for OS, WS, and IS.

OS (Output Stationary) — M→N→K:
    Output C[m,n] accumulates in PEs over K.
    load_codebook / load_indices / load_scales on every tile (first-k flag).
    load_activation on every tile.

WS (Weight Stationary) — M→K→N:
    W_deq[m,k] stays in the dequant_weight buffer across N.
    load_codebook / load_indices / load_scales once per (m,k), i.e. n==0.
    load_activation on every tile (A[k,n] streams through N).

IS (Input Stationary) — K→N→M:
    A[k,n] stays in the activation buffer across M.
    load_activation once per (k,n), i.e. m==0.
    load_codebook / load_indices / load_scales on every tile
    (W[m,k] rows change per m; scales are re-used but small).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.config import VQConfig
from dataflow.vq_tiler import VQTileConfig, VQTiler


@dataclass
class VQTileOp:
    """A single tile operation in a VQ execution schedule."""

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


class VQDataflow:
    """VQ dataflow tile schedule generation (OS, WS, IS).

    Generates an ordered list of VQTileOp that defines:
    - What data to load (codebook, indices, scales, activation)
    - When to dequantize and compute
    - When to store output (last K-tile of each output tile)
    """

    def __init__(self, tiler: VQTiler, vq_config: VQConfig):
        self.tiler = tiler
        self.vq = vq_config

    def generate_schedule(
        self,
        M: int,
        N: int,
        K: int,
        dataflow: str = "OS",
        codebook_base_addr: int = 0,
        index_base_addr: int = 0,
        scale_base_addr: int = 0,
        activation_base_addr: int = 0,
        output_base_addr: int = 0,
    ) -> tuple[VQTileConfig, list[VQTileOp]]:
        """Generate ordered list of tile operations for a VQ MatMul.

        Args:
            M, N, K: MatMul dimensions C[M,N] = W_deq[M,K] * A[K,N]
            dataflow: "OS", "WS", or "IS"

        Returns:
            (tile_config, schedule) tuple
        """
        tc = self.tiler.compute_tiles(M, N, K)
        d = self.vq.vector_dim
        idx_bytes = self.vq.index_elem_bytes
        act_bpe = self.tiler.act_bytes_per_elem
        acc_bytes = self.tiler.acc_bytes
        use_scaling = self.vq.use_scaling
        scale_bytes = self.vq.scale_bytes if use_scaling else 0
        zp_bytes = self.vq.zero_point_bytes if use_scaling else 0
        K_groups = math.ceil(K / d)

        if dataflow == "WS":
            return self._schedule_ws(tc, M, N, K, d, idx_bytes, act_bpe, acc_bytes,
                                     use_scaling, scale_bytes, zp_bytes, K_groups)
        elif dataflow == "IS":
            return self._schedule_is(tc, M, N, K, d, idx_bytes, act_bpe, acc_bytes,
                                     use_scaling, scale_bytes, zp_bytes, K_groups)
        else:
            return self._schedule_os(tc, M, N, K, d, idx_bytes, act_bpe, acc_bytes,
                                     use_scaling, scale_bytes, zp_bytes, K_groups)

    def _schedule_os(
        self, tc, M, N, K, d, idx_bytes, act_bpe, acc_bytes,
        use_scaling, scale_bytes, zp_bytes, K_groups,
    ) -> tuple[VQTileConfig, list[VQTileOp]]:
        """OS: M→N→K.  Output C[m,n] accumulates in PEs over K."""
        schedule: list[VQTileOp] = []

        for m in range(tc.num_m_tiles):
            eff_m = min(tc.tile_m, M - m * tc.tile_m)
            for n in range(tc.num_n_tiles):
                eff_n = min(tc.tile_n, N - n * tc.tile_n)
                for k in range(tc.num_k_tiles):
                    eff_k = min(tc.tile_k, K - k * tc.tile_k)
                    is_first_k = k == 0
                    is_last_k = k == tc.num_k_tiles - 1
                    k_group_start = (k * tc.tile_k) // d

                    idx_offset = (m * tc.tile_m * K_groups + k_group_start) * idx_bytes
                    sc_offset = k_group_start * (scale_bytes + zp_bytes)
                    act_offset = (k * tc.tile_k * N + n * tc.tile_n) * act_bpe
                    out_offset = (m * tc.tile_m * N + n * tc.tile_n) * acc_bytes

                    schedule.append(VQTileOp(
                        m_idx=m, n_idx=n, k_idx=k,
                        tile_m=eff_m, tile_n=eff_n, tile_k=eff_k,
                        load_codebook=is_first_k,
                        load_indices=True,
                        load_scales=use_scaling,
                        load_activation=True,
                        store_output=is_last_k,
                        accumulate=not is_first_k,
                        codebook_dram_offset=0,
                        index_dram_offset=idx_offset,
                        scale_dram_offset=sc_offset,
                        activation_dram_offset=act_offset,
                        output_dram_offset=out_offset,
                    ))

        return tc, schedule

    def _schedule_ws(
        self, tc, M, N, K, d, idx_bytes, act_bpe, acc_bytes,
        use_scaling, scale_bytes, zp_bytes, K_groups,
    ) -> tuple[VQTileConfig, list[VQTileOp]]:
        """WS: M→K→N.  W_deq[m,k] stays in dequant_weight buffer across N.

        load_indices / load_codebook / load_scales only on n==0 (once per (m,k)).
        load_activation on every tile (A[k,n] streams through N).
        """
        schedule: list[VQTileOp] = []

        for m in range(tc.num_m_tiles):
            eff_m = min(tc.tile_m, M - m * tc.tile_m)
            for k in range(tc.num_k_tiles):
                eff_k = min(tc.tile_k, K - k * tc.tile_k)
                is_last_k = k == tc.num_k_tiles - 1
                k_group_start = (k * tc.tile_k) // d

                idx_offset = (m * tc.tile_m * K_groups + k_group_start) * idx_bytes
                sc_offset = k_group_start * (scale_bytes + zp_bytes)

                for n in range(tc.num_n_tiles):
                    eff_n = min(tc.tile_n, N - n * tc.tile_n)
                    is_first_n = n == 0

                    act_offset = (k * tc.tile_k * N + n * tc.tile_n) * act_bpe
                    out_offset = (m * tc.tile_m * N + n * tc.tile_n) * acc_bytes

                    schedule.append(VQTileOp(
                        m_idx=m, n_idx=n, k_idx=k,
                        tile_m=eff_m, tile_n=eff_n, tile_k=eff_k,
                        load_codebook=is_first_n,          # codebook once per (m,k)
                        load_indices=is_first_n,           # W[m,k] indices once per (m,k)
                        load_scales=use_scaling and is_first_n,
                        load_activation=True,              # A[k,n] streams through N
                        store_output=is_last_k,
                        accumulate=(k > 0),
                        codebook_dram_offset=0,
                        index_dram_offset=idx_offset,
                        scale_dram_offset=sc_offset,
                        activation_dram_offset=act_offset,
                        output_dram_offset=out_offset,
                    ))

        return tc, schedule

    def _schedule_is(
        self, tc, M, N, K, d, idx_bytes, act_bpe, acc_bytes,
        use_scaling, scale_bytes, zp_bytes, K_groups,
    ) -> tuple[VQTileConfig, list[VQTileOp]]:
        """IS: K→N→M.  A[k,n] stays in activation buffer across M.

        load_activation only on m==0 (once per (k,n)).
        load_indices on every tile (W[m,k] row changes per m).
        load_scales only on m==0 (scales are per k-group, same for all m).
        """
        schedule: list[VQTileOp] = []

        for k in range(tc.num_k_tiles):
            eff_k = min(tc.tile_k, K - k * tc.tile_k)
            is_last_k = k == tc.num_k_tiles - 1
            k_group_start = (k * tc.tile_k) // d
            sc_offset = k_group_start * (scale_bytes + zp_bytes)

            for n in range(tc.num_n_tiles):
                eff_n = min(tc.tile_n, N - n * tc.tile_n)
                act_offset = (k * tc.tile_k * N + n * tc.tile_n) * act_bpe

                for m in range(tc.num_m_tiles):
                    eff_m = min(tc.tile_m, M - m * tc.tile_m)
                    is_first_m = m == 0

                    idx_offset = (m * tc.tile_m * K_groups + k_group_start) * idx_bytes
                    out_offset = (m * tc.tile_m * N + n * tc.tile_n) * acc_bytes

                    schedule.append(VQTileOp(
                        m_idx=m, n_idx=n, k_idx=k,
                        tile_m=eff_m, tile_n=eff_n, tile_k=eff_k,
                        load_codebook=is_first_m,          # codebook once per (k,n)
                        load_indices=True,                 # W[m,k] rows change per m
                        load_scales=use_scaling and is_first_m,  # scales same for all m
                        load_activation=is_first_m,        # A[k,n] stays across M
                        store_output=is_last_k,
                        accumulate=(k > 0),
                        codebook_dram_offset=0,
                        index_dram_offset=idx_offset,
                        scale_dram_offset=sc_offset,
                        activation_dram_offset=act_offset,
                        output_dram_offset=out_offset,
                    ))

        return tc, schedule
