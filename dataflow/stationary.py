from __future__ import annotations

from dataclasses import dataclass

from dataflow.tiler import TileConfig, Tiler


@dataclass
class TileOp:
    """A single tile operation in the execution schedule."""

    m_idx: int
    n_idx: int
    k_idx: int
    tile_m: int
    tile_n: int
    tile_k: int

    load_weight: bool
    load_activation: bool
    store_output: bool
    accumulate: bool

    weight_dram_offset: int = 0
    activation_dram_offset: int = 0
    output_dram_offset: int = 0


class StationaryDataflow:
    """Unified stationary dataflow schedule generation for OS, WS, and IS.

    Dataflow determines which operand is "stationary" (kept in PEs) and
    consequently the loop order, load pattern, and store pattern.

    OS (Output Stationary):
        Loop: M -> N -> K (inner).  Output (m,n) stays in PE accumulators.
        Weight and activation loaded every K-tile; output stored on last K.

    WS (Weight Stationary):
        Loop: M -> K -> N (inner).  Weight W[m,k] stays in PEs.
        Weight loaded once per (m,k); activation loaded every N-tile;
        output stored on last K (accumulates over K).

    IS (Input Stationary):
        Loop: K -> N -> M (inner).  Activation A[k,n] stays in PEs.
        Activation loaded once per (k,n); weight loaded every M-tile;
        output stored on last K (accumulates over K).
    """

    def __init__(self, tiler: Tiler, dataflow: str = "OS"):
        self.tiler = tiler
        self.dataflow = dataflow

    def generate_schedule(
        self,
        M: int,
        N: int,
        K: int,
    ) -> tuple[TileConfig, list[TileOp]]:
        """Generate ordered list of tile operations for a MatMul.

        Args:
            M, N, K: MatMul dimensions C[M,N] = A[M,K] * B[K,N]

        Returns:
            (tile_config, schedule) tuple
        """
        tc = self.tiler.compute_tiles(M, N, K, dataflow=self.dataflow)
        bpe = self.tiler.bytes_per_elem
        acc_bytes = self.tiler.acc_bytes

        if self.dataflow == "WS":
            return self._schedule_ws(tc, M, N, K, bpe, acc_bytes)
        elif self.dataflow == "IS":
            return self._schedule_is(tc, M, N, K, bpe, acc_bytes)
        else:  # OS
            return self._schedule_os(tc, M, N, K, bpe, acc_bytes)

    def _schedule_os(
        self, tc: TileConfig, M: int, N: int, K: int, bpe: int, acc_bytes: int
    ) -> tuple[TileConfig, list[TileOp]]:
        """OS: M -> N -> K loop.  Output (m,n) accumulates over K in PEs."""
        schedule: list[TileOp] = []

        for m in range(tc.num_m_tiles):
            eff_m = min(tc.tile_m, M - m * tc.tile_m)
            for n in range(tc.num_n_tiles):
                eff_n = min(tc.tile_n, N - n * tc.tile_n)
                for k in range(tc.num_k_tiles):
                    eff_k = min(tc.tile_k, K - k * tc.tile_k)

                    is_last_k = k == tc.num_k_tiles - 1

                    w_offset = (m * tc.tile_m * K + k * tc.tile_k) * bpe
                    a_offset = (k * tc.tile_k * N + n * tc.tile_n) * bpe
                    o_offset = (m * tc.tile_m * N + n * tc.tile_n) * acc_bytes

                    schedule.append(
                        TileOp(
                            m_idx=m, n_idx=n, k_idx=k,
                            tile_m=eff_m, tile_n=eff_n, tile_k=eff_k,
                            load_weight=True,
                            load_activation=True,
                            store_output=is_last_k,
                            accumulate=k > 0,
                            weight_dram_offset=w_offset,
                            activation_dram_offset=a_offset,
                            output_dram_offset=o_offset,
                        )
                    )

        return tc, schedule

    def _schedule_ws(
        self, tc: TileConfig, M: int, N: int, K: int, bpe: int, acc_bytes: int
    ) -> tuple[TileConfig, list[TileOp]]:
        """WS: M -> K -> N loop.

        W[M,K] (weight) stays in PEs across N iterations.
        A[K,N] (activation) streams through along the temporal axis T=N.
        Partial sums for different K-tiles at the same (m,n) are accumulated.
        """
        schedule: list[TileOp] = []

        for m in range(tc.num_m_tiles):
            eff_m = min(tc.tile_m, M - m * tc.tile_m)
            for k in range(tc.num_k_tiles):
                eff_k = min(tc.tile_k, K - k * tc.tile_k)
                for n in range(tc.num_n_tiles):
                    eff_n = min(tc.tile_n, N - n * tc.tile_n)

                    is_first_n = n == 0
                    is_last_k = k == tc.num_k_tiles - 1

                    w_offset = (m * tc.tile_m * K + k * tc.tile_k) * bpe
                    a_offset = (k * tc.tile_k * N + n * tc.tile_n) * bpe
                    o_offset = (m * tc.tile_m * N + n * tc.tile_n) * acc_bytes

                    schedule.append(
                        TileOp(
                            m_idx=m, n_idx=n, k_idx=k,
                            tile_m=eff_m, tile_n=eff_n, tile_k=eff_k,
                            load_weight=is_first_n,
                            load_activation=True,
                            store_output=is_last_k,
                            accumulate=k > 0,
                            weight_dram_offset=w_offset,
                            activation_dram_offset=a_offset,
                            output_dram_offset=o_offset,
                        )
                    )

        return tc, schedule

    def _schedule_is(
        self, tc: TileConfig, M: int, N: int, K: int, bpe: int, acc_bytes: int
    ) -> tuple[TileConfig, list[TileOp]]:
        """IS: K -> N -> M loop.

        A[K,N] (activation) stays in PEs across M iterations.
        W[M,K] (weight) streams through along the temporal axis T=M.
        Partial sums for different K-tiles at the same (m,n) are accumulated.
        """
        schedule: list[TileOp] = []

        for k in range(tc.num_k_tiles):
            eff_k = min(tc.tile_k, K - k * tc.tile_k)
            for n in range(tc.num_n_tiles):
                eff_n = min(tc.tile_n, N - n * tc.tile_n)
                for m in range(tc.num_m_tiles):
                    eff_m = min(tc.tile_m, M - m * tc.tile_m)

                    is_first_m = m == 0
                    is_last_k = k == tc.num_k_tiles - 1

                    w_offset = (m * tc.tile_m * K + k * tc.tile_k) * bpe
                    a_offset = (k * tc.tile_k * N + n * tc.tile_n) * bpe
                    o_offset = (m * tc.tile_m * N + n * tc.tile_n) * acc_bytes

                    schedule.append(
                        TileOp(
                            m_idx=m, n_idx=n, k_idx=k,
                            tile_m=eff_m, tile_n=eff_n, tile_k=eff_k,
                            load_weight=True,
                            load_activation=is_first_m,
                            store_output=is_last_k,
                            accumulate=k > 0,
                            weight_dram_offset=w_offset,
                            activation_dram_offset=a_offset,
                            output_dram_offset=o_offset,
                        )
                    )

        return tc, schedule
