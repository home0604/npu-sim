from __future__ import annotations

from dataclasses import dataclass

from .tiler import TileConfig, Tiler


@dataclass
class TileOp:
    """A single tile operation in the execution schedule."""

    m_idx: int
    n_idx: int
    k_idx: int
    tile_m: int  # effective tile dimensions (may be smaller at edges)
    tile_n: int
    tile_k: int

    load_weight: bool  # True if weight tile needs to be loaded from DRAM
    load_activation: bool  # True if activation tile needs to be loaded
    store_output: bool  # True if output should be written back (last K-tile)
    accumulate: bool  # True if accumulating onto existing partial sum

    # DRAM address offsets (relative to base)
    weight_dram_offset: int = 0
    activation_dram_offset: int = 0
    output_dram_offset: int = 0


class WSDataflow:
    """Weight Stationary dataflow tile schedule generation.

    Tile loop order (minimizes weight reloading):

        for m in range(num_m):
            for n in range(num_n):
                load_weight(m, k=0)          # weight stays in PE
                for k in range(num_k):
                    load_activation(k, n)
                    compute()                # accumulate partial sums
                    if k == last:
                        store_output(m, n)

    Weight tiles are loaded once per (m, n) pair and reused across all K-tiles.
    The K-loop accumulates partial sums in the PE output registers.
    """

    def __init__(self, tiler: Tiler):
        self.tiler = tiler

    def generate_schedule(
        self,
        M: int,
        N: int,
        K: int,
        weight_base_addr: int = 0,
        activation_base_addr: int = 0,
        output_base_addr: int = 0,
    ) -> tuple[TileConfig, list[TileOp]]:
        """Generate ordered list of tile operations for a MatMul.

        Args:
            M, N, K: MatMul dimensions C[M,N] = A[M,K] * B[K,N]
            weight_base_addr: DRAM base address for weight matrix A
            activation_base_addr: DRAM base address for activation matrix B
            output_base_addr: DRAM base address for output matrix C

        Returns:
            (tile_config, schedule) tuple
        """
        tc = self.tiler.compute_tiles(M, N, K)
        bpe = self.tiler.bytes_per_elem
        acc_bytes = self.tiler.acc_bytes
        schedule: list[TileOp] = []

        for m in range(tc.num_m_tiles):
            eff_m = min(tc.tile_m, M - m * tc.tile_m)

            for n in range(tc.num_n_tiles):
                eff_n = min(tc.tile_n, N - n * tc.tile_n)

                for k in range(tc.num_k_tiles):
                    eff_k = min(tc.tile_k, K - k * tc.tile_k)

                    is_first_k = k == 0
                    is_last_k = k == tc.num_k_tiles - 1

                    # Weight address: A[m*tile_m : m*tile_m+eff_m, k*tile_k : k*tile_k+eff_k]
                    # Stored row-major: row * K + col
                    w_offset = (m * tc.tile_m * K + k * tc.tile_k) * bpe

                    # Activation address: B[k*tile_k : k*tile_k+eff_k, n*tile_n : n*tile_n+eff_n]
                    a_offset = (k * tc.tile_k * N + n * tc.tile_n) * bpe

                    # Output address: C[m*tile_m : m*tile_m+eff_m, n*tile_n : n*tile_n+eff_n]
                    o_offset = (m * tc.tile_m * N + n * tc.tile_n) * acc_bytes

                    schedule.append(
                        TileOp(
                            m_idx=m,
                            n_idx=n,
                            k_idx=k,
                            tile_m=eff_m,
                            tile_n=eff_n,
                            tile_k=eff_k,
                            load_weight=is_first_k,  # only load on first K-tile
                            load_activation=True,
                            store_output=is_last_k,
                            accumulate=not is_first_k,
                            weight_dram_offset=w_offset,
                            activation_dram_offset=a_offset,
                            output_dram_offset=o_offset,
                        )
                    )

        return tc, schedule
