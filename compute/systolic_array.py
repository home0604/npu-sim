from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.datatypes import AccumulatorType, DataType


@dataclass
class CycleBreakdown:
    """Cycle breakdown for a single tile computation."""

    pipeline_fill_cycles: int = 0
    compute_cycles: int = 0
    pipeline_drain_cycles: int = 0
    total_cycles: int = 0
    weight_load_cycles: int = 0


@dataclass
class TileResult:
    """Result of one tile computation on the systolic array."""

    output: torch.Tensor
    tile_m: int
    tile_n: int
    tile_k: int
    cycles: CycleBreakdown


class SystolicArray:
    """Weight-Stationary Systolic Array model.

    Uses PyTorch for actual computation and analytical formulas for cycle counting.

    Weight Stationary Dataflow:
    - Weights are pre-loaded into PEs and remain stationary
    - Activations flow horizontally through the array
    - Partial sums flow vertically (accumulate down columns)

    For an MxN array computing (tile_M, tile_K) x (tile_K, tile_N):
    - Pipeline fill:  M + N - 1 cycles (activations reach all PEs)
    - Computation:    K cycles (one K-element per cycle flows through)
    - Pipeline drain: M + N - 1 cycles (last results propagate out)
    - Total:          K + 2*(M + N - 1) cycles per tile

    When tile dimensions exceed array dimensions, multiple passes are needed.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        dtype: DataType = DataType.INT8,
        acc_dtype: AccumulatorType = AccumulatorType.INT32,
    ):
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.acc_dtype = acc_dtype

    def compute_tile(self, weight: torch.Tensor, activation: torch.Tensor) -> TileResult:
        """Execute a tile computation and return result with cycle count.

        Args:
            weight: shape (tile_m, tile_k)
            activation: shape (tile_k, tile_n)

        Returns:
            TileResult with output tensor and cycle breakdown.
        """
        tile_m, tile_k = weight.shape
        _, tile_n = activation.shape

        # Actual computation via PyTorch (use float for accumulation accuracy)
        output = torch.matmul(weight.float(), activation.float())
        if self.acc_dtype == AccumulatorType.INT32:
            output = output.to(torch.int32)

        cycles = self.analytical_cycles(tile_m, tile_n, tile_k)

        return TileResult(
            output=output,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            cycles=cycles,
        )

    def analytical_cycles(
        self, tile_m: int, tile_n: int, tile_k: int, include_weight_load: bool = False
    ) -> CycleBreakdown:
        """Compute analytical cycle count for a tile.

        When tile dimensions exceed array dimensions, multiple passes are used.
        Each pass processes min(M, remaining_m) x min(N, remaining_n).

        For K-dimension with WS accumulator registers: consecutive K-tiles
        don't need extra fill/drain between them (partial sums stay in PEs).
        So the formula for a single (m_pass, n_pass) with the full K:
            fill + K + drain = (eff_m + eff_n - 1) + tile_k + (eff_m + eff_n - 1)
        """
        M, N = self.rows, self.cols

        m_passes = (tile_m + M - 1) // M
        n_passes = (tile_n + N - 1) // N

        total_fill = 0
        total_compute = 0
        total_drain = 0

        for mp in range(m_passes):
            eff_m = min(M, tile_m - mp * M)
            for np_ in range(n_passes):
                eff_n = min(N, tile_n - np_ * N)

                fill = eff_m + eff_n - 1
                compute = tile_k
                drain = eff_m + eff_n - 1

                total_fill += fill
                total_compute += compute
                total_drain += drain

        total = total_fill + total_compute + total_drain

        weight_load = 0
        if include_weight_load:
            weight_load = self.weight_load_cycles(tile_m, tile_k)
            total += weight_load

        return CycleBreakdown(
            pipeline_fill_cycles=total_fill,
            compute_cycles=total_compute,
            pipeline_drain_cycles=total_drain,
            total_cycles=total,
            weight_load_cycles=weight_load,
        )

    def weight_load_cycles(self, tile_m: int, tile_k: int) -> int:
        """Cycles to load a weight tile into the systolic array.

        In WS dataflow, weights are loaded into the PEs before computation.
        With M columns (rows of array), can load one row of weights per cycle.
        Total: ceil(tile_m / M) * tile_k cycles.
        """
        m_passes = (tile_m + self.rows - 1) // self.rows
        return m_passes * tile_k

    def peak_tops(self) -> float:
        """Peak throughput in TOPS (Tera Operations Per Second)."""
        ops_per_cycle = self.rows * self.cols * 2  # MAC = 2 ops
        return ops_per_cycle * 1e6 / 1e12  # assuming 1GHz default
