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
    """Systolic Array model supporting OS/WS/IS dataflows.

    Uses PyTorch for actual computation and analytical formulas for cycle counting.

    Dataflow mapping (EONSim-style SR/SC/T abstraction):
    - OS: SR=M, SC=N, T=K  (output stays in PE, accumulate over K)
    - WS: SR=K, SC=N, T=M  (weight stays in PE, inputs stream through)
    - IS: SR=K, SC=M, T=N  (input stays in PE, weights stream through)

    For an RxC array with mapped dimensions (SR, SC, T):
    - Pipeline fill:  eff_sr + eff_sc - 1 cycles
    - Computation:    T cycles
    - Pipeline drain: eff_sr + eff_sc - 1 cycles

    When mapped spatial dimensions exceed array dimensions, multiple passes are needed.
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

        cycles = self.analytical_cycles(tile_m, tile_n, tile_k, dataflow="OS")

        return TileResult(
            output=output,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            cycles=cycles,
        )

    def _map_dataflow(
        self, tile_m: int, tile_n: int, tile_k: int, dataflow: str
    ) -> tuple[int, int, int]:
        """Map (M, N, K) to (SR, SC, T) based on dataflow type."""
        if dataflow == "WS":
            return tile_k, tile_n, tile_m
        elif dataflow == "IS":
            return tile_k, tile_m, tile_n
        else:  # OS (default)
            return tile_m, tile_n, tile_k

    def analytical_cycles(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        include_weight_load: bool = False,
        dataflow: str = "OS",
    ) -> CycleBreakdown:
        """Compute analytical cycle count for a tile.

        Maps (tile_m, tile_n, tile_k) to spatial rows (SR), spatial cols (SC),
        and temporal axis (T) based on the dataflow, then applies the unified
        systolic array formula.

        When mapped spatial dimensions exceed array dimensions, multiple passes
        are used. Each pass processes min(R, remaining_sr) x min(C, remaining_sc).
        """
        R, C = self.rows, self.cols
        sr, sc, t = self._map_dataflow(tile_m, tile_n, tile_k, dataflow)

        sr_passes = (sr + R - 1) // R
        sc_passes = (sc + C - 1) // C

        total_fill = 0
        total_compute = 0
        total_drain = 0

        for sp in range(sr_passes):
            eff_sr = min(R, sr - sp * R)
            for cp in range(sc_passes):
                eff_sc = min(C, sc - cp * C)

                fill = eff_sr + eff_sc - 1
                compute = t
                drain = eff_sr + eff_sc - 1

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
