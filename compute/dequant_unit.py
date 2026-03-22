"""GPTVQ Dequantization Unit.

Converts compressed weight representation (codebook + indices + optional scaling)
into dequantized weight tiles for the systolic array.

Basic mode:
    W[i, j*d:(j+1)*d] = codebook[indices[i, j]]

With scaling:
    W[i, j*d:(j+1)*d] = scales[j] * codebook[indices[i, j]] + zero_points[j]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from core.config import GPTVQConfig


@dataclass
class DequantResult:
    """Result of dequantizing a weight tile."""

    output: torch.Tensor  # dequantized weight [tile_m, tile_k]
    cycles: int  # analytical dequantization cycles
    num_vectors: int  # number of vectors processed


class DequantizationUnit:
    """GPTVQ dequantization hardware unit model.

    Performs codebook lookup and optional scale/zero-point application.
    Uses analytical cycle formula with configurable pipeline depth and throughput.

    Pipeline: index_read → codebook_lookup → scale_multiply (optional)
    Throughput: dequant_throughput vectors per cycle after pipeline fills.
    """

    def __init__(self, config: GPTVQConfig):
        self.config = config
        self.codebook_size = config.codebook_size
        self.vector_dim = config.vector_dim
        self.index_bits = config.index_bits
        self.use_scaling = config.use_scaling
        self.pipeline_stages = config.dequant_pipeline_stages
        self.throughput = config.dequant_throughput

    def dequant_cycles(self, tile_m: int, tile_k: int) -> int:
        """Analytical cycle count for dequantizing a weight tile.

        Args:
            tile_m: number of rows in the weight tile
            tile_k: number of columns (must be multiple of vector_dim)

        Returns:
            Total dequantization cycles (pipeline fill + steady state).
        """
        d = self.vector_dim
        num_vectors = tile_m * math.ceil(tile_k / d)
        # Pipelined throughput: after pipeline fills, throughput vectors/cycle
        cycles = self.pipeline_stages + math.ceil(num_vectors / self.throughput)
        return cycles

    def dequantize(
        self,
        indices: torch.Tensor,
        codebook: torch.Tensor,
        tile_m: int,
        tile_k: int,
        scales: torch.Tensor | None = None,
        zero_points: torch.Tensor | None = None,
    ) -> DequantResult:
        """Functional dequantization for correctness checking.

        Args:
            indices: [tile_m, tile_k // vector_dim] integer indices
            codebook: [codebook_size, vector_dim] centroid vectors
            tile_m: weight tile rows
            tile_k: weight tile columns
            scales: optional [tile_k // vector_dim] per-group scale factors
            zero_points: optional [tile_k // vector_dim] per-group zero points

        Returns:
            DequantResult with dequantized weight tensor and cycle count.
        """
        d = self.vector_dim
        num_groups = math.ceil(tile_k / d)
        num_vectors = tile_m * num_groups

        # Codebook lookup: gather vectors by index
        flat_indices = indices.long().flatten()  # [tile_m * num_groups]
        vectors = codebook[flat_indices]  # [tile_m * num_groups, d]
        vectors = vectors.view(tile_m, num_groups, d)  # [tile_m, num_groups, d]

        if self.use_scaling and scales is not None:
            vectors = vectors * scales.unsqueeze(0).unsqueeze(-1)
            if zero_points is not None:
                vectors = vectors + zero_points.unsqueeze(0).unsqueeze(-1)

        # Reshape to [tile_m, tile_k]
        output = vectors.reshape(tile_m, num_groups * d)
        if output.shape[1] > tile_k:
            output = output[:, :tile_k]

        cycles = self.dequant_cycles(tile_m, tile_k)

        return DequantResult(
            output=output,
            cycles=cycles,
            num_vectors=num_vectors,
        )
