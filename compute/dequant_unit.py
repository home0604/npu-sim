"""VQ Dequantization Unit.

Converts compressed weight representation (codebook + indices + optional scaling)
into dequantized weight tiles for the systolic array.

Basic mode:
    W[i, j*d:(j+1)*d] = codebook[indices[i, j]]

With scaling:
    W[i, j*d:(j+1)*d] = scales[j] * codebook[indices[i, j]] + zero_points[j]

With RVQ (num_stages > 1):
    W[i, j*d:(j+1)*d] = sum(codebook_s[indices_s[i, j]] for s in stages)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from core.config import VQConfig, SRAMConfig


@dataclass
class DequantResult:
    """Result of dequantizing a weight tile."""

    output: torch.Tensor  # dequantized weight [tile_m, tile_k]
    cycles: int  # analytical dequantization cycles
    num_vectors: int  # number of vectors processed


class DequantizationUnit:
    """VQ dequantization hardware unit model.

    Performs codebook lookup and optional scale/zero-point application.
    Dequant cycles are derived from SRAM bank structure:

        read_lat  = ceil(E * S / (N_bank * w_b))  [cycles per lookup]
        write_lat = SRAM write latency             [cycles per result write]
        total = pipeline_stages + num_vectors * max(read_lat, write_lat)

    Where:
        E      = vector_dim * codebook_entry_bytes * 8  (bits per entry)
        S      = num_stages  (RVQ/AQ stages)
        N_bank = SRAM bank count
        w_b    = bank_width_bytes * 8  (bits per bank)

    Read (codebook) and write (dequant_weight) use separate physical SRAMs,
    so they pipeline: the latency bottleneck is max(read_lat, write_lat).
    """

    def __init__(self, config: VQConfig, sram_config: SRAMConfig):
        self.config = config
        self.codebook_size = config.codebook_size
        self.vector_dim = config.vector_dim
        self.index_bits = config.index_bits
        self.use_scaling = config.use_scaling
        self.pipeline_stages = config.dequant_pipeline_stages
        cb_banks = sram_config.codebook_sram.num_banks
        self.sram_n_banks = cb_banks if cb_banks > 0 else sram_config.num_banks
        self.sram_bank_width_bytes = sram_config.bank_width_bytes
        self.sram_read_latency = sram_config.read_latency_cycles
        self.sram_write_latency = sram_config.write_latency_cycles

    def _read_latency_per_lookup(self) -> int:
        """Cycles to read one codebook entry using all SRAM banks in parallel.

        read_latency = ceil(E * S / (N_bank * w_b))
        Minimum 1 cycle.
        """
        E_bits = self.vector_dim * self.config.codebook_entry_bytes * 8
        S = self.config.num_stages
        N_bank = self.sram_n_banks
        w_b_bits = self.sram_bank_width_bytes * 8
        return max(1, math.ceil((E_bits * S) / (N_bank * w_b_bits)))

    def _latency_per_lookup(self) -> int:
        """Cycles per lookup in steady state (pipelined read + write).

        Codebook read and dequant_weight write use separate SRAMs,
        so they overlap: bottleneck = max(read_lat, write_lat).
        """
        return max(self._read_latency_per_lookup(), self.sram_write_latency)

    def dequant_cycles(self, tile_m: int, tile_k: int) -> int:
        """Analytical cycle count for dequantizing a weight tile.

        Args:
            tile_m: number of rows in the weight tile
            tile_k: number of columns (must be multiple of vector_dim)

        Returns:
            pipeline_stages + num_lookups * max(read_lat, write_lat)
        """
        d = self.vector_dim
        num_lookups = tile_m * math.ceil(tile_k / d)
        latency = self._latency_per_lookup()
        return self.pipeline_stages + num_lookups * latency

    def fused_preload_cycles(
        self, tile_m: int, tile_n: int, tile_k: int,
        array_rows: int, array_cols: int, dataflow: str,
    ) -> int:
        """Fused dequant+preload cycles for WS dataflow.

        Each cycle: 1 codebook lookup → d scalars → d PEs on SA top row.
        Per SA row: ceil(eff_sc / d) lookups.  Per pass: eff_sr rows.
        Lookup latency from codebook SRAM config (same as _latency_per_lookup).
        """
        if dataflow == "WS":
            sr, sc = tile_k, tile_n
        elif dataflow == "IS":
            sr, sc = tile_k, tile_m
        else:
            sr, sc = tile_m, tile_n

        R, C = array_rows, array_cols
        d = self.vector_dim
        lat = self._latency_per_lookup()
        total = 0
        for sp in range((sr + R - 1) // R):
            eff_sr = min(R, sr - sp * R)
            for cp in range((sc + C - 1) // C):
                eff_sc = min(C, sc - cp * C)
                total += eff_sr * math.ceil(eff_sc / d) * lat
        return total

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

        flat_indices = indices.long().flatten()  # [tile_m * num_groups]
        vectors = codebook[flat_indices]          # [tile_m * num_groups, d]
        vectors = vectors.view(tile_m, num_groups, d)

        if self.use_scaling and scales is not None:
            vectors = vectors * scales.unsqueeze(0).unsqueeze(-1)
            if zero_points is not None:
                vectors = vectors + zero_points.unsqueeze(0).unsqueeze(-1)

        output = vectors.reshape(tile_m, num_groups * d)
        if output.shape[1] > tile_k:
            output = output[:, :tile_k]

        cycles = self.dequant_cycles(tile_m, tile_k)

        return DequantResult(
            output=output,
            cycles=cycles,
            num_vectors=num_vectors,
        )
