from __future__ import annotations

import math
from dataclasses import dataclass

from core.datatypes import DataType


@dataclass
class SpecialFunctionResult:
    cycles: int
    memory_read_bytes: int
    memory_write_bytes: int


class SpecialFunctionUnit:
    """Analytical cycle models for non-MatMul operations.

    These operations use dedicated hardware or element-wise/reduction units,
    not the systolic array. Cycle estimates are approximate.
    """

    @staticmethod
    def softmax_cycles(seq_len: int, num_heads: int, dtype: DataType) -> SpecialFunctionResult:
        """Analytical cycle count for Softmax.

        Per row of length seq_len:
        1. Find max: tree reduction -> log2(seq_len) comparisons
        2. Subtract max: seq_len ops
        3. Exponential: seq_len ops (lookup table, ~4 cycles each)
        4. Sum: tree reduction -> log2(seq_len) additions
        5. Division: seq_len divisions (~4 cycles each)

        Total per row: ~10 * seq_len + 2*log2(seq_len)
        Total: num_heads * seq_len * per_row (score matrix is seq_len x seq_len per head)
        """
        log2_s = int(math.ceil(math.log2(max(seq_len, 2))))
        per_row = seq_len * 10 + 2 * log2_s
        total_rows = num_heads * seq_len  # each head has seq_len rows
        total_cycles = total_rows * per_row

        # Memory: read score matrix + write softmax output
        matrix_bytes = num_heads * seq_len * seq_len * dtype.num_bytes
        return SpecialFunctionResult(
            cycles=total_cycles,
            memory_read_bytes=matrix_bytes,
            memory_write_bytes=matrix_bytes,
        )

    @staticmethod
    def layernorm_cycles(
        seq_len: int, hidden_dim: int, dtype: DataType
    ) -> SpecialFunctionResult:
        """Analytical cycle count for LayerNorm.

        Per token (one row of hidden_dim):
        1. Compute mean: hidden_dim additions + 1 division
        2. Compute variance: hidden_dim subtractions + muls + sum + div
        3. Normalize: hidden_dim muls + additions
        4. Scale and shift (gamma, beta): hidden_dim muls + additions

        Total per token: ~5 * hidden_dim + 2*log2(hidden_dim)
        Total: seq_len tokens
        """
        log2_d = int(math.ceil(math.log2(max(hidden_dim, 2))))
        per_token = 5 * hidden_dim + 2 * log2_d
        total_cycles = seq_len * per_token

        # Memory: read input + gamma + beta, write output
        input_bytes = seq_len * hidden_dim * dtype.num_bytes
        param_bytes = hidden_dim * 4 * 2  # gamma + beta in FP32
        return SpecialFunctionResult(
            cycles=total_cycles,
            memory_read_bytes=input_bytes + param_bytes,
            memory_write_bytes=input_bytes,
        )

    @staticmethod
    def gelu_cycles(seq_len: int, hidden_dim: int, dtype: DataType) -> SpecialFunctionResult:
        """Analytical cycle count for GELU activation.

        Element-wise operation: ~8 cycles per element (approximation using tanh).
        """
        num_elements = seq_len * hidden_dim
        total_cycles = num_elements * 8

        elem_bytes = num_elements * dtype.num_bytes
        return SpecialFunctionResult(
            cycles=total_cycles,
            memory_read_bytes=elem_bytes,
            memory_write_bytes=elem_bytes,
        )

    @staticmethod
    def add_residual_cycles(
        seq_len: int, hidden_dim: int, dtype: DataType
    ) -> SpecialFunctionResult:
        """Element-wise addition for residual connection. 1 cycle per element."""
        num_elements = seq_len * hidden_dim
        elem_bytes = num_elements * dtype.num_bytes
        return SpecialFunctionResult(
            cycles=num_elements,
            memory_read_bytes=elem_bytes * 2,  # two inputs
            memory_write_bytes=elem_bytes,
        )
