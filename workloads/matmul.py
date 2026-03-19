from __future__ import annotations

from dataclasses import dataclass

from core.datatypes import DataType


@dataclass
class MatMulWorkload:
    """Describes a MatMul operation: C[M,N] = A[M,K] * B[K,N]."""

    M: int
    N: int
    K: int
    name: str = "MatMul"
    dtype: DataType = DataType.INT8

    @property
    def total_mac_ops(self) -> int:
        return self.M * self.N * self.K

    @property
    def weight_bytes(self) -> int:
        return self.M * self.K * self.dtype.num_bytes

    @property
    def activation_bytes(self) -> int:
        return self.K * self.N * self.dtype.num_bytes

    @property
    def output_bytes(self) -> int:
        return self.M * self.N * 4  # accumulator type (INT32/FP32)

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.activation_bytes + self.output_bytes
