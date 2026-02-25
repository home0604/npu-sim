from enum import Enum
import torch


class DataType(Enum):
    INT8 = ("int8", 1, torch.int8)
    FP16 = ("fp16", 2, torch.float16)
    BF16 = ("bf16", 2, torch.bfloat16)
    FP32 = ("fp32", 4, torch.float32)

    def __init__(self, label: str, num_bytes: int, torch_dtype: torch.dtype):
        self.label = label
        self.num_bytes = num_bytes
        self.torch_dtype = torch_dtype


class AccumulatorType(Enum):
    INT32 = ("int32", 4, torch.int32)
    FP32 = ("fp32", 4, torch.float32)

    def __init__(self, label: str, num_bytes: int, torch_dtype: torch.dtype):
        self.label = label
        self.num_bytes = num_bytes
        self.torch_dtype = torch_dtype
