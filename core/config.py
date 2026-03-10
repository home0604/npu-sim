from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SystolicArrayConfig:
    rows: int = 32
    cols: int = 32
    clock_freq_mhz: int = 1000


@dataclass
class DataTypeConfig:
    compute_dtype: str = "INT8"
    accumulator_dtype: str = "INT32"

    @property
    def bytes_per_element(self) -> int:
        return {"INT8": 1, "FP16": 2, "BF16": 2, "FP32": 4}[self.compute_dtype]

    @property
    def accumulator_bytes(self) -> int:
        return {"INT32": 4, "FP32": 4}[self.accumulator_dtype]


@dataclass
class SRAMConfig:
    total_size_kb: int = 512
    num_banks: int = 32
    bank_width_bytes: int = 64
    port_type: str = "dual"  # "single" or "dual"
    read_latency_cycles: int = 1
    write_latency_cycles: int = 1
    weight_buffer_fraction: float = 0.4
    activation_buffer_fraction: float = 0.3
    output_buffer_fraction: float = 0.3
    # GPTVQ buffer fractions (used when gptvq.enabled is True)
    codebook_buffer_fraction: float = 0.05
    index_buffer_fraction: float = 0.20
    scale_buffer_fraction: float = 0.05


@dataclass
class DRAMConfig:
    use_dramsim3: bool = False
    config_file: str = "ext/DRAMsim3/configs/DDR4_8Gb_x16_2400.ini"
    output_dir: str = "dramsim3_output"
    bandwidth_gbps: float = 25.6
    latency_ns: float = 50.0


@dataclass
class DoubleBufferConfig:
    enabled: bool = True


@dataclass
class GPTVQConfig:
    """GPTVQ (GPT Vector Quantization) configuration.

    Basic mode: codebook + weight index → dequantized weight.
    Optional scaling: s * codebook[idx] + z per vector group.
    """

    enabled: bool = False
    codebook_size: int = 16  # R: number of centroids
    vector_dim: int = 2  # d: dimension of each codebook vector
    index_bits: int = 4  # bits per index (log2(codebook_size))
    num_codebooks: int = 1  # number of codebooks (1 for standard VQ)
    use_scaling: bool = False  # enable per-group scaling (s, z)
    scale_dtype: str = "FP16"  # dtype for scaling factors s
    zero_point_dtype: str = "FP16"  # dtype for zero points z
    codebook_dtype: str = "FP16"  # dtype for codebook entries
    dequant_pipeline_stages: int = 3  # pipeline depth of dequant unit
    dequant_throughput: int = 1  # vectors dequantized per cycle (pipelined)

    @property
    def codebook_entry_bytes(self) -> int:
        return {"FP16": 2, "BF16": 2, "FP32": 4, "INT8": 1}[self.codebook_dtype]

    @property
    def scale_bytes(self) -> int:
        return {"FP16": 2, "BF16": 2, "FP32": 4, "INT8": 1}[self.scale_dtype]

    @property
    def zero_point_bytes(self) -> int:
        return {"FP16": 2, "BF16": 2, "FP32": 4, "INT8": 1}[self.zero_point_dtype]

    @property
    def index_elem_bytes(self) -> int:
        """Bytes per index element (ceil(index_bits / 8))."""
        return (self.index_bits + 7) // 8


@dataclass
class NPUConfig:
    name: str = "NPU-Sim"
    systolic: SystolicArrayConfig = field(default_factory=SystolicArrayConfig)
    dtype: DataTypeConfig = field(default_factory=DataTypeConfig)
    sram: SRAMConfig = field(default_factory=SRAMConfig)
    dram: DRAMConfig = field(default_factory=DRAMConfig)
    double_buffer: DoubleBufferConfig = field(default_factory=DoubleBufferConfig)
    gptvq: GPTVQConfig = field(default_factory=GPTVQConfig)


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a dataclass instance."""
    if not isinstance(data, dict):
        return data
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, val in data.items():
        if key in fieldtypes:
            ft = fieldtypes[key]
            # Resolve string type annotations
            if isinstance(ft, str):
                ft = eval(ft)
            if hasattr(ft, "__dataclass_fields__") and isinstance(val, dict):
                kwargs[key] = _dict_to_dataclass(ft, val)
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> NPUConfig:
    """Load NPU configuration from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return _dict_to_dataclass(NPUConfig, data)
