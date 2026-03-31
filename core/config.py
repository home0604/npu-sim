from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SystolicArrayConfig:
    rows: int = 32
    cols: int = 32
    clock_freq_mhz: int = 1000
    dataflow: str = "OS"


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
class CodebookSRAMConfig:
    """Codebook SRAM overrides. 0 or "" means use the global SRAMConfig value."""
    num_banks: int = 0
    port_type: str = ""  # "" = use global


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
    # VQ buffer fractions (used when vq.enabled is True)
    codebook_buffer_fraction: float = 0.10
    index_buffer_fraction: float = 0.05
    scale_buffer_fraction: float = 0.05
    dequant_weight_buffer_fraction: float = 0.25
    # Codebook SRAM overrides (unset fields fall back to global values)
    codebook_sram: CodebookSRAMConfig = field(default_factory=lambda: CodebookSRAMConfig())


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
class VQConfig:
    """VQ (Vector Quantization) configuration.

    Basic mode: codebook + weight index → dequantized weight.
    Optional scaling: s * codebook[idx] + z per vector group.
    """

    enabled: bool = False
    codebook_size: int = 16  # R: number of centroids
    vector_dim: int = 2  # d: dimension of each codebook vector
    index_bits: int = 4  # bits per index (log2(codebook_size))
    num_stages: int = 1  # S: RVQ/AQ stage 수 (standard VQ = 1)
    use_scaling: bool = False  # enable per-group scaling (s, z)
    scale_dtype: str = "FP16"  # dtype for scaling factors s
    zero_point_dtype: str = "FP16"  # dtype for zero points z
    codebook_dtype: str = "FP16"  # dtype for codebook entries
    dequant_pipeline_stages: int = 3  # pipeline depth of dequant unit

    _DTYPE_BYTES = {"FP16": 2, "BF16": 2, "FP32": 4, "INT8": 1, "INT4": 0.5}

    @property
    def codebook_entry_bytes(self) -> float:
        return self._DTYPE_BYTES[self.codebook_dtype]

    @property
    def scale_bytes(self) -> int:
        return int(self._DTYPE_BYTES[self.scale_dtype])

    @property
    def zero_point_bytes(self) -> int:
        return int(self._DTYPE_BYTES[self.zero_point_dtype])

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
    vq: VQConfig = field(default_factory=VQConfig)


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


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | Path) -> NPUConfig:
    """Load NPU configuration from a YAML file.

    Supports `extends: path/to/base.yaml` for config inheritance.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if base := data.pop("extends", None):
        with open(path.parent / base) as f:
            data = _deep_merge(yaml.safe_load(f), data)
    return _dict_to_dataclass(NPUConfig, data)
