from __future__ import annotations

from dataclasses import dataclass, field

from core.datatypes import DataType


@dataclass
class AttentionConfig:
    """Configuration for multi-head attention."""

    seq_len: int
    hidden_dim: int
    num_heads: int
    head_dim: int | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads


@dataclass
class AttentionOp:
    """A single operation within attention."""

    name: str
    op_type: str  # "matmul", "softmax", "layernorm", "gelu", "add"
    M: int = 0
    N: int = 0
    K: int = 0
    seq_len: int = 0
    hidden_dim: int = 0
    num_heads: int = 0


def generate_attention_ops(config: AttentionConfig) -> list[AttentionOp]:
    """Generate the list of operations for multi-head attention.

    Operations:
    1. QKV projection:    MatMul(S, 3*H*d, D)
    2. Attention scores:  H x MatMul(S, S, d)
    3. Softmax
    4. Attention output:  H x MatMul(S, d, S)
    5. Output projection: MatMul(S, D, D)
    """
    S = config.seq_len
    D = config.hidden_dim
    H = config.num_heads
    d = config.head_dim

    ops: list[AttentionOp] = []

    # QKV projection (fused)
    ops.append(AttentionOp(name="QKV_proj", op_type="matmul", M=S, N=3 * H * d, K=D))

    # Attention scores per head
    for h in range(H):
        ops.append(AttentionOp(name=f"attn_score_h{h}", op_type="matmul", M=S, N=S, K=d))

    # Softmax
    ops.append(
        AttentionOp(
            name="softmax", op_type="softmax", seq_len=S, num_heads=H
        )
    )

    # Attention output per head
    for h in range(H):
        ops.append(AttentionOp(name=f"attn_out_h{h}", op_type="matmul", M=S, N=d, K=S))

    # Output projection
    ops.append(AttentionOp(name="output_proj", op_type="matmul", M=S, N=D, K=D))

    return ops


@dataclass
class TransformerLayerConfig:
    """Configuration for a full transformer layer."""

    seq_len: int
    hidden_dim: int
    num_heads: int
    ffn_dim: int | None = None
    head_dim: int | None = None

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.hidden_dim
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads


def generate_transformer_layer_ops(config: TransformerLayerConfig) -> list[AttentionOp]:
    """Generate all operations for a full transformer layer.

    1. LayerNorm
    2. Multi-head Attention
    3. Residual Add
    4. LayerNorm
    5. FFN Up: MatMul(S, ffn_dim, D)
    6. GELU
    7. FFN Down: MatMul(S, D, ffn_dim)
    8. Residual Add
    """
    S = config.seq_len
    D = config.hidden_dim

    ops: list[AttentionOp] = []

    # LayerNorm 1
    ops.append(AttentionOp(name="layernorm_1", op_type="layernorm", seq_len=S, hidden_dim=D))

    # Attention
    attn_config = AttentionConfig(S, D, config.num_heads, config.head_dim)
    ops.extend(generate_attention_ops(attn_config))

    # Residual add
    ops.append(AttentionOp(name="residual_1", op_type="add", seq_len=S, hidden_dim=D))

    # LayerNorm 2
    ops.append(AttentionOp(name="layernorm_2", op_type="layernorm", seq_len=S, hidden_dim=D))

    # FFN up projection
    ops.append(AttentionOp(name="ffn_up", op_type="matmul", M=S, N=config.ffn_dim, K=D))

    # GELU
    ops.append(AttentionOp(name="gelu", op_type="gelu", seq_len=S, hidden_dim=config.ffn_dim))

    # FFN down projection
    ops.append(AttentionOp(name="ffn_down", op_type="matmul", M=S, N=D, K=config.ffn_dim))

    # Residual add
    ops.append(AttentionOp(name="residual_2", op_type="add", seq_len=S, hidden_dim=D))

    return ops
