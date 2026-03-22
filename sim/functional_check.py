"""Functionality check: run tile schedule with real tensors and compare to reference."""

from __future__ import annotations

import math

import torch

from compute.dequant_unit import DequantizationUnit
from compute.systolic_array import SystolicArray
from core.datatypes import AccumulatorType, DataType
from dataflow.gptvq_dataflow import GPTVQTileOp
from dataflow.gptvq_tiler import GPTVQTileConfig
from dataflow.tiler import TileConfig, Tiler
from dataflow.stationary import StationaryDataflow, TileOp


def run_schedule_functional(
    schedule: list[TileOp],
    tile_config: TileConfig,
    A: torch.Tensor,
    B: torch.Tensor,
    systolic: SystolicArray,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """Execute the tile schedule with real data; return simulated output C_sim.

    C_sim[M,N] = A[M,K] @ B[K,N] computed tile-by-tile in the same order as the
    cycle-accurate scheduler (m, n, k loop, accumulate over k).

    Args:
        schedule: List of TileOp from StationaryDataflow.generate_schedule.
        tile_config: TileConfig from Tiler.compute_tiles(M, N, K).
        A: Weight matrix [M, K] in compute dtype.
        B: Activation matrix [K, N] in compute dtype.
        systolic: SystolicArray used for compute_tile.
        M, N, K: MatMul dimensions.

    Returns:
        C_sim: Output tensor [M, N] in accumulator dtype.
    """
    acc_dtype = systolic.acc_dtype.torch_dtype
    C_sim = torch.zeros((M, N), dtype=acc_dtype, device=A.device)

    for tile in schedule:
        row_start_a = tile.m_idx * tile_config.tile_m
        col_start_a = tile.k_idx * tile_config.tile_k
        weight_tile = A[
            row_start_a : row_start_a + tile.tile_m,
            col_start_a : col_start_a + tile.tile_k,
        ]

        row_start_b = tile.k_idx * tile_config.tile_k
        col_start_b = tile.n_idx * tile_config.tile_n
        act_tile = B[
            row_start_b : row_start_b + tile.tile_k,
            col_start_b : col_start_b + tile.tile_n,
        ]

        result = systolic.compute_tile(weight_tile, act_tile).output

        out_row = tile.m_idx * tile_config.tile_m
        out_col = tile.n_idx * tile_config.tile_n
        if tile.accumulate:
            C_sim[out_row : out_row + tile.tile_m, out_col : out_col + tile.tile_n] += result
        else:
            C_sim[out_row : out_row + tile.tile_m, out_col : out_col + tile.tile_n] = result

    return C_sim


def reference_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    acc_dtype: AccumulatorType,
) -> torch.Tensor:
    """Reference C[M,N] = A[M,K] @ B[K,N] with accumulator dtype.

    Matches systolic logic: float matmul then cast to acc_dtype.
    """
    C = torch.matmul(A.float(), B.float())
    return C.to(acc_dtype.torch_dtype)


def reference_attention(
    x: torch.Tensor,
    W_qkv: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reference multi-head attention in float. Out shape [seq_len, hidden_dim]."""
    # x [S, D], W_qkv [D, 3*D], W_o [D, D]
    x = x.float()
    W_qkv = W_qkv.float()
    W_o = W_o.float()
    S, D = x.shape
    qkv = torch.matmul(x, W_qkv)
    qkv = qkv.view(S, 3, num_heads, head_dim).transpose(1, 2)  # [S, H, 3, d]
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each [S, H, d]
    scale = head_dim ** -0.5
    # scores[s, t, h] = sum_d q[s,h,d]*k[t,h,d] -> [S, S, H]
    scores = torch.einsum("shd,thd->sth", q, k) * scale
    attn = torch.softmax(scores, dim=1)  # over key dim t
    # out[s, h, d] = sum_t attn[s,t,h]*v[t,h,d]
    out = torch.einsum("sth,thd->shd", attn, v)
    out = out.reshape(S, D)
    return torch.matmul(out, W_o)


def run_attention_functional(
    x: torch.Tensor,
    W_qkv: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    head_dim: int,
    dataflow: StationaryDataflow,
    tiler: Tiler,
    systolic: SystolicArray,
) -> torch.Tensor:
    """Run attention with tile-by-tile matmuls (same schedule as simulator). Output float [seq_len, hidden_dim]."""
    S, D = x.shape
    x = x.float()
    W_qkv = W_qkv.float()
    W_o = W_o.float()

    def matmul_func(M: int, N: int, K: int, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        tc, schedule = dataflow.generate_schedule(M, N, K)
        return run_schedule_functional(schedule, tc, A, B, systolic, M, N, K)

    qkv = matmul_func(S, 3 * D, D, x, W_qkv).float()
    heads_out = []
    for h in range(num_heads):
        q_h = qkv[:, h * head_dim : (h + 1) * head_dim]
        k_h = qkv[:, D + h * head_dim : D + (h + 1) * head_dim]
        v_h = qkv[:, 2 * D + h * head_dim : 2 * D + (h + 1) * head_dim]
        scores = matmul_func(S, S, head_dim, q_h, k_h.T).float()
        scores = scores * (head_dim ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        out_h = matmul_func(S, head_dim, S, attn, v_h).float()
        heads_out.append(out_h)
    out = torch.cat(heads_out, dim=-1)
    return matmul_func(S, D, D, out, W_o).float()


def reference_transformer_layer(
    x: torch.Tensor,
    W_ln1_gamma: torch.Tensor,
    W_ln1_beta: torch.Tensor,
    W_qkv: torch.Tensor,
    W_o: torch.Tensor,
    W_ln2_gamma: torch.Tensor,
    W_ln2_beta: torch.Tensor,
    W_ffn_up: torch.Tensor,
    W_ffn_down: torch.Tensor,
    num_heads: int,
    head_dim: int,
    ffn_dim: int,
) -> torch.Tensor:
    """Reference transformer layer (Pre-LN): LN -> Attn -> Residual -> LN -> FFN -> Residual. Float."""
    x = x.float()
    ln1 = torch.nn.functional.layer_norm(x, (x.shape[-1],), W_ln1_gamma.float(), W_ln1_beta.float())
    attn_out = reference_attention(ln1, W_qkv, W_o, num_heads, head_dim)
    x = x + attn_out
    ln2 = torch.nn.functional.layer_norm(x, (x.shape[-1],), W_ln2_gamma.float(), W_ln2_beta.float())
    ffn = torch.matmul(ln2, W_ffn_up.float())
    ffn = torch.nn.functional.gelu(ffn)
    ffn = torch.matmul(ffn, W_ffn_down.float())
    return x + ffn


def run_transformer_layer_functional(
    x: torch.Tensor,
    W_ln1_gamma: torch.Tensor,
    W_ln1_beta: torch.Tensor,
    W_qkv: torch.Tensor,
    W_o: torch.Tensor,
    W_ln2_gamma: torch.Tensor,
    W_ln2_beta: torch.Tensor,
    W_ffn_up: torch.Tensor,
    W_ffn_down: torch.Tensor,
    num_heads: int,
    head_dim: int,
    ffn_dim: int,
    dataflow: StationaryDataflow,
    tiler: Tiler,
    systolic: SystolicArray,
) -> torch.Tensor:
    """Run transformer layer with tile matmuls; LN/GELU/add in float. Output float [seq_len, hidden_dim]."""
    x = x.float()
    ln1 = torch.nn.functional.layer_norm(x, (x.shape[-1],), W_ln1_gamma.float(), W_ln1_beta.float())
    attn_out = run_attention_functional(ln1, W_qkv, W_o, num_heads, head_dim, dataflow, tiler, systolic)
    x = x + attn_out
    ln2 = torch.nn.functional.layer_norm(x, (x.shape[-1],), W_ln2_gamma.float(), W_ln2_beta.float())
    S, D = x.shape

    def matmul_func(M: int, N: int, K: int, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        tc, schedule = dataflow.generate_schedule(M, N, K)
        return run_schedule_functional(schedule, tc, A, B, systolic, M, N, K)

    ffn = matmul_func(S, ffn_dim, D, ln2, W_ffn_up.float()).float()
    ffn = torch.nn.functional.gelu(ffn)
    ffn = matmul_func(S, D, ffn_dim, ffn, W_ffn_down.float()).float()
    return x + ffn


def compute_diff_report(
    C_sim: torch.Tensor,
    C_ref: torch.Tensor,
    acc_dtype: AccumulatorType,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    max_sample: int = 10,
) -> dict[str, Any]:
    """Compute a concrete difference report between C_sim and C_ref.

    Verifies that the compute core's mul/accum with the given data type matches
    the reference. Returns a dict suitable for logging or JSON.
    """
    report: dict[str, Any] = {
        "match": False,
        "acc_dtype": acc_dtype.name,
        "shape": list(C_sim.shape),
        "num_elements": int(C_sim.numel()),
        "num_mismatch": 0,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "l1_error": None,
        "sample_mismatches": [],
    }

    if C_sim.shape != C_ref.shape:
        report["error"] = f"Shape mismatch: C_sim {C_sim.shape} vs C_ref {C_ref.shape}"
        return report

    diff_tensor = (C_sim.float() - C_ref.float()).abs()
    mismatch_mask = C_sim != C_ref
    if acc_dtype == AccumulatorType.INT32:
        # For INT32 we consider exact match; diff for reporting still numeric
        report["num_mismatch"] = int(mismatch_mask.sum().item())
    else:
        # FP32: mismatch = not allclose
        mismatch_mask = ~torch.isclose(C_sim, C_ref, rtol=rtol, atol=atol)
        report["num_mismatch"] = int(mismatch_mask.sum().item())

    report["max_abs_diff"] = float(diff_tensor.max().item())
    report["mean_abs_diff"] = float(diff_tensor.mean().item())
    report["l1_error"] = float(diff_tensor.sum().item())

    if acc_dtype == AccumulatorType.INT32:
        report["match"] = report["num_mismatch"] == 0
    else:
        report["match"] = report["num_mismatch"] == 0

    # Sample first few mismatches: (i, j, c_sim, c_ref, abs_diff)
    if report["num_mismatch"] > 0:
        flat_idx = torch.where(mismatch_mask.flatten())[0]
        n_sample = min(max_sample, flat_idx.shape[0])
        for k in range(n_sample):
            idx = flat_idx[k].item()
            i = idx // C_sim.shape[1]
            j = idx % C_sim.shape[1]
            c_s = int(C_sim.flatten()[idx].item()) if acc_dtype == AccumulatorType.INT32 else float(C_sim.flatten()[idx].item())
            c_r = int(C_ref.flatten()[idx].item()) if acc_dtype == AccumulatorType.INT32 else float(C_ref.flatten()[idx].item())
            d = float(diff_tensor.flatten()[idx].item())
            report["sample_mismatches"].append({"i": i, "j": j, "C_sim": c_s, "C_ref": c_r, "abs_diff": d})

    return report


def compare_outputs(
    C_sim: torch.Tensor,
    C_ref: torch.Tensor,
    acc_dtype: AccumulatorType,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> tuple[bool, str]:
    """Compare simulated output to reference. Return (match, message)."""
    if C_sim.shape != C_ref.shape:
        return False, f"Shape mismatch: C_sim {C_sim.shape} vs C_ref {C_ref.shape}"

    if acc_dtype == AccumulatorType.INT32:
        if not torch.equal(C_sim, C_ref):
            diff = (C_sim != C_ref).sum().item()
            max_abs_diff = (C_sim.float() - C_ref.float()).abs().max().item()
            return False, (
                f"Output mismatch (INT32): {diff} elements differ, max_abs_diff={max_abs_diff}"
            )
        return True, "Output matches reference (exact, INT32)."

    # FP32 accumulator
    if not torch.allclose(C_sim, C_ref, rtol=rtol, atol=atol):
        max_diff = (C_sim - C_ref).abs().max().item()
        return False, f"Output mismatch (FP32): max_abs_diff={max_diff} (rtol={rtol}, atol={atol})"
    return True, "Output matches reference (allclose, FP32)."


def run_gptvq_schedule_functional(
    schedule: list[GPTVQTileOp],
    tile_config: GPTVQTileConfig,
    codebook: torch.Tensor,
    indices: torch.Tensor,
    activation: torch.Tensor,
    dequant_unit: DequantizationUnit,
    systolic: SystolicArray,
    M: int,
    N: int,
    K: int,
    scales: torch.Tensor | None = None,
    zero_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute GPTVQ schedule with real data for correctness checking.

    Dequantizes weight tile-by-tile using codebook + indices (+ optional scaling),
    then computes matmul in same tile order as the cycle-accurate scheduler.
    """
    acc_dtype = systolic.acc_dtype.torch_dtype
    C_sim = torch.zeros((M, N), dtype=acc_dtype)
    d = dequant_unit.vector_dim

    for tile in schedule:
        k_group_start = (tile.k_idx * tile_config.tile_k) // d
        k_groups = math.ceil(tile.tile_k / d)
        row_start = tile.m_idx * tile_config.tile_m
        idx_tile = indices[
            row_start : row_start + tile.tile_m,
            k_group_start : k_group_start + k_groups,
        ]

        sc_tile = None
        zp_tile = None
        if dequant_unit.use_scaling and scales is not None:
            sc_tile = scales[k_group_start : k_group_start + k_groups]
            if zero_points is not None:
                zp_tile = zero_points[k_group_start : k_group_start + k_groups]

        dequant_result = dequant_unit.dequantize(
            idx_tile, codebook, tile.tile_m, tile.tile_k,
            scales=sc_tile, zero_points=zp_tile,
        )
        weight_tile = dequant_result.output

        k_start = tile.k_idx * tile_config.tile_k
        n_start = tile.n_idx * tile_config.tile_n
        act_tile = activation[
            k_start : k_start + tile.tile_k,
            n_start : n_start + tile.tile_n,
        ]

        result = systolic.compute_tile(weight_tile.float(), act_tile.float()).output

        out_row = tile.m_idx * tile_config.tile_m
        out_col = tile.n_idx * tile_config.tile_n
        if tile.accumulate:
            C_sim[out_row : out_row + tile.tile_m, out_col : out_col + tile.tile_n] += result
        else:
            C_sim[out_row : out_row + tile.tile_m, out_col : out_col + tile.tile_n] = result

    return C_sim


def format_diff_report(report: dict[str, Any]) -> str:
    """Format diff report as a readable multi-line string."""
    lines = [
        "--- Correctness / diff report ---",
        f"  Accumulator dtype: {report.get('acc_dtype', '?')}",
        f"  Shape: {report.get('shape', [])}  total elements: {report.get('num_elements', 0)}",
        f"  Match: {report.get('match', False)}",
        f"  Num mismatch: {report.get('num_mismatch', 0)}",
        f"  Max |C_sim - C_ref|: {report.get('max_abs_diff')}",
        f"  Mean |C_sim - C_ref|: {report.get('mean_abs_diff')}",
        f"  L1 error (sum of |diff|): {report.get('l1_error')}",
    ]
    sample = report.get("sample_mismatches", [])
    if sample:
        lines.append("  First mismatches (i, j, C_sim, C_ref, |diff|):")
        for s in sample[:10]:
            lines.append(f"    ({s['i']}, {s['j']}): sim={s['C_sim']} ref={s['C_ref']} |diff|={s['abs_diff']}")
    if report.get("error"):
        lines.append(f"  Error: {report['error']}")
    lines.append("---")
    return "\n".join(lines)
