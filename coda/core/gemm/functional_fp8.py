import torch
import dataclasses
from quack.gemm_config import GemmConfig
from quack.rms_final_reduce import _rms_final_reduce_out

from coda.core.gemm import epilogues
from coda.core.ops import misc_utils
from coda.core.gemm.gemm_interface import (
    _kernel_op,
    _extend_configs,
    epilogue_launch,
    epilogue_autotune,
    GEMM_CONFIGS,
)


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=False,
)
def _gemm_swiglu_fp8_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
    post_act: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "alpha": alpha,
        "mAuxOut": post_act,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmScalarScaleSwiGLU,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmScalarScaleSwiGLU, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_swiglu_fp8", mutates_args=("D", "post_act"))
def _gemm_swiglu_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
    post_act: torch.Tensor,
) -> None:
    _gemm_swiglu_fp8_tuned(
        A=A,
        B=B,
        D=D,
        alpha=alpha,
        post_act=post_act,
    )


def gemm_swiglu_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: torch.Tensor,
    pre_act: torch.Tensor | None = None,
    post_act: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0
    assert A.dtype in _FP8_DTYPES
    assert B.dtype in _FP8_DTYPES
    assert scale.numel() == 1
    assert scale.dtype == torch.float32
    if output_dtype is None:
        output_dtype = torch.bfloat16
    if pre_act is None:
        pre_act = torch.empty(M, N, dtype=output_dtype, device=A.device)
    if post_act is None:
        post_act = torch.empty(M, N // 2, dtype=output_dtype, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=pre_act,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmScalarScaleSwiGLU,
        epi_args={
            "alpha": scale,
            "mAuxOut": post_act,
        },
    )
    _gemm_swiglu_fp8(
        A=A,
        B=B,
        D=D,
        alpha=epi_args["alpha"],
        post_act=epi_args["mAuxOut"],
    )
    return pre_act, post_act


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=False,
)
def _gemm_rope_fp8_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "alpha": alpha,
        "mPos": pos,
        "mFreq": freq,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmScalarScaleRoPE,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmScalarScaleRoPE, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_rope_fp8", mutates_args=("D",))
def _gemm_rope_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
) -> None:
    _gemm_rope_fp8_tuned(
        A=A,
        B=B,
        D=D,
        alpha=alpha,
        pos=pos,
        freq=freq,
    )


def gemm_rope_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    out: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0
    assert A.dtype in _FP8_DTYPES
    assert B.dtype in _FP8_DTYPES
    assert positions.shape == (M,)
    assert positions.dtype in (torch.float32, torch.int32)
    assert frequencies.shape == (N,)
    assert frequencies.dtype == torch.float32
    if output_dtype is None:
        output_dtype = torch.bfloat16
    if out is None:
        out = torch.empty(M, N, dtype=output_dtype, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=out,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmScalarScaleRoPE,
        epi_args={
            "alpha": scale,
            "mPos": positions,
            "mFreq": frequencies,
        },
    )
    _gemm_rope_fp8(
        A=A,
        B=B,
        D=D,
        alpha=epi_args["alpha"],
        pos=epi_args["mPos"],
        freq=epi_args["mFreq"],
    )
    return out
