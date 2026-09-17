import torch
import dataclasses
from quack.gemm_config import GemmConfig

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


@_kernel_op(
    name="coda::_gemm_swiglu_fp8_epi",
    mutates_args=("D", "postact"),
)
@epilogue_autotune(
    gated=True,
    configs=FP8_GEMM_CONFIGS,
)
def _gemm_swiglu_fp8_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
    postact: torch.Tensor,
    config: GemmConfig,
) -> None:
    epilogue_launch(
        epi_fn=epilogues.alpha_swiglu_preact_epi,
        A=A,
        B=B,
        D=D,
        epi_args={
            "alpha": alpha,
            "postact": postact,
        },
        config=config,
        fp8_fast_accum=True,
    )


def gemm_swiglu_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: torch.Tensor,
    preact: torch.Tensor | None = None,
    postact: torch.Tensor | None = None,
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
    if preact is None:
        preact = torch.empty(M, N, dtype=output_dtype, device=A.device)
    if postact is None:
        postact = torch.empty(M, N // 2, dtype=output_dtype, device=A.device)
    _gemm_swiglu_fp8_epi(
        A=A,
        B=B.mT,
        D=preact,
        alpha=scale,
        postact=postact,
    )
    return preact, postact


@_kernel_op(
    name="coda::_gemm_rope_fp8_epi",
    mutates_args=("D",),
)
@epilogue_autotune(
    configs=FP8_GEMM_CONFIGS,
)
def _gemm_rope_fp8_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    config: GemmConfig,
) -> None:
    epilogue_launch(
        epi_fn=epilogues.alpha_rope_posfreq_epi,
        A=A,
        B=B,
        D=D,
        epi_args={
            "alpha": alpha,
            "pos": pos,
            "freq": freq,
        },
        config=config,
        fp8_fast_accum=True,
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
    _gemm_rope_fp8_epi(
        A=A,
        B=B.mT,
        D=out,
        alpha=scale,
        pos=positions,
        freq=frequencies,
    )
    return out
