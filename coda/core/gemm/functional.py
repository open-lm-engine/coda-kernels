import torch
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm as quack_gemm
from quack.cross_entropy import cross_entropy_fwd_out
from quack.rms_final_reduce import _rms_final_reduce_out
from quack.autotuner import autotune, AutotuneConfig
from quack.cute_dsl_utils import get_device_capacity

from coda.core.ops import misc_utils
from coda.core.ops.constants import AUTOTUNE_CACHE_RESULTS
from coda.core.epilogue.utils import (
    preprocess_epi_args,
    make_epi_keys,
)
from coda.core.gemm.gemm_interface import (
    _kernel_op,
    _gemm_epilogue_tuned,
    _preprocess_gemm_operands,
    prune_gemm_configs,
    GEMM_CONFIGS,
)
from coda.core.gemm.registry import (
    GemmScale,
    GemmScalarScale,
    GemmScalarScaleResidualRMSNormBwd,
    GemmSwiGLU,
    GemmScaleSwiGLU,
    GemmSwiGLUBwdZdZ,
    GemmRoPE,
    GemmScaleRoPE,
    GemmLSE,
    GemmScaleLSE,
    GemmLSESelectLogits,
    GemmQKVSqSum,
    GemmResidualSqSumScaledAux,
)

_DEVICE_CAPACITY = 9
assert get_device_capacity()[0] == _DEVICE_CAPACITY


@autotune(
    configs=[
        AutotuneConfig(backend="quack"),
        AutotuneConfig(backend="cublas"),
    ],
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    backend: str,
) -> None:
    if backend == "quack":
        # setting `split_k=None` so the autotuner adds split-K candidates
        # only for occupancy-starved shapes (fewer tiles than SMs)
        quack_gemm(A=A, B=B, out=out, tuned=True, split_k=None)
    else:
        torch.matmul(A, B, out=out)


@_kernel_op("coda::_gemm", mutates_args=("out",))
def _gemm(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    _gemm_tuned(A=A, B=B, out=out)


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    _gemm(A=A, B=B, out=out)
    return out


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_scalar_scale_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "alpha": alpha,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmScalarScale,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmScalarScale, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_scalar_scale", mutates_args=("D",))
def _gemm_scalar_scale(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
) -> None:
    _gemm_scalar_scale_tuned(
        A=A,
        B=B,
        D=D,
        alpha=alpha,
    )


def gemm_scalar_scale(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=out,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmScalarScale,
        epi_args={
            "alpha": alpha,
        },
    )
    _gemm_scalar_scale(
        A=A,
        B=B,
        D=D,
        alpha=epi_args["alpha"],
    )
    return out


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_swiglu_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    post_act: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "mAuxOut": post_act,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmSwiGLU,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmSwiGLU, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_swiglu", mutates_args=("D", "post_act"))
def _gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    post_act: torch.Tensor,
) -> None:
    _gemm_swiglu_tuned(
        A=A,
        B=B,
        D=D,
        post_act=post_act,
    )


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    pre_act: torch.Tensor | None = None,
    post_act: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0, f"swiglu needs an even gate||up width, got N={N}"
    if pre_act is None:
        pre_act = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if post_act is None:
        post_act = torch.empty(M, N // 2, dtype=A.dtype, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=pre_act,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmSwiGLU,
        epi_args={
            "mAuxOut": post_act,
        },
    )
    _gemm_swiglu(
        A=A,
        B=B,
        D=D,
        post_act=epi_args["mAuxOut"],
    )
    return pre_act, post_act


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_rmsnorm_swiglu_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    R: torch.Tensor,
    O: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "mColVecBroadcast": R,
        "mAuxOut": O,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmScaleSwiGLU,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmScaleSwiGLU, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_rmsnorm_swiglu", mutates_args=("D", "O"))
def _gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    R: torch.Tensor,
    O: torch.Tensor,
) -> None:
    _gemm_rmsnorm_swiglu_tuned(
        A=A,
        B=B,
        D=D,
        R=R,
        O=O,
    )


def gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    pre: torch.Tensor | None = None,
    post: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0
    assert rstd.shape == (M,)
    assert rstd.dtype == torch.float32
    if pre is None:
        pre = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if post is None:
        post = torch.empty(M, N // 2, dtype=A.dtype, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=pre,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmScaleSwiGLU,
        epi_args={
            "mColVecBroadcast": rstd,
            "mAuxOut": post,
        },
    )
    _gemm_rmsnorm_swiglu(
        A=A,
        B=B,
        D=D,
        R=epi_args["mColVecBroadcast"],
        O=epi_args["mAuxOut"],
    )
    return pre, post


# a head spans at most ceil(head_dim / tile_n) + 1 tiles; size ssq for the narrowest sm90 tile
_SQSUM_MIN_TILE_N = min(
    c.tile_n
    for c in GEMM_CONFIGS
    if c.device_capacity == _DEVICE_CAPACITY
)


def _sqsum_num_segments(head_dim: int) -> int:
    return misc_utils.ceil_div(head_dim, _SQSUM_MIN_TILE_N) + 1


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    key=["head_dim", "num_segments"],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_qkv_sqsum_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    ssq: torch.Tensor,
    head_dim: int,
    num_segments: int,
    config: GemmConfig,
) -> None:
    epi_args = {
        "mSqSumVec": ssq,
        "head_dim": head_dim,
        "num_segments": num_segments,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmQKVSqSum,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmQKVSqSum, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_qkv_sqsum", mutates_args=("D", "ssq"))
def _gemm_qkv_sqsum(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    ssq: torch.Tensor,
    head_dim: int,
    num_segments: int,
) -> None:
    _gemm_qkv_sqsum_tuned(
        A=A,
        B=B,
        D=D,
        ssq=ssq,
        head_dim=head_dim,
        num_segments=num_segments,
    )


def gemm_qkv_sqsum(
    A: torch.Tensor,
    B: torch.Tensor,
    head_dim: int,
    num_segments: int,
    out: torch.Tensor | None = None,
    ssq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if ssq is None:
        # zero-init as heads whose segments a tile never writes must read 0
        ssq = torch.zeros(M, (N // head_dim) * num_segments, dtype=torch.float32, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=out,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmQKVSqSum,
        epi_args={
            "mSqSumVec": ssq,
            "head_dim": head_dim,
            "num_segments": num_segments,
        },
    )
    _gemm_qkv_sqsum(
        A=A,
        B=B,
        D=D,
        ssq=epi_args["mSqSumVec"],
        head_dim=head_dim,
        num_segments=num_segments,
    )
    return out, ssq


@torch.compile(fullgraph=True, dynamic=False)
def _lse_reduce_compiled(lses: torch.Tensor, lse_partial: torch.Tensor) -> None:
    torch.logsumexp(lse_partial, dim=1, out=lses)


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    key=["vocab_size"],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_lse_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    lses: torch.Tensor,
    vocab_size: int,
    config: GemmConfig,
) -> None:
    M, _, _ = A.shape
    n_tiles = misc_utils.ceil_div(vocab_size, config.tile_n)
    lse_partial = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epi_args = preprocess_epi_args(
        GemmCls=GemmLSE,
        epi_args={
            "mLSEVec": lse_partial,
            "vocab_size": vocab_size,
        },
    )
    _gemm_epilogue_tuned(
        GemmCls=GemmLSE,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmLSE, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )
    _lse_reduce_compiled(
        lses=lses,
        lse_partial=lse_partial,
    )


@_kernel_op("coda::_gemm_lse", mutates_args=("D", "lses"))
def _gemm_lse(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    lses: torch.Tensor,
    vocab_size: int,
) -> None:
    _gemm_lse_tuned(
        A=A,
        B=B,
        D=D,
        lses=lses,
        vocab_size=vocab_size,
    )


def gemm_lse(
    A: torch.Tensor,
    B: torch.Tensor,
    logits: torch.Tensor | None = None,
    lses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, vocab_size = B.shape
    if logits is None:
        logits = torch.empty(M, vocab_size, dtype=A.dtype, device=A.device)
    if lses is None:
        lses = torch.empty(M, dtype=torch.float32, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=logits,
        C=None,
    )
    _gemm_lse(
        A=A,
        B=B,
        D=D,
        lses=lses,
        vocab_size=vocab_size,
    )
    return logits, lses


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    key=["vocab_size"],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_rmsnorm_lse_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    R: torch.Tensor,
    lses: torch.Tensor,
    vocab_size: int,
    config: GemmConfig,
) -> None:
    M, _, _ = A.shape
    n_tiles = misc_utils.ceil_div(vocab_size, config.tile_n)
    lse_partial = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epi_args = preprocess_epi_args(
        GemmCls=GemmScaleLSE,
        epi_args={
            "mColVecBroadcast": R,
            "mLSEVec": lse_partial,
            "vocab_size": vocab_size,
        },
    )
    _gemm_epilogue_tuned(
        GemmCls=GemmScaleLSE,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmScaleLSE, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )
    _lse_reduce_compiled(
        lses=lses,
        lse_partial=lse_partial,
    )


@_kernel_op("coda::_gemm_rmsnorm_lse", mutates_args=("D", "lses"))
def _gemm_rmsnorm_lse(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    R: torch.Tensor,
    lses: torch.Tensor,
    vocab_size: int,
) -> None:
    _gemm_rmsnorm_lse_tuned(
        A=A,
        B=B,
        D=D,
        R=R,
        lses=lses,
        vocab_size=vocab_size,
    )


def gemm_rmsnorm_lse(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    logits: torch.Tensor | None = None,
    lses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, vocab_size = B.shape
    assert rstd.shape == (M,)
    assert rstd.dtype == torch.float32
    if logits is None:
        logits = torch.empty(M, vocab_size, dtype=A.dtype, device=A.device)
    if lses is None:
        lses = torch.empty(M, dtype=torch.float32, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=logits,
        C=None,
    )
    _gemm_rmsnorm_lse(
        A=A,
        B=B,
        D=D,
        R=rstd,
        lses=lses,
        vocab_size=vocab_size,
    )
    return logits, lses


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    key=["vocab_size", "ignore_index"],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_lse_select_logits_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    lses: torch.Tensor | None,
    target: torch.Tensor,
    losses: torch.Tensor,
    target_logits: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
    config: GemmConfig,
) -> None:
    M, _, _ = A.shape
    n_tiles = misc_utils.ceil_div(vocab_size, config.tile_n)
    lse_partial = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epi_args = preprocess_epi_args(
        GemmCls=GemmLSESelectLogits,
        epi_args={
            "mLSEVec": lse_partial,
            "mTarget": target,
            "mLogits": target_logits,
            "vocab_size": vocab_size,
        },
    )
    _gemm_epilogue_tuned(
        GemmCls=GemmLSESelectLogits,
        A=A,
        B=B,
        D=None,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmLSESelectLogits, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )
    cross_entropy_fwd_out(
        x=lse_partial,
        target=target,
        target_logit=target_logits,
        loss=losses,
        lse=lses,
        dx=None,
        weight=None,
        ignore_index=ignore_index,
    )


@_kernel_op("coda::_gemm_lse_select_logits", mutates_args=("lses", "losses", "target_logits"))
def _gemm_lse_select_logits(
    A: torch.Tensor,
    B: torch.Tensor,
    lses: torch.Tensor | None,
    target: torch.Tensor,
    losses: torch.Tensor,
    target_logits: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
) -> None:
    _gemm_lse_select_logits_tuned(
        A=A,
        B=B,
        lses=lses,
        target=target,
        losses=losses,
        target_logits=target_logits,
        vocab_size=vocab_size,
        ignore_index=ignore_index,
    )


def gemm_lse_select_logits(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_lse: bool,
    losses: torch.Tensor | None = None,
    target_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    assert target.dtype == torch.int32
    M, _ = A.shape
    _, vocab_size = B.shape
    if losses is None:
        losses = torch.empty(M, dtype=torch.float32, device=A.device)
    if target_logits is None:
        target_logits = torch.empty(M, dtype=A.dtype, device=A.device)
    if return_lse:
        lses = torch.empty(M, dtype=torch.float32, device=A.device)
    else:
        lses = None
    A, B, _, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=None,
        C=None,
    )
    _gemm_lse_select_logits(
        A=A,
        B=B,
        lses=lses,
        target=target,
        losses=losses,
        target_logits=target_logits,
        vocab_size=vocab_size,
        ignore_index=ignore_index,
    )
    return losses, lses, target_logits


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_rmsnorm_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    R: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "mColVecBroadcast": R,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmScale,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmScale, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_rmsnorm", mutates_args=("D",))
def _gemm_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    R: torch.Tensor,
) -> None:
    _gemm_rmsnorm_tuned(
        A=A,
        B=B,
        D=D,
        R=R,
    )


def gemm_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    assert rstd.shape == (M,)
    assert rstd.dtype == torch.float32
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=out,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmScale,
        epi_args={
            "mColVecBroadcast": rstd,
        },
    )
    _gemm_rmsnorm(
        A=A,
        B=B,
        D=D,
        R=epi_args["mColVecBroadcast"],
    )
    return out


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_residual_partial_rmsnorm_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    O: torch.Tensor,
    eps: float,
    config: GemmConfig,
) -> None:
    M, N, _ = D.shape
    n_tiles = misc_utils.ceil_div(N, config.tile_n)
    partials = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epi_args = preprocess_epi_args(
        GemmCls=GemmResidualSqSumScaledAux,
        epi_args={
            "mSqSumVec": partials,
            "mRowVecScale": W,
            "mAuxOut": O,
            "mResidual": C,
        },
    )
    _gemm_epilogue_tuned(
        GemmCls=GemmResidualSqSumScaledAux,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmResidualSqSumScaledAux, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )
    _rms_final_reduce_out(
        x=partials,
        rstd=R,
        scale=1.0 / N,
        eps=eps,
    )


@_kernel_op("coda::_gemm_residual_partial_rmsnorm", mutates_args=("D", "R", "O"))
def _gemm_residual_partial_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    O: torch.Tensor,
    eps: float,
) -> None:
    _gemm_residual_partial_rmsnorm_tuned(
        A=A,
        B=B,
        D=D,
        C=C,
        W=W,
        R=R,
        O=O,
        eps=eps,
    )


def gemm_residual_partial_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    eps: float,
    pre: torch.Tensor | None = None,
    post: torch.Tensor | None = None,
    rstd: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    assert C.shape == (M, N)
    assert W.shape == (N,)
    if pre is None:
        pre = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if post is None:
        post = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if rstd is None:
        rstd = torch.empty(M, dtype=torch.float32, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=pre,
        C=None,
    )
    _gemm_residual_partial_rmsnorm(
        A=A,
        B=B,
        D=D,
        C=C,
        W=W,
        R=rstd,
        O=post,
        eps=eps,
    )
    return pre, post, rstd


# @torch.compile(fullgraph=True, dynamic=False)
def _sum_reduce_compiled(partials: torch.Tensor, out: torch.Tensor, dim: int) -> None:
    assert out.dtype == partials.dtype
    torch.sum(partials, dim=dim, out=out)


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_residual_partial_rmsnorm_bwd_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    dW: torch.Tensor,
    ZdZ: torch.Tensor,
    C_out: torch.Tensor,
    alpha: torch.Tensor | None,
    accumulate: bool,
    config: GemmConfig,
) -> None:
    M, N, _ = D.shape
    m_tiles = misc_utils.ceil_div(M, config.tile_m)
    # Tile-major (N, m_tiles) so the finishing reduction runs down contiguous memory.
    # The epilogue gets the .mT view and is unchanged.
    partials_T = torch.empty(N, m_tiles, dtype=torch.float32, device=A.device)
    if alpha is not None:
        extra_epi_args = {"alpha": alpha}
    else:
        # for `ScalarScale` we need to omit `alpha` when it's None
        # due to artifacts of our epi-lowering
        extra_epi_args = {}
    epi_args = preprocess_epi_args(
        GemmCls=GemmScalarScaleResidualRMSNormBwd,
        epi_args={
            "mColVecR": R,
            "mColVecZdZ": ZdZ,
            "mRowVecW": W,
            "mMatrixC": C,
            "mAuxOut": C_out,
            "mDWVec": partials_T.mT,
            **extra_epi_args,
        },
    )
    _gemm_epilogue_tuned(
        GemmCls=GemmScalarScaleResidualRMSNormBwd,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmScalarScaleResidualRMSNormBwd, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=accumulate,
        config=config,
    )
    _sum_reduce_compiled(
        partials=partials_T,
        out=dW,
        dim=-1,
    )


@_kernel_op("coda::_gemm_residual_partial_rmsnorm_bwd", mutates_args=("D", "dW", "C_out"))
def _gemm_residual_partial_rmsnorm_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    dW: torch.Tensor,
    ZdZ: torch.Tensor,
    C_out: torch.Tensor,
    alpha: torch.Tensor | None,
    accumulate: bool,
) -> None:
    _gemm_residual_partial_rmsnorm_bwd_tuned(
        A=A,
        B=B,
        D=D,
        C=C,
        W=W,
        R=R,
        dW=dW,
        ZdZ=ZdZ,
        C_out=C_out,
        alpha=alpha,
        accumulate=accumulate,
    )


def gemm_residual_partial_rmsnorm_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    W: torch.Tensor,
    pre: torch.Tensor,
    ZdZ: torch.Tensor,
    rstd: torch.Tensor,
    alpha: torch.Tensor | None = None,
    dX: torch.Tensor | None = None,
    dW: torch.Tensor | None = None,
    post: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    N, _ = B.shape
    assert W.shape == (N,)
    assert pre.shape == (M, N)
    assert ZdZ.shape == (M,)
    assert ZdZ.dtype == torch.float32
    assert rstd.shape == (M,)
    assert rstd.dtype == torch.float32
    if dX is None:
        accumulate = False
        dX = torch.empty(M, N, dtype=A.dtype, device=A.device)
    else:
        accumulate = True
        assert dX.shape == (M, N)
    if dW is None:
        dW = torch.empty(N, dtype=torch.float32, device=A.device)
    if post is None:
        post = torch.empty(M, N, dtype=A.dtype, device=A.device)

    # Transpose B for GEMM (we want A @ B.T)
    B_T = B.mT

    A, B_T, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B_T,
        D=dX,
        C=None,
    )
    _gemm_residual_partial_rmsnorm_bwd(
        A=A,
        B=B_T,
        D=D,
        C=pre,
        W=W,
        R=rstd,
        dW=dW,
        ZdZ=ZdZ,
        C_out=post,
        alpha=alpha,
        accumulate=accumulate,
    )
    return dX, dW, post


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_swiglu_bwd_zdz_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    ZdZ: torch.Tensor,
    Z_packed: torch.Tensor,
    dZ_packed: torch.Tensor,
    scale: float | None,
    config: GemmConfig,
) -> None:
    M, N, _ = D.shape
    n_tiles = misc_utils.ceil_div(N, config.tile_n)
    partials = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epi_args = preprocess_epi_args(
        GemmCls=GemmSwiGLUBwdZdZ,
        epi_args={
            "mZPacked": Z_packed,
            "mAuxOut": dZ_packed,
            "mZdZVec": partials,
            "scale": scale,
        },
    )
    _gemm_epilogue_tuned(
        GemmCls=GemmSwiGLUBwdZdZ,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmSwiGLUBwdZdZ, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )
    _sum_reduce_compiled(
        partials=partials,
        out=ZdZ,
        dim=-1,
    )


@_kernel_op("coda::_gemm_swiglu_bwd_zdz", mutates_args=("D", "ZdZ", "dZ_packed"))
def _gemm_swiglu_bwd_zdz(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    ZdZ: torch.Tensor,
    Z_packed: torch.Tensor,
    dZ_packed: torch.Tensor,
    scale: float | None,
) -> None:
    _gemm_swiglu_bwd_zdz_tuned(
        A=A,
        B=B,
        D=D,
        ZdZ=ZdZ,
        Z_packed=Z_packed,
        dZ_packed=dZ_packed,
        scale=scale,
    )


def gemm_swiglu_bwd_zdz(
    A: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    scale: float | None = None,
    dZ: torch.Tensor | None = None,
    ZdZ: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    N, _ = B.shape
    assert Z.shape == (M, 2 * N)
    assert Z.dtype == A.dtype
    if dZ is None:
        dZ = torch.empty(M, 2 * N, dtype=A.dtype, device=A.device)
    if ZdZ is None:
        ZdZ = torch.empty(M, dtype=torch.float32, device=A.device)
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)

    # Transpose B for GEMM (we want A @ B.T)
    B_T = B.mT

    # Z and dZ travel as (M, N) 32-bit containers: two model-dtype halves per element
    Z_packed = Z.view(dtype=torch.int32)
    dZ_packed = dZ.view(dtype=torch.int32)

    A, B_T, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B_T,
        D=out,
        C=None,
    )
    _gemm_swiglu_bwd_zdz(
        A=A,
        B=B_T,
        D=D,
        ZdZ=ZdZ,
        Z_packed=Z_packed,
        dZ_packed=dZ_packed,
        scale=scale,
    )
    return dZ, ZdZ, out


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_rope_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "mPos": pos,
        "mFreq": freq,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmRoPE,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmRoPE, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_rope", mutates_args=("D",))
def _gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
) -> None:
    _gemm_rope_tuned(
        A=A,
        B=B,
        D=D,
        pos=pos,
        freq=freq,
    )


def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0
    assert positions.shape == (M,)
    assert positions.dtype in (torch.float32, torch.int32)
    assert frequencies.shape == (N,)
    assert frequencies.dtype == torch.float32
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=out,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmRoPE,
        epi_args={
            "mPos": positions,
            "mFreq": frequencies,
        },
    )
    _gemm_rope(
        A=A,
        B=B,
        D=D,
        pos=epi_args["mPos"],
        freq=epi_args["mFreq"],
    )
    return out


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm_rmsnorm_rope_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    R: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "mColVecBroadcast": R,
        "mPos": pos,
        "mFreq": freq,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmScaleRoPE,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmScaleRoPE, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        fp8_fast_accum=False,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_rmsnorm_rope", mutates_args=("D",))
def _gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    R: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
) -> None:
    _gemm_rmsnorm_rope_tuned(
        A=A,
        B=B,
        D=D,
        R=R,
        pos=pos,
        freq=freq,
    )


def gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0
    assert rstd.shape == (M,)
    assert rstd.dtype == torch.float32
    assert positions.shape == (M,)
    assert positions.dtype in (torch.float32, torch.int32)
    assert frequencies.shape == (N,)
    assert frequencies.dtype == torch.float32
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    A, B, D, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=out,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmScaleRoPE,
        epi_args={
            "mColVecBroadcast": rstd,
            "mPos": positions,
            "mFreq": frequencies,
        },
    )
    _gemm_rmsnorm_rope(
        A=A,
        B=B,
        D=D,
        R=epi_args["mColVecBroadcast"],
        pos=epi_args["mPos"],
        freq=epi_args["mFreq"],
    )
    return out
