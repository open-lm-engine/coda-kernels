import torch
from einops import rearrange
from quack.autotuner import autotune, AutotuneConfig
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm as quack_gemm
from quack.cross_entropy import cross_entropy_fwd_out
from quack.rms_final_reduce import _rms_final_reduce_out
from quack.epilogue.library import lse_epi, lse_target_epi, rstd_lse_epi
from quack.epilogue.rotary import rope_posfreq_epi, rstd_rope_posfreq_epi

from coda.core.gemm import epilogues
from coda.core.ops import misc_utils
from coda.core.ops.constants import AUTOTUNE_CACHE_RESULTS
from coda.core.gemm.gemm_interface import (
    _kernel_op,
    epilogue_launch,
    epilogue_autotune,
)


_DEVICE_CAPACITY = 9
assert get_device_capacity()[0] == _DEVICE_CAPACITY


@_kernel_op(
    name="coda::_gemm",
    mutates_args=("out",),
)
@autotune(
    configs=[
        AutotuneConfig(backend="quack"),
        AutotuneConfig(backend="cublas"),
    ],
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _gemm(
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


@_kernel_op(
    name="coda::_gemm_scalar_scale_epi",
    mutates_args=("D",),
)
@epilogue_autotune()
def _gemm_scalar_scale_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
    config: GemmConfig,
) -> None:
    epilogue_launch(
        epi_fn=epilogues.alpha_epi,
        A=A,
        B=B,
        D=D,
        epi_args={"alpha": alpha},
        config=config,
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
    _gemm_scalar_scale_epi(
        A=A,
        B=B.mT,
        D=out,
        alpha=alpha,
    )
    return out


@_kernel_op(
    name="coda::_gemm_swiglu_epi",
    mutates_args=("D", "postact"),
)
@epilogue_autotune(
    gated=True,
)
def _gemm_swiglu_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    postact: torch.Tensor,
    config: GemmConfig,
) -> None:
    epilogue_launch(
        epi_fn=epilogues.swiglu_preact_epi,
        A=A,
        B=B,
        D=D,
        epi_args={"postact": postact},
        config=config,
    )


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    preact: torch.Tensor | None = None,
    postact: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0, f"swiglu needs an even gate||up width, got N={N}"
    if preact is None:
        preact = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if postact is None:
        postact = torch.empty(M, N // 2, dtype=A.dtype, device=A.device)
    _gemm_swiglu_epi(
        A=A,
        B=B.mT,
        D=preact,
        postact=postact,
    )
    return preact, postact


@_kernel_op(
    name="coda::_gemm_rmsnorm_swiglu_epi",
    mutates_args=("D", "postact"),
)
@epilogue_autotune(
    gated=True,
)
def _gemm_rmsnorm_swiglu_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    rstd: torch.Tensor,
    postact: torch.Tensor,
    config: GemmConfig,
) -> None:
    epilogue_launch(
        epi_fn=epilogues.rstd_swiglu_scaled_preact_epi,
        A=A,
        B=B,
        D=D,
        epi_args={
            "rstd": rstd,
            "postact": postact,
        },
        config=config,
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
    _gemm_rmsnorm_swiglu_epi(
        A=A,
        B=B.mT,
        D=pre,
        rstd=rstd,
        postact=post,
    )
    return pre, post


@torch.compile(fullgraph=True, dynamic=False)
def _lse_reduce_compiled(lses: torch.Tensor, partials: torch.Tensor) -> None:
    torch.logsumexp(partials, dim=1, out=lses)


@_kernel_op(
    name="coda::_gemm_lse_epi",
    mutates_args=("logits", "lses"),
)
@epilogue_autotune()
def _gemm_lse_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    logits: torch.Tensor,
    lses: torch.Tensor,
    config: GemmConfig,
) -> None:
    M, N = logits.shape
    n_tiles = misc_utils.ceil_div(N, config.tile_n)
    partials = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epilogue_launch(
        epi_fn=lse_epi,
        A=A,
        B=B,
        D=logits,
        epi_args={"lse": partials},
        config=config,
    )
    _lse_reduce_compiled(
        lses=lses,
        partials=partials,
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
    _gemm_lse_epi(
        A=A,
        B=B.mT,
        logits=logits,
        lses=lses,
    )
    return logits, lses


@_kernel_op(
    name="coda::_gemm_rmsnorm_lse_epi",
    mutates_args=("logits", "lses"),
)
@epilogue_autotune()
def _gemm_rmsnorm_lse_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    logits: torch.Tensor,
    lses: torch.Tensor,
    config: GemmConfig,
) -> None:
    M, N = logits.shape
    n_tiles = misc_utils.ceil_div(N, config.tile_n)
    partials = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epilogue_launch(
        epi_fn=rstd_lse_epi,
        A=A,
        B=B,
        D=logits,
        epi_args={
            "rstd": rstd,
            "lse": partials,
        },
        config=config,
    )
    _lse_reduce_compiled(
        lses=lses,
        partials=partials,
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
    _gemm_rmsnorm_lse_epi(
        A=A,
        B=B.mT,
        rstd=rstd,
        logits=logits,
        lses=lses,
    )
    return logits, lses


@_kernel_op(
    name="coda::_lse_select_logits_epi",
    mutates_args=("lses", "losses", "target_logit"),
)
@epilogue_autotune()
def _lse_select_logits_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    lses: torch.Tensor | None,
    target: torch.Tensor,
    losses: torch.Tensor,
    target_logit: torch.Tensor,
    ignore_index: int,
    config: GemmConfig,
) -> None:
    M, _ = A.shape
    N, _ = B.shape
    n_tiles = misc_utils.ceil_div(N, config.tile_n)
    partials = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epilogue_launch(
        epi_fn=lse_target_epi,
        A=A,
        B=B,
        D=None,
        epi_args={
            "lse": partials,
            "target": target,
            "target_logit": target_logit,
        },
        config=config,
    )
    cross_entropy_fwd_out(
        x=partials,
        target=target,
        target_logit=target_logit,
        loss=losses,
        lse=lses,
        dx=None,
        weight=None,
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
    if losses is None:
        losses = torch.empty(M, dtype=torch.float32, device=A.device)
    if target_logits is None:
        target_logits = torch.empty(M, dtype=torch.float32, device=A.device)
    if return_lse:
        lses = torch.empty(M, dtype=torch.float32, device=A.device)
    else:
        lses = None
    _lse_select_logits_epi(
        A=A,
        B=B.mT,
        lses=lses,
        target=target,
        losses=losses,
        target_logit=target_logits,
        ignore_index=ignore_index,
    )
    return losses, lses, target_logits


@_kernel_op(
    name="coda::_gemm_rmsnorm_epi",
    mutates_args=("D",),
)
@epilogue_autotune()
def _gemm_rmsnorm_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    rstd: torch.Tensor,
    config: GemmConfig,
) -> None:
    epilogue_launch(
        epi_fn=epilogues.rstd_epi,
        A=A,
        B=B,
        D=D,
        epi_args={"rstd": rstd},
        config=config,
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
    _gemm_rmsnorm_epi(
        A=A,
        B=B.mT,
        D=out,
        rstd=rstd,
    )
    return out


@_kernel_op(
    name="coda::_gemm_residual_partial_rmsnorm_epi",
    mutates_args=("D", "rstd", "O"),
)
@epilogue_autotune()
def _gemm_residual_partial_rmsnorm_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    O: torch.Tensor,
    eps: float,
    config: GemmConfig,
) -> None:
    M, N = D.shape
    n_tiles = misc_utils.ceil_div(N, config.tile_n)
    partials = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epilogue_launch(
        epi_fn=epilogues.residual_sqsum_scaled_epi,
        A=A,
        B=B,
        D=D,
        C=C,
        epi_args={
            "weight": weight,
            "scaled_out": O,
            "sqsum": partials,
        },
        config=config,
    )
    _rms_final_reduce_out(
        x=partials,
        rstd=rstd,
        scale=1.0 / N,
        eps=eps,
    )


def gemm_residual_partial_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    pre: torch.Tensor | None = None,
    post: torch.Tensor | None = None,
    rstd: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    assert C.shape == (M, N)
    assert weight.shape == (N,)
    if pre is None:
        pre = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if post is None:
        post = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if rstd is None:
        rstd = torch.empty(M, dtype=torch.float32, device=A.device)
    _gemm_residual_partial_rmsnorm_epi(
        A=A,
        B=B.mT,
        D=pre,
        C=C,
        weight=rearrange(weight, "n -> 1 n"),
        rstd=rstd,
        O=post,
        eps=eps,
    )
    return pre, post, rstd


# @torch.compile(fullgraph=True, dynamic=False)
def _sum_reduce_compiled(partials: torch.Tensor, out: torch.Tensor, dim: int) -> None:
    assert out.dtype == partials.dtype
    torch.sum(partials, dim=dim, out=out)


@_kernel_op(
    name="coda::_gemm_residual_partial_rmsnorm_bwd_epi_store",
    mutates_args=("D", "dW", "C_out"),
)
@epilogue_autotune()
def _gemm_residual_partial_rmsnorm_bwd_epi_store(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    ZdZ: torch.Tensor,
    dW: torch.Tensor,
    C_out: torch.Tensor,
    alpha: torch.Tensor | None,
    config: GemmConfig,
) -> None:
    M, N = D.shape
    m_tiles = misc_utils.ceil_div(M, config.tile_m)
    # RowVecReduce requires N-contiguous partials
    partials = torch.empty(m_tiles, N, dtype=torch.float32, device=A.device)
    epi_args = {
        "rstd": rstd,
        "zdz": ZdZ,
        "weight": weight,
        "pre": C,
        "normed": C_out,
        "dweight": partials,
    }
    # an absent `alpha` must be absent from the epilogue signature for its term to compile out
    if alpha is not None:
        epi_fn = epilogues.alpha_residual_rmsnorm_bwd_epi
        epi_args["alpha"] = alpha
    else:
        epi_fn = epilogues.residual_rmsnorm_bwd_epi
    epilogue_launch(
        epi_fn=epi_fn,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        config=config,
        add_to_output=False,
    )
    _sum_reduce_compiled(
        partials=partials,
        out=dW,
        dim=0,
    )


@_kernel_op(
    name="coda::_gemm_residual_partial_rmsnorm_bwd_epi_accum",
    mutates_args=("D", "dW", "C_out"),
)
@epilogue_autotune()
def _gemm_residual_partial_rmsnorm_bwd_epi_accum(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    ZdZ: torch.Tensor,
    dW: torch.Tensor,
    C_out: torch.Tensor,
    alpha: torch.Tensor | None,
    config: GemmConfig,
) -> None:
    M, N = D.shape
    m_tiles = misc_utils.ceil_div(M, config.tile_m)
    # RowVecReduce requires N-contiguous partials
    partials = torch.empty(m_tiles, N, dtype=torch.float32, device=A.device)
    epi_args = {
        "rstd": rstd,
        "zdz": ZdZ,
        "weight": weight,
        "pre": C,
        "normed": C_out,
        "dweight": partials,
    }
    # an absent `alpha` must be absent from the epilogue signature for its term to compile out
    if alpha is not None:
        epi_fn = epilogues.alpha_residual_rmsnorm_bwd_epi
        epi_args["alpha"] = alpha
    else:
        epi_fn = epilogues.residual_rmsnorm_bwd_epi
    epilogue_launch(
        epi_fn=epi_fn,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        config=config,
        add_to_output=True,
    )
    _sum_reduce_compiled(
        partials=partials,
        out=dW,
        dim=0,
    )


def gemm_residual_partial_rmsnorm_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    weight: torch.Tensor,
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
    assert weight.shape == (N,)
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
    # B arrives (n, k), so no `.mT`
    if accumulate:
        _gemm_residual_partial_rmsnorm_bwd_epi_accum(
            A=A,
            B=B,
            D=dX,
            C=pre,
            weight=rearrange(weight, "n -> 1 n"),
            rstd=rstd,
            ZdZ=ZdZ,
            dW=dW,
            C_out=post,
            alpha=alpha,
        )
    else:
        _gemm_residual_partial_rmsnorm_bwd_epi_store(
            A=A,
            B=B,
            D=dX,
            C=pre,
            weight=rearrange(weight, "n -> 1 n"),
            rstd=rstd,
            ZdZ=ZdZ,
            dW=dW,
            C_out=post,
            alpha=alpha,
        )
    return dX, dW, post


@_kernel_op(
    name="coda::_gemm_swiglu_bwd_zdz_epi",
    mutates_args=("D", "ZdZ", "dZ"),
)
@epilogue_autotune()
def _gemm_swiglu_bwd_zdz_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    ZdZ: torch.Tensor,
    Z: torch.Tensor,
    dZ: torch.Tensor,
    config: GemmConfig,
) -> None:
    M, _ = A.shape
    # B: (N, K), D: (M, N), Z and dZ: (M, 2N)
    N, _ = B.shape
    n_tiles = misc_utils.ceil_div(N, config.tile_n)
    partials = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    # packed_cd_b16x2: dZ rides on D and the preact on C
    epilogue_launch(
        epi_fn=epilogues.dswiglu_preact_zdz_epi,
        A=A,
        B=B,
        D=dZ,
        C=Z,
        epi_args={
            "postact": D,
            "zdz": partials,
        },
        config=config,
    )
    _sum_reduce_compiled(
        partials=partials,
        out=ZdZ,
        dim=-1,
    )


@_kernel_op(
    name="coda::_gemm_swiglu_bwd_zdz_epi_scaled",
    mutates_args=("D", "ZdZ", "dZ"),
)
@epilogue_autotune()
def _gemm_swiglu_bwd_zdz_epi_scaled(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    ZdZ: torch.Tensor,
    Z: torch.Tensor,
    dZ: torch.Tensor,
    scale: float,
    config: GemmConfig,
) -> None:
    M, _ = A.shape
    # B: (N, K), D: (M, N), Z and dZ: (M, 2N)
    N, _ = B.shape
    n_tiles = misc_utils.ceil_div(N, config.tile_n)
    partials = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    # packed_cd_b16x2: dZ rides on D and the preact on C
    epilogue_launch(
        epi_fn=epilogues.dswiglu_preact_zdz_scaled_epi,
        A=A,
        B=B,
        D=dZ,
        C=Z,
        epi_args={
            "postact": D,
            "zdz": partials,
            "scale": scale,
        },
        config=config,
    )
    _sum_reduce_compiled(
        partials=partials,
        out=ZdZ,
        dim=-1,
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
    # B arrives (n, k), so no `.mT`
    if scale is None:
        _gemm_swiglu_bwd_zdz_epi(
            A=A,
            B=B,
            D=out,
            ZdZ=ZdZ,
            Z=Z,
            dZ=dZ,
        )
    else:
        _gemm_swiglu_bwd_zdz_epi_scaled(
            A=A,
            B=B,
            D=out,
            ZdZ=ZdZ,
            Z=Z,
            dZ=dZ,
            scale=scale,
        )
    return dZ, ZdZ, out


@_kernel_op(
    name="coda::_gemm_rope_epi",
    mutates_args=("D",),
)
@epilogue_autotune()
def _gemm_rope_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    config: GemmConfig,
) -> None:
    epilogue_launch(
        epi_fn=rope_posfreq_epi,
        A=A,
        B=B,
        D=D,
        epi_args={
            "pos": pos,
            "freq": freq,
        },
        config=config,
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
    _gemm_rope_epi(
        A=A,
        B=B.mT,
        D=out,
        pos=positions,
        freq=frequencies,
    )
    return out


@_kernel_op(
    name="coda::_gemm_rmsnorm_rope_epi",
    mutates_args=("D",),
)
@epilogue_autotune()
def _gemm_rmsnorm_rope_epi(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    rstd: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    config: GemmConfig,
) -> None:
    epilogue_launch(
        epi_fn=rstd_rope_posfreq_epi,
        A=A,
        B=B,
        D=D,
        epi_args={
            "rstd": rstd,
            "pos": pos,
            "freq": freq,
        },
        config=config,
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
    _gemm_rmsnorm_rope_epi(
        A=A,
        B=B.mT,
        D=out,
        rstd=rstd,
        pos=positions,
        freq=frequencies,
    )
    return out
