import torch
import dataclasses
import cutlass
import cutlass.cute as cute

from einops import rearrange
from quack.activation import dswiglu
from quack.autotuner import autotune, AutotuneConfig
from quack.tile_scheduler import RasterOrder

from coda.core.ops.constants import AUTOTUNE_CACHE_RESULTS, NUM_BITS_PER_COPY
from coda.core.ops.misc_utils import static_assert, ceil_div
from coda.core.gemm.gemm_interface import _kernel_op
from coda.core.elementwise.rope import qknorm_rope_bwd_
from coda.core.elementwise.zdz import rope_bwd_zdz_
from coda.core.elementwise.short_conv import short_conv_fwd_, short_conv_bwd_
from coda.core.elementwise.cross_entropy import cross_entropy_fwd_bwd_
from coda.core.elementwise.templates import ElementwiseConfig, _elementwise_op_tuned


_ELEMENTWISE_CONFIGS = tuple(
    ElementwiseConfig(
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    for thr_m, thr_n, val_m in (
        (4, 32, 4),
        (8, 64, 4),
        (16, 16, 4),
        (16, 16, 8),
        (1, 128, 4),
        (8, 128, 4),
        (4, 256, 4),
        (2, 512, 4),
    )
)

_CE_ELEMENTWISE_CONFIGS = tuple(
    ElementwiseConfig(
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    for thr_m, thr_n, val_m in (
        (1, 1024, 2),
        (1, 512, 2),
        (4, 128, 1),
    )
)

_ZDZ_CONFIGS = tuple(
    ElementwiseConfig(
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    for thr_m, thr_n, val_m in (
        (4, 32, 1),
        (8, 32, 1),
        (4, 32, 2),
        (8, 32, 2),
        (4, 32, 4),
        (2, 64, 2),
        (4, 64, 2),
        (1, 128, 2),
        (2, 128, 2),
        (1, 256, 2),
        # a CTA that covers the whole row is fastest
        *((1, thr_n, 1) for thr_n in range(128, 1024 + 1, 32)),
    )
)


@dataclasses.dataclass(frozen=True)
class ShortConvConfig(object):
    thr_m: int
    thr_n: int
    val_m: int
    num_bits_per_copy: int
    raster_order: RasterOrder


# both pools come from a wide sweep: each stays within 1% of the best config on every measured shape
_SHORT_CONV_FWD_CONFIGS = tuple(
    ShortConvConfig(
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        num_bits_per_copy=num_bits_per_copy,
        raster_order=raster_order,
    )
    for thr_m, thr_n, val_m, num_bits_per_copy, raster_order in (
        (2, 32, 16, 64, RasterOrder.AlongN),
        (2, 32, 8, 64, RasterOrder.AlongN),
        (4, 32, 8, 64, RasterOrder.AlongN),
        (4, 32, 4, 64, RasterOrder.AlongN),
        (8, 32, 8, 64, RasterOrder.AlongN),
        (4, 16, 8, 64, RasterOrder.AlongN),
        (4, 16, 8, 128, RasterOrder.AlongN),
        (8, 16, 8, 128, RasterOrder.AlongN),
        (8, 16, 8, 128, RasterOrder.AlongM),
        (8, 8, 8, 128, RasterOrder.AlongN),
    )
)


_SHORT_CONV_BWD_CONFIGS = tuple(
    ShortConvConfig(
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        num_bits_per_copy=num_bits_per_copy,
        raster_order=raster_order,
    )
    for thr_m, thr_n, val_m, num_bits_per_copy, raster_order in (
        (16, 8, 16, 128, RasterOrder.AlongN),
        (16, 16, 16, 128, RasterOrder.AlongN),
        (8, 16, 16, 64, RasterOrder.AlongN),
        (16, 16, 8, 64, RasterOrder.AlongN),
        (8, 16, 8, 64, RasterOrder.AlongN),
        (8, 16, 8, 64, RasterOrder.AlongM),
        (4, 32, 16, 64, RasterOrder.AlongN),
        (8, 32, 8, 64, RasterOrder.AlongN),
        (4, 32, 8, 64, RasterOrder.AlongN),
        (16, 8, 8, 128, RasterOrder.AlongN),
    )
)


def _sum_reduce(partials: torch.Tensor, out: torch.Tensor, dim: int | tuple[int, ...]) -> None:
    assert out.dtype == partials.dtype
    torch.sum(partials, dim=dim, out=out)


@torch.compile(fullgraph=True, dynamic=False)
def _sum_reduce_compiled(partials: torch.Tensor, out: torch.Tensor, dim: int | tuple[int, ...]) -> None:
    _sum_reduce(partials=partials, out=out, dim=dim)


def _prune_rope_configs(configs: list[AutotuneConfig], named_args: dict, **kwargs) -> list[AutotuneConfig]:
    kwargs = named_args | kwargs
    x = kwargs["x"]
    assert x.ndim == 2
    packed_cols = x.shape[1] // 2
    dtype_width = x.element_size() * 8
    vector_size = NUM_BITS_PER_COPY // (2 * dtype_width)
    configs_pruned = [
        c for c in configs
        if packed_cols % (c.kwargs["config"].thr_n * vector_size) == 0
    ]

    dq = kwargs["dq"]
    assert dq.ndim == 2
    packed_cols_dq = dq.shape[1] // 2
    configs_pruned = [
        c for c in configs_pruned
        if packed_cols_dq % (c.kwargs["config"].thr_n * vector_size) == 0
    ]

    return configs_pruned


def _prune_rope_bwd_zdz_configs(configs: list[AutotuneConfig], named_args: dict, **kwargs) -> list[AutotuneConfig]:
    kwargs = named_args | kwargs
    y = kwargs["y"]
    assert y.ndim == 2
    packed_cols = y.shape[1] // 2
    dtype_width = y.element_size() * 8
    vector_size = NUM_BITS_PER_COPY // (2 * dtype_width)
    return [
        c for c in configs
        if packed_cols % (c.kwargs["config"].thr_n * vector_size) == 0
    ]


@cute.jit
def _dswiglu_op(tX: cute.Tensor, tY: cute.Tensor, tZ: cute.Tensor) -> None:
    static_assert(tX.dtype == cute.Int32)
    static_assert(tZ.dtype == cute.Int32)
    static_assert(tY.dtype in (cute.Float16, cute.BFloat16))
    dtype = tY.dtype
    tX_pair = cute.recast_tensor(tX, dtype=dtype)
    tZ_pair = cute.recast_tensor(tZ, dtype=dtype)
    for i in cutlass.range_constexpr(cute.size(tY)):
        g = tX_pair[2 * i].to(dtype=cutlass.Float32)
        u = tX_pair[2 * i + 1].to(dtype=cutlass.Float32)
        dout = tY[i].to(dtype=cutlass.Float32)
        dg, du, _ = dswiglu(x=g, y=u, dout=dout)
        tZ_pair[2 * i] = dg.to(dtype=dtype)
        tZ_pair[2 * i + 1] = du.to(dtype=dtype)


@_kernel_op("coda::_dswiglu_backward", mutates_args=("Z",))
def _dswiglu_backward(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor) -> None:
    return _elementwise_op_tuned(op=_dswiglu_op, X=X, Y=Y, Z=Z)


def dswiglu_backward(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
    grad_pre: torch.Tensor | None = None,
) -> torch.Tensor:
    assert preact.dtype in (torch.bfloat16, torch.float16)
    assert grad_out.dtype == preact.dtype
    assert preact.is_contiguous()
    assert grad_out.is_contiguous()
    if grad_pre is None:
        grad_pre = torch.empty_like(preact)
    _dswiglu_backward(
        X=preact.view(dtype=torch.int32),
        Y=grad_out,
        Z=grad_pre.view(dtype=torch.int32),
    )
    return grad_pre


@autotune(
    configs=[AutotuneConfig(config=c) for c in _CE_ELEMENTWISE_CONFIGS],
    key=["ignore_index"],
    # the kernel overwrites the logits with their gradient; `zdz` may be `lses` itself, and may be None
    restore_value=("logits", "lses"),
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _cross_entropy_fwd_bwd_tuned(
    logits: torch.Tensor,
    lses: torch.Tensor,
    target: torch.Tensor,
    losses: torch.Tensor,
    zdz: torch.Tensor | None,
    ignore_index: int,
    config: ElementwiseConfig | None,
) -> None:
    if config is None:
        config = ElementwiseConfig(thr_m=4, thr_n=32, val_m=4)

    if zdz is None:
        partials = None
    else:
        M, N = logits.shape
        dtype_width = logits.element_size() * 8
        vector_size = NUM_BITS_PER_COPY // dtype_width
        n_tiles = ceil_div(N, config.thr_n * vector_size)
        partials = torch.empty(M, n_tiles, dtype=torch.float32, device=logits.device)
    cross_entropy_fwd_bwd_(
        logits=logits,
        lses=lses,
        target=target,
        losses=losses,
        partials=partials,
        ignore_index=ignore_index,
        thr_m=config.thr_m,
        thr_n=config.thr_n,
        val_m=config.val_m,
    )
    if zdz is not None:
        _sum_reduce_compiled(
            partials=partials,
            out=zdz,
            dim=-1,
        )


@_kernel_op("coda::_cross_entropy_fwd_bwd", mutates_args=("logits", "losses", "zdz"))
def _cross_entropy_fwd_bwd(
    logits: torch.Tensor,
    lses: torch.Tensor,
    target: torch.Tensor,
    losses: torch.Tensor,
    zdz: torch.Tensor | None,
    ignore_index: int,
) -> None:
    _cross_entropy_fwd_bwd_tuned(
        logits=logits,
        lses=lses,
        target=target,
        losses=losses,
        zdz=zdz,
        ignore_index=ignore_index,
    )


def cross_entropy_fwd_bwd(
    logits: torch.Tensor,
    lses: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool = False,
    losses: torch.Tensor | None = None,
    zdz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if losses is None:
        # zero-init as the kernel never writes ignored rows' losses
        losses = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    else:
        losses.zero_()
    if return_zdz and zdz is None:
        # no zero-init: the kernel writes per-tile partials and the reduce overwrites every row
        zdz = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    _cross_entropy_fwd_bwd(
        logits=logits,
        lses=lses,
        target=target,
        losses=losses,
        zdz=zdz,
        ignore_index=ignore_index,
    )
    return losses, zdz


@autotune(
    configs=[AutotuneConfig(config=c) for c in _ELEMENTWISE_CONFIGS],
    key=["head_dim", "num_heads_q", "num_heads_k", "eps"],
    # `dx` may be `x` itself, or the buffer behind `dq` and `dk`
    restore_value=("dx",),
    prune_configs_by={"early_config_prune": _prune_rope_configs},
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _qknorm_rope_bwd_tuned(
    dx: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dgamma: torch.Tensor,
    x: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    config: ElementwiseConfig | None,
) -> None:
    if config is None:
        config = ElementwiseConfig(thr_m=4, thr_n=32, val_m=4)

    tile_m = config.thr_m * config.val_m
    num_m_tiles = ceil_div(x.shape[0], tile_m)
    num_heads_qk = num_heads_q + num_heads_k
    dgamma_partials = torch.empty(
        num_m_tiles,
        x.shape[1],
        dtype=torch.float32,
        device=x.device,
    )
    qknorm_rope_bwd_(
        dx=dx,
        dq=dq,
        dk=dk,
        dgamma=dgamma_partials,
        x=x,
        head_mean_sq=head_mean_sq,
        gamma=gamma,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        thr_m=config.thr_m,
        thr_n=config.thr_n,
        val_m=config.val_m,
    )
    # fold the per-column partials into the two head gammas
    dgamma_partials = rearrange(
        dgamma_partials,
        "nt (h d) -> nt h d",
        nt=num_m_tiles,
        h=num_heads_qk,
        d=head_dim,
    )
    _sum_reduce(
        partials=dgamma_partials[:, :num_heads_q, :],
        out=dgamma[:head_dim],
        dim=(0, 1),
    )
    _sum_reduce(
        partials=dgamma_partials[:, num_heads_q:, :],
        out=dgamma[head_dim:],
        dim=(0, 1),
    )


@_kernel_op("coda::_qknorm_rope_bwd", mutates_args=("dx", "dgamma"))
def _qknorm_rope_bwd(
    dx: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dgamma: torch.Tensor,
    x: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
) -> None:
    _qknorm_rope_bwd_tuned(
        dx=dx,
        dq=dq,
        dk=dk,
        dgamma=dgamma,
        x=x,
        head_mean_sq=head_mean_sq,
        gamma=gamma,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
    )


def qknorm_rope_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    x: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    dx: torch.Tensor | None = None,
    dgamma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # x = [q, k]
    if dx is None:
        dx = torch.empty_like(x)
    if dgamma is None:
        dgamma = torch.empty_like(gamma, dtype=torch.float32)
    _qknorm_rope_bwd(
        dx=dx,
        dq=dq,
        dk=dk,
        dgamma=dgamma,
        x=x,
        head_mean_sq=head_mean_sq,
        gamma=gamma,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
    )
    return dx, dgamma


@autotune(
    configs=[AutotuneConfig(config=c) for c in _ZDZ_CONFIGS],
    prune_configs_by={"early_config_prune": _prune_rope_bwd_zdz_configs},
    # `dz` may be `y` or `dy` itself
    restore_value=("dz",),
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _rope_bwd_zdz_tuned(
    y: torch.Tensor,
    dy: torch.Tensor,
    dz: torch.Tensor,
    zdz: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    scale: float,
    config: ElementwiseConfig | None,
) -> None:
    if config is None:
        config = ElementwiseConfig(thr_m=4, thr_n=32, val_m=4)

    rope_bwd_zdz_(
        y=y,
        dy=dy,
        dz=dz,
        zdz=zdz,
        pos=pos,
        freq=freq,
        scale=scale,
        thr_m=config.thr_m,
        thr_n=config.thr_n,
        val_m=config.val_m,
    )


@_kernel_op("coda::_rope_bwd_zdz", mutates_args=("dz", "zdz"))
def _rope_bwd_zdz(
    y: torch.Tensor,
    dy: torch.Tensor,
    dz: torch.Tensor,
    zdz: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    scale: float,
) -> None:
    _rope_bwd_zdz_tuned(
        y=y,
        dy=dy,
        dz=dz,
        zdz=zdz,
        pos=pos,
        freq=freq,
        scale=scale,
    )


def rope_bwd_zdz(
    y: torch.Tensor,
    dy: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    scale: float = 1.0,
    dz: torch.Tensor | None = None,
    zdz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert y.ndim == 2
    assert y.shape == dy.shape
    assert y.dtype == dy.dtype
    assert pos.shape == (y.shape[0],)
    assert pos.dtype in (torch.int32, torch.float32)
    assert freq.shape == (y.shape[1],)
    assert freq.dtype == torch.float32
    if dz is None:
        dz = torch.empty_like(y)
    if zdz is None:
        zdz = torch.empty(y.shape[0], dtype=torch.float32, device=y.device)
    _rope_bwd_zdz(
        y=y,
        dy=dy,
        dz=dz,
        zdz=zdz,
        pos=pos,
        freq=freq,
        scale=scale,
    )
    return dz, zdz
