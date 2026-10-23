import torch
from einops import rearrange
from quack.rms_final_reduce import rms_final_reduce as quack_rms_final_reduce
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.ops.constants import ALLOW_INPLACE_GRAD_OUTPUT
from coda.core.elementwise.functional import (
    cross_entropy_fwd_bwd,
    rope_bwd_zdz,
)
from coda.core.gemm.functional import (
    gemm as coda_gemm,
    gemm_scalar_scale,
    gemm_residual_partial_rmsnorm,
    gemm_residual_partial_rmsnorm_bwd,
    gemm_rmsnorm_lse,
    gemm_rmsnorm_rope,
    gemm_rmsnorm_swiglu,
    gemm_swiglu_bwd_zdz,
)

USE_CODA_GEMM = False


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if USE_CODA_GEMM:
        return coda_gemm(A=A, B=B)
    else:
        return torch.matmul(A, B)


def block_pre_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert wn.dtype == x.dtype
    assert x.ndim == 2
    dim = x.shape[1]
    h = x * rearrange(wn, "d -> 1 d")
    rstd = quack_rms_final_reduce(
        x=x.float() ** 2,
        scale=1.0 / dim,
        eps=eps,
    )
    qkv = gemm_rmsnorm_rope(
        A=h,
        B=w.mT,
        rstd=rstd,
        positions=positions,
        frequencies=frequencies,
    )
    return qkv, rstd


def block_pre_backward(
    dx: torch.Tensor,
    dy: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    qkv: torch.Tensor,
    rstd: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2
    dim = x.shape[1]
    # y = R z and dz = R^T dy, hence
    # sum(z * dz) = z^T R^T dy = (R z)^T dy = sum(y * dy).
    dz, zdz = rope_bwd_zdz(
        y=qkv,
        dy=dy,
        pos=positions,
        freq=frequencies,
        scale=1.0 / dim,
    )
    dx_out = dx if ALLOW_INPLACE_GRAD_OUTPUT else dx.clone()
    dx_out, dwn, x_out = gemm_residual_partial_rmsnorm_bwd(
        A=dz,
        # TODO: the function itself also transposes `B`
        # maybe we can directly pass in `BT` there
        B=w.mT,
        weight=wn,
        pre=x,
        ZdZ=zdz,
        rstd=rstd,
        dX=dx_out,
    )
    dw = gemm(dz.mT, x_out)
    return dx_out, dw, dwn.to(dtype=wn.dtype)


class BlockPre(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        wn: torch.Tensor,
        positions: torch.Tensor,
        frequencies: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv, rstd = block_pre_forward(
            x=x,
            w=w,
            wn=wn,
            positions=positions,
            frequencies=frequencies,
            eps=eps,
        )
        ctx.save_for_backward(
            x,
            w,
            wn,
            qkv,
            rstd,
            positions,
            frequencies,
        )
        return x, qkv

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dx: torch.Tensor,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        (
            x,
            w,
            wn,
            qkv,
            rstd,
            positions,
            frequencies,
        ) = ctx.saved_tensors
        dx_out, dw, dwn = block_pre_backward(
            dx=dx,
            dy=dy,
            x=x,
            w=w,
            wn=wn,
            qkv=qkv,
            rstd=rstd,
            positions=positions,
            frequencies=frequencies,
        )
        return (
            dx_out,
            dw,
            dwn,
            None,  # positions
            None,  # frequencies
            None,  # eps
        )


def block_post_forward(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
    ignore_index: int,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    x1, h1, rstd1 = gemm_residual_partial_rmsnorm(
        A=y0,
        B=w0.mT,
        C=x0,
        weight=wn0,
        eps=eps,
    )
    z1, y1 = gemm_rmsnorm_swiglu(
        A=h1,
        B=w1.mT,
        rstd=rstd1,
    )
    x2, h2, rstd2 = gemm_residual_partial_rmsnorm(
        A=y1,
        B=w2.mT,
        C=x1,
        weight=wn1,
        eps=eps,
    )
    logits, lses = gemm_rmsnorm_lse(
        A=h2,
        B=w3.mT,
        rstd=rstd2,
    )
    losses, zdz2 = cross_entropy_fwd_bwd(
        logits=logits,
        lses=lses,
        target=targets,
        ignore_index=ignore_index,
        return_zdz=True,
    )
    dlogits = logits
    if reduction == "mean":
        scale = 1.0 / (targets != ignore_index).sum().float()
        loss = losses.sum() * scale
    else:
        scale = None
        loss = losses.sum()
    return loss, x1, x2, rstd1, rstd2, z1, scale, dlogits, zdz2


def block_post_backward(
    dloss: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    rstd1: torch.Tensor,
    rstd2: torch.Tensor,
    z1: torch.Tensor,
    scale: torch.Tensor | None,
    dlogits: torch.Tensor,
    zdz2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x2.ndim == 2
    assert x1.shape == x2.shape
    dim = x2.shape[1]
    if scale is not None:
        # `scale` captures `1 / count`
        dloss = dloss * scale
    dx_out, dwn1, x_out2 = gemm_residual_partial_rmsnorm_bwd(
        A=dlogits,
        B=w3.mT,
        weight=wn1,
        pre=x2,
        ZdZ=zdz2 * (dloss / dim),
        rstd=rstd2,
        alpha=dloss,
        dX=None,
    )
    dw3 = gemm_scalar_scale(
        A=dlogits.mT,
        B=x_out2,
        alpha=dloss,
    )
    dz1, zdz1, y1 = gemm_swiglu_bwd_zdz(
        A=dx_out,
        B=w2.mT,
        Z=z1,
        scale=1.0 / dim,
    )
    dw2 = gemm(dx_out.mT, y1)
    dx_out, dwn0, x_out1 = gemm_residual_partial_rmsnorm_bwd(
        A=dz1,
        B=w1.mT,
        weight=wn0,
        pre=x1,
        ZdZ=zdz1,
        rstd=rstd1,
        dX=dx_out,
    )
    dw1 = gemm(dz1.mT, x_out1)
    dy0 = gemm(dx_out, w0)
    dw0 = gemm(dx_out.mT, y0)
    return (
        dx_out,
        dy0,
        dw0,
        dw1,
        dw2,
        dw3,
        dwn0.to(dtype=wn0.dtype),
        dwn1.to(dtype=wn1.dtype),
    )


class BlockPost(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x0: torch.Tensor,
        y0: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        wn0: torch.Tensor,
        wn1: torch.Tensor,
        targets: torch.Tensor,
        eps: float,
        ignore_index: int,
        reduction: str,
    ) -> torch.Tensor:
        (
            loss,
            x1,
            x2,
            rstd1,
            rstd2,
            z1,
            scale,
            dlogits,
            zdz2,
        ) = block_post_forward(
            x0=x0,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            targets=targets,
            eps=eps,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        ctx.save_for_backward(
            y0,
            w0,
            w1,
            w2,
            w3,
            wn0,
            wn1,
            x1,
            x2,
            rstd1,
            rstd2,
            z1,
            scale,
            dlogits,
            zdz2,
        )
        return loss

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dloss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        (
            y0,
            w0,
            w1,
            w2,
            w3,
            wn0,
            wn1,
            x1,
            x2,
            rstd1,
            rstd2,
            z1,
            scale,
            dlogits,
            zdz2,
        ) = ctx.saved_tensors
        dx0, dy0, dw0, dw1, dw2, dw3, dwn0, dwn1 = block_post_backward(
            dloss=dloss,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            x1=x1,
            x2=x2,
            rstd1=rstd1,
            rstd2=rstd2,
            z1=z1,
            scale=scale,
            dlogits=dlogits,
            zdz2=zdz2,
        )
        return (
            dx0,
            dy0,
            dw0,
            dw1,
            dw2,
            dw3,
            dwn0,
            dwn1,
            None,  # targets
            None,  # eps
            None,  # ignore_index
            None,  # reduction
        )


def block_forward(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1, h1, rstd1 = gemm_residual_partial_rmsnorm(
        A=y0,
        B=w0.mT,
        C=x0,
        weight=wn0,
        eps=eps,
    )
    z1, y1 = gemm_rmsnorm_swiglu(
        A=h1,
        B=w1.mT,
        rstd=rstd1,
    )
    x2, h2, rstd2 = gemm_residual_partial_rmsnorm(
        A=y1,
        B=w2.mT,
        C=x1,
        weight=wn1,
        eps=eps,
    )
    qkv = gemm_rmsnorm_rope(
        A=h2,
        B=w3.mT,
        rstd=rstd2,
        positions=positions,
        frequencies=frequencies,
    )
    return qkv, x1, x2, rstd1, rstd2, z1


def block_backward(
    dx2: torch.Tensor,
    dy2: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    qkv: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    rstd1: torch.Tensor,
    rstd2: torch.Tensor,
    z1: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x2.ndim == 2
    assert x1.shape == x2.shape
    dim = x2.shape[1]
    # y = R z and dz = R^T dy, hence
    # sum(z * dz) = z^T R^T dy = (R z)^T dy = sum(y * dy).
    dz2, zdz2 = rope_bwd_zdz(
        y=qkv,
        dy=dy2,
        pos=positions,
        freq=frequencies,
        scale=1.0 / dim,
    )
    dx_out = dx2 if ALLOW_INPLACE_GRAD_OUTPUT else dx2.clone()
    dx_out, dwn1, x_out2 = gemm_residual_partial_rmsnorm_bwd(
        A=dz2,
        B=w3.mT,
        weight=wn1,
        pre=x2,
        # we don't need `/ dim` here because `rope_bwd_zdz` already does it
        ZdZ=zdz2,
        rstd=rstd2,
        dX=dx_out,
    )
    dw3 = gemm(dz2.mT, x_out2)
    dz1, zdz1, y1 = gemm_swiglu_bwd_zdz(
        A=dx_out,
        B=w2.mT,
        Z=z1,
        scale=1.0 / dim,
    )
    dw2 = gemm(dx_out.mT, y1)
    dx_out, dwn0, x_out1 = gemm_residual_partial_rmsnorm_bwd(
        A=dz1,
        B=w1.mT,
        weight=wn0,
        pre=x1,
        ZdZ=zdz1,
        rstd=rstd1,
        dX=dx_out,
    )
    dw1 = gemm(dz1.mT, x_out1)
    dy0 = gemm(dx_out, w0)
    dw0 = gemm(dx_out.mT, y0)
    return (
        dx_out,
        dy0,
        dw0,
        dw1,
        dw2,
        dw3,
        dwn0.to(dtype=wn0.dtype),
        dwn1.to(dtype=wn1.dtype),
    )


class Block(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x0: torch.Tensor,
        y0: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        wn0: torch.Tensor,
        wn1: torch.Tensor,
        positions: torch.Tensor,
        frequencies: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv, x1, x2, rstd1, rstd2, z1 = block_forward(
            x0=x0,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            positions=positions,
            frequencies=frequencies,
            eps=eps,
        )
        ctx.save_for_backward(
            y0,
            w0,
            w1,
            w2,
            w3,
            wn0,
            wn1,
            qkv,
            x1,
            x2,
            rstd1,
            rstd2,
            z1,
            positions,
            frequencies,
        )
        return x2, qkv

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dx2: torch.Tensor,
        dy2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        (
            y0,
            w0,
            w1,
            w2,
            w3,
            wn0,
            wn1,
            qkv,
            x1,
            x2,
            rstd1,
            rstd2,
            z1,
            positions,
            frequencies,
        ) = ctx.saved_tensors
        dx0, dy0, dw0, dw1, dw2, dw3, dwn0, dwn1 = block_backward(
            dx2=dx2,
            dy2=dy2,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            qkv=qkv,
            x1=x1,
            x2=x2,
            rstd1=rstd1,
            rstd2=rstd2,
            z1=z1,
            positions=positions,
            frequencies=frequencies,
        )
        return (
            dx0,
            dy0,
            dw0,
            dw1,
            dw2,
            dw3,
            dwn0,
            dwn1,
            None,  # positions
            None,  # frequencies
            None,  # eps
        )


def block_pre(
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, _ = x.shape
    dim1, dim0 = w.shape
    assert x.shape == (batch, length, dim0)
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert w.shape == (dim1, dim0)
    assert w.dtype == x.dtype
    assert wn.shape == (dim0,)
    assert wn.dtype == x.dtype
    assert positions.shape == (batch * length,)
    assert positions.dtype in (torch.float32, torch.int32)
    assert frequencies.shape == (dim1,)
    assert frequencies.dtype == torch.float32
    residual, qkv = BlockPre.apply(
        rearrange(x, "b t d -> (b t) d"),
        w,
        wn,
        positions,
        frequencies,
        eps,
    )
    return (
        rearrange(residual, "(b t) d -> b t d", b=batch),
        rearrange(qkv, "(b t) d -> b t d", b=batch),
    )


def block_post(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    batch, length, _ = x0.shape
    dim0, _ = w0.shape
    dim1, _ = w1.shape
    dim3, _ = w3.shape
    assert x0.shape == (batch, length, dim0)
    assert x0.dtype in (torch.float16, torch.bfloat16)
    assert y0.shape == (batch, length, dim0)
    assert y0.dtype == x0.dtype
    assert w0.shape == (dim0, dim0)
    assert w0.dtype == x0.dtype
    assert w1.shape == (dim1, dim0)
    assert w1.dtype == x0.dtype
    assert w2.shape == (dim0, dim1 // 2)
    assert w2.dtype == x0.dtype
    assert w3.shape == (dim3, dim0)
    assert w3.dtype == x0.dtype
    assert wn0.shape == (dim0,)
    assert wn0.dtype == x0.dtype
    assert wn1.shape == (dim0,)
    assert wn1.dtype == x0.dtype
    assert targets.shape == (batch * length,)
    assert targets.dtype == torch.int32
    assert reduction in ("mean", "sum")
    return BlockPost.apply(
        rearrange(x0, "b t d -> (b t) d"),
        rearrange(y0, "b t d -> (b t) d"),
        w0,
        w1,
        w2,
        w3,
        wn0,
        wn1,
        targets,
        eps,
        ignore_index,
        reduction,
    )


def block(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, _ = x0.shape
    dim0, _ = w0.shape
    dim1, _ = w1.shape
    dim3, _ = w3.shape
    assert x0.shape == (batch, length, dim0)
    assert x0.dtype in (torch.float16, torch.bfloat16)
    assert y0.shape == (batch, length, dim0)
    assert y0.dtype == x0.dtype
    assert w0.shape == (dim0, dim0)
    assert w0.dtype == x0.dtype
    assert w1.shape == (dim1, dim0)
    assert w1.dtype == x0.dtype
    assert w2.shape == (dim0, dim1 // 2)
    assert w2.dtype == x0.dtype
    assert w3.shape == (dim3, dim0)
    assert w3.dtype == x0.dtype
    assert wn0.shape == (dim0,)
    assert wn0.dtype == x0.dtype
    assert wn1.shape == (dim0,)
    assert wn1.dtype == x0.dtype
    assert positions.shape == (batch * length,)
    assert positions.dtype in (torch.float32, torch.int32)
    assert frequencies.shape == (dim3,)
    assert frequencies.dtype == torch.float32
    residual, qkv = Block.apply(
        rearrange(x0, "b t d -> (b t) d"),
        rearrange(y0, "b t d -> (b t) d"),
        w0,
        w1,
        w2,
        w3,
        wn0,
        wn1,
        positions,
        frequencies,
        eps,
    )
    return (
        rearrange(residual, "(b t) d -> b t d", b=batch),
        rearrange(qkv, "(b t) d -> b t d", b=batch),
    )
