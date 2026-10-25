import torch
from einops import rearrange
from quack.rms_final_reduce import rms_final_reduce as quack_rms_final_reduce
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.ops.constants import ALLOW_INPLACE_GRAD_OUTPUT
from coda.core.elementwise.functional import (
    qknorm_rope_bwd_zdz,
    xsa_fwd,
)
from coda.core.gemm.functional import (
    gemm as coda_gemm,
    gemm_scalar_scale,
    gemm_residual_partial_rmsnorm,
    gemm_residual_partial_rmsnorm_bwd,
    gemm_rmsnorm,
    gemm_rmsnorm_qknorm_rope,
    gemm_rmsnorm_swiglu,
    gemm_swiglu_bwd_zdz,
    gemm_xsa_bwd,
)
from coda.kernels.functional.cross_entropy import gemm_rmsnorm_cross_entropy_fwd_bwd

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
    wn2: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert wn.dtype == x.dtype
    assert x.ndim == 2
    dim = x.shape[1]
    size_qkv = w.shape[0]
    assert size_qkv % (3 * head_dim) == 0
    num_heads = size_qkv // (3 * head_dim)
    size_qk = 2 * num_heads * head_dim
    size_v = num_heads * head_dim

    h = x * rearrange(wn, "d -> 1 d")
    rstd = quack_rms_final_reduce(
        x=x.float() ** 2,
        scale=1.0 / dim,
        eps=eps,
    )

    # only q and k are normed per head and roped, so V is a GEMM of its own into
    # the same [Q | K | V] buffer
    qkv = x.new_empty(x.shape[0], size_qkv)
    qk, v = qkv.split([size_qk, size_v], dim=1)
    w_qk, w_v = w.split([size_qk, size_v], dim=0)

    _, preact, head_mean_sq = gemm_rmsnorm_qknorm_rope(
        A=h,
        B=w_qk.mT,
        rstd=rstd,
        weight=wn2,
        positions=positions,
        frequencies=frequencies,
        head_dim=head_dim,
        num_heads_q=num_heads,
        num_heads_k=num_heads,
        eps=eps,
        out=qk,
    )
    gemm_rmsnorm(
        A=h,
        B=w_v.mT,
        rstd=rstd,
        out=v,
    )
    return qkv, v, rstd, preact, head_mean_sq


def block_pre_backward(
    dx: torch.Tensor,
    dqkv: torch.Tensor,
    dv: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    wn2: torch.Tensor,
    qkv: torch.Tensor,
    rstd: torch.Tensor,
    preact: torch.Tensor,
    head_mean_sq: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2
    dim = x.shape[1]
    size_qkv = w.shape[0]
    assert size_qkv % (3 * head_dim) == 0
    num_heads = size_qkv // (3 * head_dim)
    size_qk = 2 * num_heads * head_dim
    size_v = num_heads * head_dim

    _, v = qkv.split([size_qk, size_v], dim=1)
    if ALLOW_INPLACE_GRAD_OUTPUT:
        # overwrite the incoming gradients in place: nothing reads them afterwards
        dz = dqkv
        dx_out = dx
    else:
        # the kernel allocates dz, and the GEMM accumulates into a copy of dx
        dz = None
        dx_out = dx.clone()

    # sum(z * dz) = sum(qkv * dqkv) fails under the per-head norm: a normed head
    # contributes eps * head_rstd^3 * dhead_rstd, formed in fp32 by the kernel; the V
    # columns add the next block's XSA grad `dv` to `dqkv`
    dz, zdz, dwn2 = qknorm_rope_bwd_zdz(
        dy=dqkv,
        v=v,
        preact=preact,
        rstd=rstd,
        head_mean_sq=head_mean_sq,
        gamma=wn2,
        pos=positions,
        freq=frequencies,
        head_dim=head_dim,
        num_heads_q=num_heads,
        num_heads_k=num_heads,
        eps=eps,
        scale=1.0 / dim,
        dv=dv,
        dz=dz,
    )
    dx_out, dwn, x_out = gemm_residual_partial_rmsnorm_bwd(
        A=dz,
        B=w.mT,
        weight=wn,
        pre=x,
        ZdZ=zdz,
        rstd=rstd,
        dX=dx_out,
    )
    dw = gemm(dz.mT, x_out)
    return dx_out, dw, dwn.to(dtype=wn.dtype), dwn2.to(dtype=wn2.dtype)


class BlockPre(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        wn: torch.Tensor,
        wn2: torch.Tensor,
        positions: torch.Tensor,
        frequencies: torch.Tensor,
        head_dim: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, v, rstd, preact, head_mean_sq = block_pre_forward(
            x=x,
            w=w,
            wn=wn,
            wn2=wn2,
            positions=positions,
            frequencies=frequencies,
            head_dim=head_dim,
            eps=eps,
        )
        ctx.save_for_backward(
            x,
            w,
            wn,
            wn2,
            qkv,
            rstd,
            preact,
            head_mean_sq,
            positions,
            frequencies,
        )
        ctx.head_dim = head_dim
        ctx.eps = eps
        # `v` is a view of `qkv`: as an output of its own, the next block's XSA grad
        # reaches the backward apart from FA's, with no slice to scatter it back
        return x, qkv, v

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dx: torch.Tensor,
        dqkv: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        (
            x,
            w,
            wn,
            wn2,
            qkv,
            rstd,
            preact,
            head_mean_sq,
            positions,
            frequencies,
        ) = ctx.saved_tensors
        dx_out, dw, dwn, dwn2 = block_pre_backward(
            dx=dx,
            dqkv=dqkv,
            dv=dv,
            x=x,
            w=w,
            wn=wn,
            wn2=wn2,
            qkv=qkv,
            rstd=rstd,
            preact=preact,
            head_mean_sq=head_mean_sq,
            positions=positions,
            frequencies=frequencies,
            head_dim=ctx.head_dim,
            eps=ctx.eps,
        )
        return (
            dx_out,
            dw,
            dwn,
            dwn2,
            None,  # positions
            None,  # frequencies
            None,  # head_dim
            None,  # eps
        )


def block_post_forward(
    x0: torch.Tensor,
    o0: torch.Tensor,
    v0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    targets: torch.Tensor,
    head_dim: int,
    eps: float,
    ignore_index: int,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    y0 = xsa_fwd(
        x=o0,
        v=v0,
        head_dim=head_dim,
    )
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
    # fused LM head and cross entropy, as accurate as fp32: the logits are never rounded to bf16
    losses, dlogits, zdz2 = gemm_rmsnorm_cross_entropy_fwd_bwd(
        A=h2,
        B=w3.mT,
        rstd=rstd2,
        target=targets,
        ignore_index=ignore_index,
        return_zdz=True,
    )
    if reduction == "mean":
        scale = 1.0 / (targets != ignore_index).sum().float()
        loss = losses.sum() * scale
    else:
        scale = None
        loss = losses.sum()
    return loss, y0, x1, x2, rstd1, rstd2, z1, scale, dlogits, zdz2


def block_post_backward(
    dloss: torch.Tensor,
    o0: torch.Tensor,
    v0: torch.Tensor,
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
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    do0, dv0 = gemm_xsa_bwd(
        A=dx_out,
        B=w0.mT,
        x=o0,
        v=v0,
        head_dim=head_dim,
    )
    dw0 = gemm(dx_out.mT, y0)
    return (
        dx_out,
        do0,
        dv0,
        dw0,
        dw1,
        dw2,
        dw3,
        dwn0.to(dtype=wn0.dtype),
        dwn1.to(dtype=wn1.dtype),
    )


class BlockPost(torch.autograd.Function):

    @staticmethod
    # `v0` is a view of the previous `qkv`: making it contiguous would copy V
    # and keep that copy alive for the backward
    @input_guard(no_guard_contiguous=["v0"])
    @autocast_custom_fwd
    def forward(
        ctx,
        x0: torch.Tensor,
        o0: torch.Tensor,
        v0: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        wn0: torch.Tensor,
        wn1: torch.Tensor,
        targets: torch.Tensor,
        head_dim: int,
        eps: float,
        ignore_index: int,
        reduction: str,
    ) -> torch.Tensor:
        (
            loss,
            y0,
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
            o0=o0,
            v0=v0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            targets=targets,
            head_dim=head_dim,
            eps=eps,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        ctx.save_for_backward(
            o0,
            v0,
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
        ctx.head_dim = head_dim
        return loss

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dloss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None]:
        (
            o0,
            v0,
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
        dx0, do0, dv0, dw0, dw1, dw2, dw3, dwn0, dwn1 = block_post_backward(
            dloss=dloss,
            o0=o0,
            v0=v0,
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
            head_dim=ctx.head_dim,
        )
        return (
            dx0,
            do0,
            dv0,
            dw0,
            dw1,
            dw2,
            dw3,
            dwn0,
            dwn1,
            None,  # targets
            None,  # head_dim
            None,  # eps
            None,  # ignore_index
            None,  # reduction
        )


def block_forward(
    x0: torch.Tensor,
    o0: torch.Tensor,
    v0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    wn2: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    size_qkv = w3.shape[0]
    assert size_qkv % (3 * head_dim) == 0
    num_heads = size_qkv // (3 * head_dim)
    size_qk = 2 * num_heads * head_dim
    size_v = num_heads * head_dim

    y0 = xsa_fwd(
        x=o0,
        v=v0,
        head_dim=head_dim,
    )
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

    # only q and k are normed per head and roped, so V is a GEMM of its own into
    # the same [Q | K | V] buffer
    qkv = x2.new_empty(x2.shape[0], size_qkv)
    qk, v2 = qkv.split([size_qk, size_v], dim=1)
    w3_qk, w3_v = w3.split([size_qk, size_v], dim=0)

    _, preact2, head_mean_sq2 = gemm_rmsnorm_qknorm_rope(
        A=h2,
        B=w3_qk.mT,
        rstd=rstd2,
        weight=wn2,
        positions=positions,
        frequencies=frequencies,
        head_dim=head_dim,
        num_heads_q=num_heads,
        num_heads_k=num_heads,
        eps=eps,
        out=qk,
    )
    gemm_rmsnorm(
        A=h2,
        B=w3_v.mT,
        rstd=rstd2,
        out=v2,
    )
    return qkv, v2, y0, x1, x2, rstd1, rstd2, z1, preact2, head_mean_sq2


def block_backward(
    dx2: torch.Tensor,
    dqkv: torch.Tensor,
    dv2: torch.Tensor,
    o0: torch.Tensor,
    v0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    wn2: torch.Tensor,
    qkv: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    rstd1: torch.Tensor,
    rstd2: torch.Tensor,
    z1: torch.Tensor,
    preact2: torch.Tensor,
    head_mean_sq2: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x2.ndim == 2
    assert x1.shape == x2.shape
    dim = x2.shape[1]
    size_qkv = w3.shape[0]
    assert size_qkv % (3 * head_dim) == 0
    num_heads = size_qkv // (3 * head_dim)
    size_qk = 2 * num_heads * head_dim
    size_v = num_heads * head_dim

    _, v2 = qkv.split([size_qk, size_v], dim=1)
    if ALLOW_INPLACE_GRAD_OUTPUT:
        # overwrite the incoming gradients in place: nothing reads them afterwards
        dz2 = dqkv
        dx_out = dx2
    else:
        # the kernel allocates dz2, and the GEMM accumulates into a copy of dx2
        dz2 = None
        dx_out = dx2.clone()

    # sum(z2 * dz2) = sum(qkv * dqkv) fails under the per-head norm: a normed head
    # contributes eps * head_rstd^3 * dhead_rstd, formed in fp32 by the kernel; the V
    # columns add the next block's XSA grad `dv2` to `dqkv`
    dz2, zdz2, dwn2 = qknorm_rope_bwd_zdz(
        dy=dqkv,
        v=v2,
        preact=preact2,
        rstd=rstd2,
        head_mean_sq=head_mean_sq2,
        gamma=wn2,
        pos=positions,
        freq=frequencies,
        head_dim=head_dim,
        num_heads_q=num_heads,
        num_heads_k=num_heads,
        eps=eps,
        scale=1.0 / dim,
        dv=dv2,
        dz=dz2,
    )
    dx_out, dwn1, x_out2 = gemm_residual_partial_rmsnorm_bwd(
        A=dz2,
        B=w3.mT,
        weight=wn1,
        pre=x2,
        # we don't need `/ dim` here because `qknorm_rope_bwd_zdz` already does it
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
    do0, dv0 = gemm_xsa_bwd(
        A=dx_out,
        B=w0.mT,
        x=o0,
        v=v0,
        head_dim=head_dim,
    )
    dw0 = gemm(dx_out.mT, y0)
    return (
        dx_out,
        do0,
        dv0,
        dw0,
        dw1,
        dw2,
        dw3,
        dwn0.to(dtype=wn0.dtype),
        dwn1.to(dtype=wn1.dtype),
        dwn2.to(dtype=wn2.dtype),
    )


class Block(torch.autograd.Function):

    @staticmethod
    # `v0` is a view of the previous `qkv`: making it contiguous would copy V
    # and keep that copy alive for the backward
    @input_guard(no_guard_contiguous=["v0"])
    @autocast_custom_fwd
    def forward(
        ctx,
        x0: torch.Tensor,
        o0: torch.Tensor,
        v0: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        wn0: torch.Tensor,
        wn1: torch.Tensor,
        wn2: torch.Tensor,
        positions: torch.Tensor,
        frequencies: torch.Tensor,
        head_dim: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            qkv,
            v2,
            y0,
            x1,
            x2,
            rstd1,
            rstd2,
            z1,
            preact2,
            head_mean_sq2,
        ) = block_forward(
            x0=x0,
            o0=o0,
            v0=v0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            wn2=wn2,
            positions=positions,
            frequencies=frequencies,
            head_dim=head_dim,
            eps=eps,
        )
        ctx.save_for_backward(
            o0,
            v0,
            y0,
            w0,
            w1,
            w2,
            w3,
            wn0,
            wn1,
            wn2,
            qkv,
            x1,
            x2,
            rstd1,
            rstd2,
            z1,
            preact2,
            head_mean_sq2,
            positions,
            frequencies,
        )
        ctx.head_dim = head_dim
        ctx.eps = eps
        # `v2` is a view of `qkv`: as an output of its own, the next block's XSA grad
        # reaches the backward apart from FA's, with no slice to scatter it back
        return x2, qkv, v2

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dx2: torch.Tensor,
        dqkv: torch.Tensor,
        dv2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        (
            o0,
            v0,
            y0,
            w0,
            w1,
            w2,
            w3,
            wn0,
            wn1,
            wn2,
            qkv,
            x1,
            x2,
            rstd1,
            rstd2,
            z1,
            preact2,
            head_mean_sq2,
            positions,
            frequencies,
        ) = ctx.saved_tensors
        dx0, do0, dv0, dw0, dw1, dw2, dw3, dwn0, dwn1, dwn2 = block_backward(
            dx2=dx2,
            dqkv=dqkv,
            dv2=dv2,
            o0=o0,
            v0=v0,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            wn2=wn2,
            qkv=qkv,
            x1=x1,
            x2=x2,
            rstd1=rstd1,
            rstd2=rstd2,
            z1=z1,
            preact2=preact2,
            head_mean_sq2=head_mean_sq2,
            positions=positions,
            frequencies=frequencies,
            head_dim=ctx.head_dim,
            eps=ctx.eps,
        )
        return (
            dx0,
            do0,
            dv0,
            dw0,
            dw1,
            dw2,
            dw3,
            dwn0,
            dwn1,
            dwn2,
            None,  # positions
            None,  # frequencies
            None,  # head_dim
            None,  # eps
        )


def block_pre(
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    wn2: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, _ = x.shape
    _, dim0 = w.shape
    assert x.shape == (batch, length, dim0)
    # `v` feeds the XSA ops, which are bf16 only
    assert x.dtype == torch.bfloat16
    # MHA with H * head_dim == dim0: [Q | K | V] rows, q and k in adjacent-pair order
    assert w.shape == (3 * dim0, dim0)
    assert w.dtype == x.dtype
    assert wn.shape == (dim0,)
    assert wn.dtype == x.dtype
    # [gamma_q | gamma_k], each in the same pair order
    assert wn2.shape == (2 * head_dim,)
    assert wn2.dtype == x.dtype
    assert positions.shape == (batch * length,)
    assert positions.dtype in (torch.float32, torch.int32)
    # q and k only: v is not roped
    assert frequencies.shape == (2 * dim0,)
    assert frequencies.dtype == torch.float32
    assert dim0 % head_dim == 0
    residual, qkv, v = BlockPre.apply(
        rearrange(x, "b t d -> (b t) d"),
        w,
        wn,
        wn2,
        positions,
        frequencies,
        head_dim,
        eps,
    )
    return (
        rearrange(residual, "(b t) d -> b t d", b=batch),
        rearrange(qkv, "(b t) d -> b t d", b=batch),
        rearrange(v, "(b t) d -> b t d", b=batch),
    )


def block_post(
    x0: torch.Tensor,
    o0: torch.Tensor,
    v0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    targets: torch.Tensor,
    head_dim: int,
    eps: float,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    batch, length, _ = x0.shape
    dim0, _ = w0.shape
    dim1, _ = w1.shape
    dim3, _ = w3.shape
    assert x0.shape == (batch, length, dim0)
    # the XSA backward packs bf16 pairs
    assert x0.dtype == torch.bfloat16
    assert o0.shape == (batch, length, dim0)
    assert o0.dtype == x0.dtype
    assert v0.shape == (batch, length, dim0)
    assert v0.dtype == x0.dtype
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
    assert dim0 % head_dim == 0
    return BlockPost.apply(
        rearrange(x0, "b t d -> (b t) d"),
        rearrange(o0, "b t d -> (b t) d"),
        rearrange(v0, "b t d -> (b t) d"),
        w0,
        w1,
        w2,
        w3,
        wn0,
        wn1,
        targets,
        head_dim,
        eps,
        ignore_index,
        reduction,
    )


def block(
    x0: torch.Tensor,
    o0: torch.Tensor,
    v0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    wn2: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, _ = x0.shape
    dim0, _ = w0.shape
    dim1, _ = w1.shape
    assert x0.shape == (batch, length, dim0)
    # the XSA backward packs bf16 pairs
    assert x0.dtype == torch.bfloat16
    assert o0.shape == (batch, length, dim0)
    assert o0.dtype == x0.dtype
    assert v0.shape == (batch, length, dim0)
    assert v0.dtype == x0.dtype
    assert w0.shape == (dim0, dim0)
    assert w0.dtype == x0.dtype
    assert w1.shape == (dim1, dim0)
    assert w1.dtype == x0.dtype
    assert w2.shape == (dim0, dim1 // 2)
    assert w2.dtype == x0.dtype
    # MHA with H * head_dim == dim0: [Q | K | V] rows, q and k in adjacent-pair order
    assert w3.shape == (3 * dim0, dim0)
    assert w3.dtype == x0.dtype
    assert wn0.shape == (dim0,)
    assert wn0.dtype == x0.dtype
    assert wn1.shape == (dim0,)
    assert wn1.dtype == x0.dtype
    # [gamma_q | gamma_k], each in the same pair order
    assert wn2.shape == (2 * head_dim,)
    assert wn2.dtype == x0.dtype
    assert positions.shape == (batch * length,)
    assert positions.dtype in (torch.float32, torch.int32)
    # q and k only: v is not roped
    assert frequencies.shape == (2 * dim0,)
    assert frequencies.dtype == torch.float32
    assert dim0 % head_dim == 0
    residual, qkv, v2 = Block.apply(
        rearrange(x0, "b t d -> (b t) d"),
        rearrange(o0, "b t d -> (b t) d"),
        rearrange(v0, "b t d -> (b t) d"),
        w0,
        w1,
        w2,
        w3,
        wn0,
        wn1,
        wn2,
        positions,
        frequencies,
        head_dim,
        eps,
    )
    return (
        rearrange(residual, "(b t) d -> b t d", b=batch),
        rearrange(qkv, "(b t) d -> b t d", b=batch),
        rearrange(v2, "(b t) d -> b t d", b=batch),
    )
