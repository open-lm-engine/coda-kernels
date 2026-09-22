import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from quack.gemm_interface import (
    gemm as quack_gemm,
    gemm_add as quack_gemm_add,
)

from coda.core.elementwise.functional import qknorm_rope_bwd
from coda.core.gemm.functional import gemm, gemm_qknorm_rope


class LinearQKNormRope(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_norm: torch.Tensor,
        positions: torch.Tensor,
        frequencies: torch.Tensor,
        head_dim: int,
        num_heads_q: int,
        num_heads_k: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # weight rows are [q | k | v]: q and k are normed per head and rotated, v is a plain projection
        size_q = head_dim * num_heads_q
        size_qk = head_dim * (num_heads_q + num_heads_k)
        qk, preact, head_mean_sq = gemm_qknorm_rope(
            x,
            weight[:size_qk, :].mT,
            weight=weight_norm,
            positions=positions,
            frequencies=frequencies,
            head_dim=head_dim,
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            eps=eps,
        )
        v = gemm(x, weight[size_qk:, :].mT)
        # q and k are views: the backward takes their gradients apart anyway, so splitting here costs nothing and
        # saves the concatenation that the caller's own split would need
        q, k = qk.split((size_q, size_qk - size_q), dim=-1)

        ctx.head_dim = head_dim
        ctx.num_heads_q = num_heads_q
        ctx.num_heads_k = num_heads_k
        ctx.eps = eps
        ctx.save_for_backward(
            x,
            weight,
            weight_norm,
            positions,
            frequencies,
            preact,
            head_mean_sq,
        )
        return q, k, v

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None]:
        (
            x,
            weight,
            weight_norm,
            positions,
            frequencies,
            preact,
            head_mean_sq,
        ) = ctx.saved_tensors
        grad_pre, dweight_norm = qknorm_rope_bwd(
            dq=dq,
            dk=dk,
            dv=dv,
            x=preact,
            head_mean_sq=head_mean_sq,
            weight=weight_norm,
            pos=positions,
            freq=frequencies,
            head_dim=ctx.head_dim,
            num_heads_q=ctx.num_heads_q,
            num_heads_k=ctx.num_heads_k,
            eps=ctx.eps,
        )
        dx = gemm(grad_pre, weight)
        dweight = gemm(grad_pre.mT, x)
        dweight_norm = dweight_norm.to(dtype=weight_norm.dtype)
        return (
            dx,
            dweight,
            dweight_norm,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def linear_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_norm: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return LinearQKNormRope.apply(
        x,
        weight,
        weight_norm,
        positions,
        frequencies,
        head_dim,
        num_heads_q,
        num_heads_k,
        eps,
    )
