import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

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
        pass

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None]:
        pass


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
