import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.ops.misc_utils import ceil_div
from coda.core.gemm.functional import gemm, gemm_sigmoid


class LinearSigmoid(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        out_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if out_dtype is None:
            out_dtype = x.dtype
        return out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        return dx, dweight, None


def linear_sigmoid(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return LinearSigmoid.apply(x, weight, out_dtype)
