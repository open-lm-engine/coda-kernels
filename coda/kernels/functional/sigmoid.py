import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

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
        # the kernel accumulates in float32 and writes the gate in `out_dtype`
        out = torch.empty(
            x.shape[0],
            weight.shape[0],
            dtype=out_dtype,
            device=x.device,
        )
        gemm_sigmoid(x, weight.mT, out=out)
        ctx.save_for_backward(x, weight, out)
        return out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight, out = ctx.saved_tensors
        grad_pre = torch.ops.aten.sigmoid_backward(dout, out).to(dtype=x.dtype)
        dx = gemm(grad_pre, weight)
        dweight = gemm(grad_pre.mT, x)
        return dx, dweight, None


def linear_sigmoid(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return LinearSigmoid.apply(x, weight, out_dtype)
