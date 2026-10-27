import math
import torch
from einops import rearrange, reduce, repeat


# ---- primitives


def swiglu(preact: torch.Tensor) -> torch.Tensor:
    # gate and up in adjacent columns (2i, 2i + 1), the pairing CODA's gated epilogue emits. In bf16, eager rounds
    # silu(gate) before the product, as a model's eager swiglu does, and a compiled one does not
    pairs = rearrange(preact, "m (n two) -> m n two", two=2)
    gate = pairs[..., 0]
    up = pairs[..., 1]
    silu = torch.nn.functional.silu(gate)
    return silu * up


def dswiglu(preact: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # swiglu's vjp in closed form, interleaved like preact, and the postact it recomputes on the way: closed so that
    # it compiles under fullgraph as torch's arm, and pinned to fp64 autograd in test_refs
    pairs = rearrange(preact, "m (n two) -> m n two", two=2)
    gate = pairs[..., 0]
    up = pairs[..., 1]
    sigmoid = torch.nn.functional.sigmoid(gate)
    silu = gate * sigmoid
    grad_silu = grad * up
    one_minus_sigmoid = 1 - sigmoid
    silu_one_minus_sigmoid = silu * one_minus_sigmoid
    dsilu = sigmoid + silu_one_minus_sigmoid
    grad_gate = grad_silu * dsilu
    grad_up = grad * silu
    grad_pairs = torch.stack((grad_gate, grad_up), dim=-1)
    grad_preact = rearrange(grad_pairs, "m n two -> m (n two)")
    postact = silu * up
    return grad_preact, postact


def rmsnorm_rstd(x: torch.Tensor, eps: float) -> torch.Tensor:
    # the norm's row scale 1 / sqrt(mean(x^2) + eps)
    square = torch.square(x)
    mean_square = reduce(square, "m n -> m", "mean")
    return torch.rsqrt(mean_square + eps)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # the model's norm, held in x's dtype
    rstd = rmsnorm_rstd(x=x, eps=eps)
    rstd = rearrange(rstd, "m -> m 1")
    normalized = x * rstd * weight
    return normalized.to(dtype=x.dtype)


def drmsnorm(
    pre: torch.Tensor,
    grad: torch.Tensor,
    weight: torch.Tensor,
    ZdZ: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # rmsnorm's vjp in closed form, given the grad on its output and its row dot ZdZ = sum(Z * dZ) / N, which the
    # kernel upstream forms from the next GEMM's own output Z and cotangent dZ; then the grad on weight, summed over
    # rows, and the norm's output recomputed for the next GEMM's wgrad. Pinned to fp64 autograd in test_refs
    rstd = rearrange(rstd, "m -> m 1")
    normed = pre * rstd
    grad_normed = grad * weight
    ZdZ = rearrange(ZdZ, "m -> m 1")
    row_dot_term = normed * ZdZ
    grad_difference = grad_normed - row_dot_term
    grad_pre_unrounded = grad_difference * rstd
    grad_pre = grad_pre_unrounded.to(dtype=pre.dtype)
    grad_weight = reduce(grad * normed, "m n -> n", "sum")
    norm_out = normed * weight
    return grad_pre, grad_weight, norm_out.to(dtype=pre.dtype)
