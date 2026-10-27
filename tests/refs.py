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
    grad_pre = grad_difference * rstd
    grad_weight = reduce(grad * normed, "m n -> n", "sum")
    norm_out = normed * weight
    return grad_pre.to(dtype=pre.dtype), grad_weight, norm_out.to(dtype=pre.dtype)


def rope_tables(positions: torch.Tensor, frequencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # each row's (cos, sin) per rotation pair, (M, N / 2), from the table CODA's ops take: pair j's inv_freq / 2 pi
    # split into an fp32 (hi, lo) at columns (2j, 2j + 1), so the angle is 2 pi pos (hi + lo), in the operands'
    # dtype. In fp64 it is the float-float angle the kernels form; in fp32 lo drops out and the product rounds, as a
    # model's fp32 table does
    hi, lo = rearrange(frequencies, "(p two) -> two p", two=2)
    two_pi = 2 * math.pi
    positions = rearrange(positions, "m -> m 1")
    turns = hi + lo
    rotations = positions * turns
    angles = two_pi * rotations
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def rope_angles(positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    # (M, head_dim / 2) radians from what a model holds, its positions and fp32 inv_freq: their product is exact in
    # fp64 (24 bits by 24), the angle CODA's float-float table carries past fp32, read by the truth and the ideal
    positions_fp64 = positions.to(dtype=torch.float64)
    positions_column = rearrange(positions_fp64, "m -> m 1")
    inv_freq_fp64 = inv_freq.to(dtype=torch.float64)
    inv_freq_row = rearrange(inv_freq_fp64, "p -> 1 p")
    return positions_column * inv_freq_row


def head_mean_square(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    # each head's mean of squares, (M, heads): the statistic a qk norm reads
    heads = rearrange(x, "m (h d) -> m h d", d=head_dim)
    square = torch.square(heads)
    return reduce(square, "m h d -> m h", "mean")


def qknorm(
    x: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
) -> torch.Tensor:
    # each head of x = [Q | K] times rsqrt(head_mean_sq + eps) and its half of gamma, scaled by the statistic and
    # back in x's dtype, as a model's norm returns it
    heads = rearrange(x, "m (h d) -> m h d", d=head_dim)
    rstd_per_head = torch.rsqrt(head_mean_sq + eps)
    rstd = rearrange(rstd_per_head, "m h -> m h 1")
    weights = _head_weights(gamma=gamma, head_dim=head_dim, num_heads_q=num_heads_q, num_heads_k=num_heads_k)
    normed_heads = heads * rstd * weights
    normed = rearrange(normed_heads, "m h d -> m (h d)")
    return normed.to(dtype=x.dtype)
