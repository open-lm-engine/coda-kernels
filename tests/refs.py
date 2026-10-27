import math
import torch
from enum import Enum
from einops import rearrange, reduce, repeat
from quack.gemm_interface import gemm as quack_gemm


class PairLayout(Enum):
    # interleaved: CODA's pairs at columns (2j, 2j + 1); split: two side-by-side halves ([gate | up], rotate_half)
    interleaved = "interleaved"
    split = "split"


# ---- primitives


def swiglu(preact: torch.Tensor) -> torch.Tensor:
    # silu(gate) * up, gate and up at adjacent columns (2i, 2i + 1)
    pairs = rearrange(preact, "m (n two) -> m n two", two=2)
    gate, up = torch.unbind(pairs, dim=-1)
    silu = torch.nn.functional.silu(gate)
    return silu * up


def swiglu_bench(
    preact: torch.Tensor,
    layout: PairLayout,
) -> torch.Tensor:
    # swiglu in either layout; split takes gate and up as the halves [gate | up] of a packed [W_gate; W_up] GEMM
    if layout == PairLayout.interleaved:
        return swiglu(preact=preact)
    assert layout == PairLayout.split
    half = preact.shape[-1] // 2
    gate, up = torch.split(preact, (half, half), dim=-1)
    silu = torch.nn.functional.silu(gate)
    return silu * up


def dswiglu(
    preact: torch.Tensor,
    grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # swiglu's vjp dpreact, interleaved like preact, and the recomputed postact
    # closed form so torch's arm compiles under fullgraph
    pairs = rearrange(preact, "m (n two) -> m n two", two=2)
    gate, up = torch.unbind(pairs, dim=-1)
    sigmoid = torch.nn.functional.sigmoid(gate)
    silu = gate * sigmoid
    dsilu = grad * up
    one_minus_sigmoid = 1 - sigmoid
    silu_one_minus_sigmoid = silu * one_minus_sigmoid
    dsilu_dgate = sigmoid + silu_one_minus_sigmoid
    dgate = dsilu * dsilu_dgate
    dup = grad * silu
    dpairs = torch.stack((dgate, dup), dim=-1)
    dpreact = rearrange(dpairs, "m n two -> m (n two)")
    postact = silu * up
    return dpreact, postact


def _dswiglu_bench(
    preact: torch.Tensor,
    grad: torch.Tensor,
    layout: PairLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    # dswiglu in either layout; split returns dpreact as [dgate | dup], like preact
    if layout == PairLayout.interleaved:
        return dswiglu(preact=preact, grad=grad)
    assert layout == PairLayout.split
    half = preact.shape[-1] // 2
    gate, up = torch.split(preact, (half, half), dim=-1)
    sigmoid = torch.nn.functional.sigmoid(gate)
    silu = gate * sigmoid
    dsilu = grad * up
    one_minus_sigmoid = 1 - sigmoid
    silu_one_minus_sigmoid = silu * one_minus_sigmoid
    dsilu_dgate = sigmoid + silu_one_minus_sigmoid
    dgate = dsilu * dsilu_dgate
    dup = grad * silu
    dpreact = torch.cat((dgate, dup), dim=-1)
    postact = silu * up
    return dpreact, postact


def rmsnorm_rstd(
    x: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # each row's 1 / sqrt(mean(x^2) + eps)
    square = torch.square(x)
    mean_square = reduce(square, "m n -> m", "mean")
    return torch.rsqrt(mean_square + eps)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # x * rstd * weight
    rstd = rmsnorm_rstd(x=x, eps=eps)
    rstd = rearrange(rstd, "m -> m 1")
    x_rstd = x * rstd
    normed = x_rstd * weight
    return normed.to(dtype=x.dtype)


def drmsnorm(
    pre: torch.Tensor,
    grad: torch.Tensor,
    weight: torch.Tensor,
    ZdZ: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # rmsnorm's vjp dpre and dweight, and the output normed recomputed for the next GEMM's wgrad; ZdZ is the row
    # dot sum(Z dZ) / N over the next GEMM's output Z
    rstd = rearrange(rstd, "m -> m 1")
    pre_rstd = pre * rstd
    dpre_rstd = grad * weight
    ZdZ = rearrange(ZdZ, "m -> m 1")
    pre_rstd_ZdZ = pre_rstd * ZdZ
    dpre_unscaled = dpre_rstd - pre_rstd_ZdZ
    dpre = dpre_unscaled * rstd
    grad_pre_rstd = grad * pre_rstd
    dweight = reduce(grad_pre_rstd, "m n -> n", "sum")
    normed = pre_rstd * weight
    return dpre.to(dtype=pre.dtype), dweight, normed.to(dtype=pre.dtype)


def rope_tables(
    positions: torch.Tensor,
    frequencies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # (M, N / 2) cos and sin of 2 pi positions (hi + lo), frequencies holding pair j's inv_freq / 2 pi as an fp32
    # (hi, lo) at columns (2j, 2j + 1); fp64 gives the kernels' float-float angle, fp32 a model's table
    pairs = rearrange(frequencies, "(p two) -> p two", two=2)
    hi, lo = torch.unbind(pairs, dim=1)
    two_pi = 2 * math.pi
    positions = rearrange(positions, "m -> m 1")
    turns = hi + lo
    rotations = positions * turns
    angles = two_pi * rotations
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def rope_angles(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    # (M, head_dim / 2) angles positions * inv_freq, in fp64 so the product of the fp32 inputs is exact
    positions = positions.to(dtype=torch.float64)
    positions = rearrange(positions, "m -> m 1")
    inv_freq = inv_freq.to(dtype=torch.float64)
    inv_freq = rearrange(inv_freq, "p -> 1 p")
    return positions * inv_freq


def rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # each pair (x0, x1) at columns (2j, 2j + 1) rotated to (x0 cos - x1 sin, x1 cos + x0 sin), -sin rotating
    # back; the (M, pairs) tables, one head's or all heads', broadcast over heads, and x's dtype sets the precision
    pairs = rearrange(x, "m (h p two) -> m h p two", p=cos.shape[-1], two=2)
    x0, x1 = torch.unbind(pairs, dim=-1)
    cos = cos.to(dtype=x.dtype)
    cos = rearrange(cos, "m p -> m 1 p")
    sin = sin.to(dtype=x.dtype)
    sin = rearrange(sin, "m p -> m 1 p")
    x0_cos = x0 * cos
    x1_sin = x1 * sin
    rotated_x0 = x0_cos - x1_sin
    x1_cos = x1 * cos
    x0_sin = x0 * sin
    rotated_x1 = x1_cos + x0_sin
    rotated = torch.stack((rotated_x0, rotated_x1), dim=-1)
    return rearrange(rotated, "m h p two -> m (h p two)")


def _rope_bench(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    layout: PairLayout,
) -> torch.Tensor:
    # rope in either layout; split pairs each head's halves, columns (j, j + head_dim / 2), as rotate_half does
    if layout == PairLayout.interleaved:
        return rope(x=x, cos=cos, sin=sin)
    assert layout == PairLayout.split
    half = head_dim // 2
    halves = rearrange(x, "m (h two p) -> m h two p", two=2, p=half)
    x0, x1 = torch.unbind(halves, dim=2)
    cos = cos.to(dtype=x.dtype)
    cos = rearrange(cos, "m (h p) -> m h p", p=half)
    sin = sin.to(dtype=x.dtype)
    sin = rearrange(sin, "m (h p) -> m h p", p=half)
    x0_cos = x0 * cos
    x1_sin = x1 * sin
    rotated_x0 = x0_cos - x1_sin
    x1_cos = x1 * cos
    x0_sin = x0 * sin
    rotated_x1 = x1_cos + x0_sin
    rotated = torch.stack((rotated_x0, rotated_x1), dim=2)
    return rearrange(rotated, "m h two p -> m (h two p)")


def head_mean_square(
    x: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    # (M, heads) mean of squares per head, qknorm's statistic
    x_heads = rearrange(x, "m (h d) -> m h d", d=head_dim)
    square = torch.square(x_heads)
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
    # each head of x = [q | k] times its rstd = rsqrt(head_mean_sq + eps) and its half of gamma
    x_heads = rearrange(x, "m (h d) -> m h d", d=head_dim)
    rstd_per_head = torch.rsqrt(head_mean_sq + eps)
    rstd_per_head = rearrange(rstd_per_head, "m h -> m h 1")
    # gamma_q weights the first num_heads_q heads, gamma_k the rest
    gamma_q, gamma_k = torch.split(gamma, (head_dim, head_dim))
    gamma_q = repeat(gamma_q, "d -> h d", h=num_heads_q)
    gamma_k = repeat(gamma_k, "d -> h d", h=num_heads_k)
    gamma_heads = torch.cat((gamma_q, gamma_k))
    x_rstd = x_heads * rstd_per_head
    normed_heads = x_rstd * gamma_heads
    normed = rearrange(normed_heads, "m h d -> m (h d)")
    return normed.to(dtype=x.dtype)


def dqknorm(
    dn: torch.Tensor,
    z: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # qknorm's vjp at the forward's statistic: dgamma, and per head dz = rstd dn gamma - rstd^3 z drstd / head_dim
    # with drstd = sum(dn gamma z); zdz_qk = sum(z dz) over the q and k heads, in closed form eps rstd^3 drstd
    # since a direct sum over a z rounded after the statistic is off by ~1000 eps
    dn_heads = rearrange(dn, "m (h d) -> m h d", d=head_dim)
    z_heads = rearrange(z, "m (h d) -> m h d", d=head_dim)
    rstd_per_head = torch.rsqrt(head_mean_sq + eps)
    rstd_per_head = rearrange(rstd_per_head, "m h -> m h 1")
    # gamma_q weights the first num_heads_q heads, gamma_k the rest
    gamma_q, gamma_k = torch.split(gamma, (head_dim, head_dim))
    gamma_q = repeat(gamma_q, "d -> h d", h=num_heads_q)
    gamma_k = repeat(gamma_k, "d -> h d", h=num_heads_k)
    gamma_heads = torch.cat((gamma_q, gamma_k))
    dn_gamma = dn_heads * gamma_heads
    dn_gamma_z = dn_gamma * z_heads
    drstd_per_head = reduce(dn_gamma_z, "m h d -> m h 1", "sum")
    rstd_dn_gamma = rstd_per_head * dn_gamma
    rstd_cubed = rstd_per_head**3
    rstd_cubed_z = rstd_cubed * z_heads
    rstd_cubed_z_drstd = rstd_cubed_z * drstd_per_head
    rstd_cubed_z_drstd_over_head_dim = rstd_cubed_z_drstd / head_dim
    dz_heads = rstd_dn_gamma - rstd_cubed_z_drstd_over_head_dim
    dn_z = dn_heads * z_heads
    dn_z_rstd = dn_z * rstd_per_head
    dgamma_heads = reduce(dn_z_rstd, "m h d -> h d", "sum")
    dgamma_q, dgamma_k = torch.split(dgamma_heads, (num_heads_q, num_heads_k))
    dgamma_q = reduce(dgamma_q, "h d -> d", "sum")
    dgamma_k = reduce(dgamma_k, "h d -> d", "sum")
    dgamma = torch.cat((dgamma_q, dgamma_k))
    # recomputed: reusing rstd_cubed above changes autograd's rounding of dhead_mean_sq
    rstd_cubed = rstd_per_head**3
    eps_rstd_cubed = eps * rstd_cubed
    zdz_per_head = eps_rstd_cubed * drstd_per_head
    zdz_qk = reduce(zdz_per_head, "m h 1 -> m", "sum")
    dz = rearrange(dz_heads, "m h d -> m (h d)")
    return dz, dgamma, zdz_qk


def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # per-row loss lse - target_logits and dlogits = softmax - onehot, ignore_index rows zeroed; the target entry
    # is minus the off-target mass, since p_t - 1 cancels to 0 on a confident row even in fp64
    kept = target != ignore_index
    safe_target = torch.where(condition=kept, input=target, other=0)
    row_index = torch.arange(logits.shape[0], device=logits.device)
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits[row_index, safe_target]
    losses = lse - target_logits
    dlogits = torch.softmax(logits, dim=-1)
    dlogits[row_index, safe_target] = 0
    off_target_mass = reduce(dlogits, "m v -> m", "sum")
    dlogits_target = -off_target_mass
    dlogits[row_index, safe_target] = dlogits_target
    losses = torch.where(condition=kept, input=losses, other=0)
    kept = rearrange(kept, "m -> m 1")
    dlogits = torch.where(condition=kept, input=dlogits, other=0)
    return losses, dlogits


# ---- gemm


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    coda_accumulator: bool = False,
) -> torch.Tensor:
    # A @ B, B (K, N), batched or not: torch's in the operands' dtype, or with coda_accumulator CODA's fp32
    # accumulator from quack with split_k=1, which sums K in CODA's order bit for bit, untuned since every config
    # gives the same bits
    if not coda_accumulator:
        return torch.matmul(input=A, other=B)
    return quack_gemm(A=A, B=B, out_dtype=torch.float32, tuned=False, split_k=1)


def gemm_symmetric(
    A: torch.Tensor,
    B: torch.Tensor | None = None,
    C: torch.Tensor | None = None,
    D_scale: float | None = None,
    C_scale: float | None = None,
    coda_accumulator: bool = False,
) -> torch.Tensor:
    # D_scale * A B + C_scale * C, B = A^T by default: Newton-Schulz's X X^T, then b A + c A A; a scale left None is
    # not multiplied
    if B is None:
        B = A.mT
    D = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    if D_scale is not None:
        D = D_scale * D
    if C is None:
        return D
    if C_scale is not None:
        C = C_scale * C
    return D + C


def gemm_scalar_scale(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: torch.Tensor,
    coda_accumulator: bool = False,
) -> torch.Tensor:
    # (A @ B) * alpha, alpha a one-element fp32 device tensor (the cross-entropy backward's dloss)
    product = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    return product * alpha


def gemm_scalar_scale_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    # a (1,) alpha promotes a bf16 product (a 0-d one would not), so cast back to the operands' dtype
    product = gemm(A=A, B=B)
    scaled = product * alpha
    return scaled.to(dtype=A.dtype)


def gemm_add_dual_precision(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    C_scale: float | None = None,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Newton-Schulz's X <- a X + B X over an fp32 master: out = A B + C_scale * C in C's dtype, and out rounded to
    # the operands' dtype, the next iteration's operand, a rounding kept since CODA's contract is that operand bit
    # for bit; without coda_accumulator a bf16 A B rounds before the add
    product = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    if C_scale is not None:
        C = C_scale * C
    out = product + C
    return out, out.to(dtype=A.dtype)


def gemm_sigmoid(
    A: torch.Tensor,
    B: torch.Tensor,
    coda_accumulator: bool = False,
) -> torch.Tensor:
    product = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    return torch.nn.functional.sigmoid(product)


def linear_sigmoid(
    x: torch.Tensor,
    weight: torch.Tensor,
    coda_accumulator: bool = False,
) -> torch.Tensor:
    # sigmoid(x @ weight^T), weight (N, K) as nn.Linear holds it; CODA's out_dtype only picks storage, so not an
    # argument here; its autograd is the backward's reference
    return gemm_sigmoid(A=x, B=weight.mT, coda_accumulator=coda_accumulator)


# ---- swiglu


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # B is (K, 2N), gate and up in adjacent columns
    preact = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    postact = swiglu(preact=preact)
    return preact, postact


def gemm_swiglu_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    layout: PairLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    # B is (K, 2N), gate and up columns interleaved or split into halves
    preact = gemm(A=A, B=B)
    postact = swiglu_bench(preact=preact, layout=layout)
    return preact, postact


def gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # preact = (A @ B) rstd, the input norm's row scale deferred onto the accumulator, and postact from it
    product = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    rstd = rearrange(rstd, "m -> m 1")
    preact = product * rstd
    postact = swiglu(preact=preact)
    return preact, postact


def gemm_swiglu_bwd_zdz(
    A: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    scale: float | None = None,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # the down projection's dgrad through swiglu: dpostact = A @ B^T, B (N, K), then dZ against the stored
    # preact Z, ZdZ = scale * sum(Z dZ), and postact recomputed for the wgrad
    dpostact = gemm(A=A, B=B.mT, coda_accumulator=coda_accumulator)
    dZ, postact = dswiglu(preact=Z, grad=dpostact)
    Z_dZ = Z * dZ
    ZdZ = reduce(Z_dZ, "m n -> m", "sum")
    if scale is not None:
        ZdZ = ZdZ * scale
    return dZ, ZdZ, postact


def gemm_swiglu_bwd_zdz_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    layout: PairLayout,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ZdZ in the model's dtype, the same whole-row sum in either layout
    dpostact = gemm(A=A, B=B.mT)
    dZ, postact = _dswiglu_bench(preact=Z, grad=dpostact, layout=layout)
    Z_dZ = Z * dZ
    ZdZ = reduce(Z_dZ, "m n -> m", "sum")
    if scale is not None:
        ZdZ = ZdZ * scale
    return dZ, ZdZ, postact


def dswiglu_backward(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    # swiglu's vjp dpreact alone
    dpreact, _ = dswiglu(preact=preact, grad=grad_out)
    return dpreact


def dswiglu_backward_bench(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
    layout: PairLayout,
) -> torch.Tensor:
    dpreact, _ = _dswiglu_bench(preact=preact, grad=grad_out, layout=layout)
    return dpreact


def linear_swiglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    coda_accumulator: bool = False,
) -> torch.Tensor:
    # weight (2N, K), gate and up rows interleaved; its autograd is the backward's reference
    _, postact = gemm_swiglu(A=x, B=weight.mT, coda_accumulator=coda_accumulator)
    return postact


def linear_swiglu_bench(
    x: torch.Tensor,
    weight: torch.Tensor,
    layout: PairLayout,
) -> torch.Tensor:
    # weight (2N, K), gate and up rows interleaved, or split as [W_gate; W_up]
    _, postact = gemm_swiglu_bench(A=x, B=weight.mT, layout=layout)
    return postact


# ---- rmsnorm


def gemm_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    coda_accumulator: bool = False,
) -> torch.Tensor:
    # (A @ B) rstd, B (K, N): the input norm's row scale deferred onto the accumulator
    product = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    rstd = rearrange(rstd, "m -> m 1")
    return product * rstd


def gemm_rmsnorm_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
) -> torch.Tensor:
    # the fp32 rstd promotes a 16-bit product, so cast back to the operands' dtype
    product = gemm(A=A, B=B)
    rstd = rearrange(rstd, "m -> m 1")
    scaled = product * rstd
    return scaled.to(dtype=A.dtype)


def gemm_residual_partial_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pre = A @ B + C, B (K, N), the partial norm post = pre * weight, and rstd of the unrounded pre, which the GEMM
    # consuming post applies
    product = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    pre = product + C
    post = pre * weight
    rstd = rmsnorm_rstd(x=pre, eps=eps)
    return pre, post, rstd


def gemm_residual_partial_rmsnorm_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    weight: torch.Tensor,
    pre: torch.Tensor,
    ZdZ: torch.Tensor,
    rstd: torch.Tensor,
    alpha: torch.Tensor | None = None,
    dX: torch.Tensor | None = None,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # dnormed = A @ B^T, B (N, K), the consuming GEMM's dgrad, times alpha (the LM head's loss scale); then
    # drmsnorm against the stored pre and rstd, dpre added to dX when given
    dnormed = gemm(A=A, B=B.mT, coda_accumulator=coda_accumulator)
    if alpha is not None:
        dnormed = dnormed * alpha
    dpre, dweight, normed = drmsnorm(pre=pre, grad=dnormed, weight=weight, ZdZ=ZdZ, rstd=rstd)
    if dX is None:
        return dpre, dweight, normed
    accumulated = dX + dpre
    return accumulated, dweight, normed


# ---- xsa: each head of the attention output without its component along the head's value


def _dot_per_head(
    left_heads: torch.Tensor,
    right_heads: torch.Tensor,
) -> torch.Tensor:
    # the dot product of two (..., h, d) per-head views over each head's d, kept (..., h, 1) to broadcast
    left_right = left_heads * right_heads
    return reduce(left_right, "... h d -> ... h 1", "sum")


def xsa_fwd(
    x: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    # per head, z = x - x_projection_scale v with x_projection_scale = <x, v> / <v, v>, no epsilon
    x_heads = rearrange(x, "... (h d) -> ... h d", d=head_dim)
    v_heads = rearrange(v, "... (h d) -> ... h d", d=head_dim)
    x_dot_v = _dot_per_head(left_heads=x_heads, right_heads=v_heads)
    v_dot_v = _dot_per_head(left_heads=v_heads, right_heads=v_heads)
    x_projection_scale = x_dot_v / v_dot_v
    x_projection = x_projection_scale * v_heads
    z_heads = x_heads - x_projection
    z_heads = z_heads.to(dtype=x.dtype)
    return rearrange(z_heads, "... h d -> ... (h d)")


def dxsa(
    grad: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xsa's vjp given grad = dz, in closed form so a test can feed dz from the GEMM's accumulator; per head:
    # dx = dz - dz_projection_scale v, dz_projection_scale = <dz, v> / <v, v>, and
    # dv = 2 dz_projection_scale x_projection_scale v - (x_projection_scale dz + dz_projection_scale x)
    dz_heads = rearrange(grad, "... (h d) -> ... h d", d=head_dim)
    x_heads = rearrange(x, "... (h d) -> ... h d", d=head_dim)
    v_heads = rearrange(v, "... (h d) -> ... h d", d=head_dim)
    v_dot_v = _dot_per_head(left_heads=v_heads, right_heads=v_heads)
    x_dot_v = _dot_per_head(left_heads=x_heads, right_heads=v_heads)
    x_projection_scale = x_dot_v / v_dot_v
    dz_dot_v = _dot_per_head(left_heads=dz_heads, right_heads=v_heads)
    dz_projection_scale = dz_dot_v / v_dot_v
    dz_projection = dz_projection_scale * v_heads
    dx_heads = dz_heads - dz_projection
    two_dz_projection_scale = 2 * dz_projection_scale
    two_dz_projection_scale_x_projection_scale = two_dz_projection_scale * x_projection_scale
    two_dz_projection_scale_x_projection_scale_v = two_dz_projection_scale_x_projection_scale * v_heads
    x_projection_scale_dz = x_projection_scale * dz_heads
    dz_projection_scale_x = dz_projection_scale * x_heads
    x_projection_scale_dz_plus_dz_projection_scale_x = x_projection_scale_dz + dz_projection_scale_x
    dv_heads = two_dz_projection_scale_x_projection_scale_v - x_projection_scale_dz_plus_dz_projection_scale_x
    dx = rearrange(dx_heads, "... h d -> ... (h d)")
    dv = rearrange(dv_heads, "... h d -> ... (h d)")
    return dx, dv


def gemm_xsa_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # the output projection's dgrad through xsa: dz = A @ B^T, B (N, K), then dxsa
    dz = gemm(A=A, B=B.mT, coda_accumulator=coda_accumulator)
    dx, dv = dxsa(grad=dz, x=x, v=v, head_dim=head_dim)
    return dx.to(dtype=x.dtype), dv.to(dtype=v.dtype)


# ---- cross entropy: the LM head's loss, its gradient formed in the forward


def _fused_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # losses, dlogits and, with return_zdz, zdz = sum(logits dlogits) for the final norm's backward

    losses, dlogits = cross_entropy(logits=logits, target=target, ignore_index=ignore_index)
    zdz = None
    if return_zdz:
        logits_dlogits = logits * dlogits
        zdz = reduce(logits_dlogits, "m n -> m", "sum")
    return losses, dlogits, zdz


def _cross_entropy_bench(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # per-row loss and dlogits = softmax - onehot as autograd forms them, in the logits' dtype; ignored rows get 0;
    # target arrives int64, as a model's labels do
    losses = torch.nn.functional.cross_entropy(
        input=logits,
        target=target,
        ignore_index=ignore_index,
        reduction="none",
    )
    kept = target != ignore_index
    safe_target = torch.where(condition=kept, input=target, other=0)
    rows = torch.arange(logits.shape[0], device=logits.device)
    probabilities = torch.nn.functional.softmax(input=logits, dim=-1)
    target_probabilities = probabilities[rows, safe_target]
    target_gradients = target_probabilities - 1
    probabilities[rows, safe_target] = target_gradients
    kept = rearrange(kept, "m -> m 1")
    dlogits = torch.where(condition=kept, input=probabilities, other=0)
    row_dot = None
    if return_zdz:
        row_dot_terms = logits * dlogits
        row_dot = reduce(row_dot_terms, "m v -> m", "sum")
    return losses, dlogits, row_dot


def gemm_cross_entropy_fwd_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool = False,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # logits = A @ B, B (K, V); CODA forms zdz over its rounded dlogits, so a test checks it with those as dZ
    logits = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    return _fused_cross_entropy(logits=logits, target=target, ignore_index=ignore_index, return_zdz=return_zdz)


def gemm_cross_entropy_fwd_bwd_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # the LM head's bf16 logits as a model's GEMM stores them, then the loss and the gradient autograd forms
    logits = gemm(A=A, B=B)
    return _cross_entropy_bench(logits=logits, target=target, ignore_index=ignore_index, return_zdz=return_zdz)


def gemm_rmsnorm_cross_entropy_fwd_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool = False,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # logits = (A @ B) rstd, the final norm's row scale deferred onto the accumulator
    product = gemm(A=A, B=B, coda_accumulator=coda_accumulator)
    rstd = rearrange(rstd, "m -> m 1")
    logits = product * rstd
    return _fused_cross_entropy(logits=logits, target=target, ignore_index=ignore_index, return_zdz=return_zdz)


def gemm_rmsnorm_cross_entropy_fwd_bwd_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # logits = (A @ B) rstd, cast back to the model's dtype before the loss
    logits = gemm_rmsnorm_bench(A=A, B=B, rstd=rstd)
    return _cross_entropy_bench(logits=logits, target=target, ignore_index=ignore_index, return_zdz=return_zdz)


def linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    coda_accumulator: bool = False,
) -> torch.Tensor:
    # weight (V, K) as nn.Linear holds it, target int64; its autograd is the backward's reference
    logits = gemm(A=x, B=weight.mT, coda_accumulator=coda_accumulator)
    return torch.nn.functional.cross_entropy(
        input=logits,
        target=target,
        ignore_index=ignore_index,
        reduction=reduction,
    )
