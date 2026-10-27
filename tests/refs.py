import math
import torch
from enum import Enum
from einops import rearrange, reduce, repeat
from quack.gemm_interface import gemm as quack_gemm


class PairLayout(Enum):
    interleaved = "interleaved"
    split = "split"


# ---- primitives


def swiglu(preact: torch.Tensor) -> torch.Tensor:
    # gate and up in adjacent columns (2i, 2i + 1), the pairing CODA's gated epilogue emits. In bf16, eager rounds
    # silu(gate) before the product, as a model's eager swiglu does, and a compiled one does not
    pairs = rearrange(preact, "m (n two) -> m n two", two=2)
    gate, up = torch.unbind(pairs, dim=-1)
    silu = torch.nn.functional.silu(gate)
    return silu * up


def swiglu_bench(preact: torch.Tensor, layout: PairLayout = PairLayout.interleaved) -> torch.Tensor:
    # swiglu in either layout, split with gate and up the contiguous halves [gate | up] of a GEMM over a model's
    # packed [W_gate; W_up]: the forward whose autograd the dswiglu_backward benchmark times
    if layout == PairLayout.interleaved:
        return swiglu(preact=preact)
    assert layout == PairLayout.split
    half = preact.shape[-1] // 2
    gate, up = torch.split(preact, (half, half), dim=-1)
    silu = torch.nn.functional.silu(gate)
    return silu * up


def dswiglu(preact: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # swiglu's vjp dpreact in closed form, interleaved like preact, and the postact it recomputes on the way: closed
    # so that it compiles under fullgraph as torch's arm, and pinned to fp64 autograd in test_refs
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
    # dswiglu in either layout, split on the halves, dgate and dup concatenated as the split preact holds them
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


def rmsnorm_rstd(x: torch.Tensor, eps: float) -> torch.Tensor:
    # the norm's row scale 1 / sqrt(mean(x^2) + eps)
    square = torch.square(x)
    mean_square = reduce(square, "m n -> m", "mean")
    return torch.rsqrt(mean_square + eps)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # the model's norm
    rstd = rmsnorm_rstd(x=x, eps=eps)
    rstd = rearrange(rstd, "m -> m 1")
    x_rstd = x * rstd
    return x_rstd * weight


def drmsnorm(
    pre: torch.Tensor,
    grad: torch.Tensor,
    weight: torch.Tensor,
    ZdZ: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # rmsnorm's vjp in closed form, dpre, given grad on its output and its row dot ZdZ = sum(Z dZ) / N, which the
    # kernel upstream forms from the next GEMM's own output Z and cotangent dZ; then dweight, summed over rows, and
    # the norm's output normed, recomputed for the next GEMM's wgrad. Pinned to fp64 autograd in test_refs
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
    return dpre, dweight, normed


def rope_tables(positions: torch.Tensor, frequencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # each row's (cos, sin) per rotation pair, (M, N / 2), from the table CODA's ops take: pair j's inv_freq / 2 pi
    # split into an fp32 (hi, lo) at columns (2j, 2j + 1), so the angle is 2 pi pos (hi + lo), in the operands'
    # dtype. In fp64 it is the float-float angle the kernels form; in fp32 lo drops out and the product rounds, as a
    # model's fp32 table does
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


def rope_angles(positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    # (M, head_dim / 2) radians from what a model holds, its positions and fp32 inv_freq: their product is exact in
    # fp64 (24 bits by 24), the angle CODA's float-float table carries past fp32, read by the truth and the ideal
    positions = positions.to(dtype=torch.float64)
    positions = rearrange(positions, "m -> m 1")
    inv_freq = inv_freq.to(dtype=torch.float64)
    inv_freq = rearrange(inv_freq, "p -> 1 p")
    return positions * inv_freq


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # CODA's layout: each adjacent pair (x0, x1) at columns (2j, 2j + 1) rotated by theta_j,
    # (x0 cos - x1 sin, x1 cos + x0 sin); -sin rotates by -theta_j, the transpose. The (M, pairs) tables, all heads'
    # or one head's, broadcast across the heads and are cast to x's dtype, so x sets the precision
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
    # rope in either layout, split as a model's rotate_half writes it: each head's halves (x0, x1) at columns
    # (j, j + head_dim / 2) rotated by theta_j, (x0 cos - x1 sin, x1 cos + x0 sin). The (M, pairs) tables, all
    # heads' or one head's, broadcast across the heads and are cast to x's dtype, so x sets the precision
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


def head_mean_square(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    # each head's mean of squares, (M, heads): the statistic a qk norm reads
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
    # gamma is [gamma_q | gamma_k]: its q half weights the first num_heads_q heads, its k half the rest
    gamma_q, gamma_k = torch.split(gamma, (head_dim, head_dim))
    gamma_q = repeat(gamma_q, "d -> h d", h=num_heads_q)
    gamma_k = repeat(gamma_k, "d -> h d", h=num_heads_k)
    gamma_heads = torch.cat((gamma_q, gamma_k))
    x_rstd = x_heads * rstd_per_head
    normed_heads = x_rstd * gamma_heads
    return rearrange(normed_heads, "m h d -> m (h d)")


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
    # qknorm's vjp in closed form at the forward's statistic: per head, with rstd = rsqrt(head_mean_sq + eps) and
    # drstd = sum(dn gamma z), dz = rstd dn gamma - rstd^3 z drstd / head_dim, and dgamma sums dn z rstd over rows
    # and each half's heads. Also zdz_qk, the q and k heads' part of an input norm's row dot sum(z dz), in its
    # closed form eps rstd^3 drstd: summed directly over a z rounded after the statistic, it misses by ~1000 eps
    dn_heads = rearrange(dn, "m (h d) -> m h d", d=head_dim)
    z_heads = rearrange(z, "m (h d) -> m h d", d=head_dim)
    rstd_per_head = torch.rsqrt(head_mean_sq + eps)
    rstd_per_head = rearrange(rstd_per_head, "m h -> m h 1")
    # gamma is [gamma_q | gamma_k]: its q half weights the first num_heads_q heads, its k half the rest
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
    # recomputed, not reused: reusing the rstd_cubed above changes autograd's rounding of dhead_mean_sq
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
    # each row's loss lse - target_logits, and the gradient of the summed loss on the logits, softmax - onehot,
    # whose target entry dlogits_target is minus the off-target mass: on a confident row p_t - 1 cancels to 0 even
    # in fp64. A row whose target is ignore_index gets 0 for both
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


def mm(A: torch.Tensor, B: torch.Tensor, coda_accumulator: bool = False) -> torch.Tensor:
    # every reference's GEMM, A @ B with B (K, N), or a batch of them. By default torch's, in the operands' dtype:
    # what a model runs, and with fp64 operands the truth (on a CPU too). With coda_accumulator, CODA's fp32
    # accumulator of the 16-bit product: quack's GEMM never splitting K, so it sums K as CODA's mainloop does, bit
    # for bit; untuned, as every config sums K in the same order and gives the same bits. fp8's is refs_fp8's
    if not coda_accumulator:
        return torch.mm(A, B)
    return quack_gemm(A=A, B=B, out_dtype=torch.float32, tuned=False, split_k=1)


def gemm_scalar_scale(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: torch.Tensor,
    coda_accumulator: bool = False,
) -> torch.Tensor:
    # alpha is a one-element fp32 tensor on the device, the cross-entropy backward's dloss, scaling the accumulator
    product = mm(A=A, B=B, coda_accumulator=coda_accumulator)
    return product * alpha


def gemm_scalar_scale_bench(A: torch.Tensor, B: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    # an alpha of shape (1,) promotes a bf16 product where a 0-d one does not, so it goes back to the operands'
    # dtype, where CODA stores it
    product = mm(A=A, B=B)
    scaled = product * alpha
    return scaled.to(dtype=A.dtype)


def gemm_sigmoid(A: torch.Tensor, B: torch.Tensor, coda_accumulator: bool = False) -> torch.Tensor:
    product = mm(A=A, B=B, coda_accumulator=coda_accumulator)
    return torch.nn.functional.sigmoid(product)


def linear_sigmoid(x: torch.Tensor, weight: torch.Tensor, coda_accumulator: bool = False) -> torch.Tensor:
    # weight is (N, K), as nn.Linear holds it; out_dtype is CODA's storage choice, not the op's meaning. Autograd of
    # this is the backward's reference
    return gemm_sigmoid(A=x, B=weight.mT, coda_accumulator=coda_accumulator)


# ---- swiglu


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # B is (K, 2N), gate and up in adjacent columns
    preact = mm(A=A, B=B, coda_accumulator=coda_accumulator)
    postact = swiglu(preact=preact)
    return preact, postact


def gemm_swiglu_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    layout: PairLayout = PairLayout.interleaved,
) -> tuple[torch.Tensor, torch.Tensor]:
    # B is (K, 2N), gate and up in adjacent columns, or split as the contiguous halves of its columns. In bf16 the
    # GEMM rounds its output once, as torch's does
    preact = mm(A=A, B=B)
    postact = swiglu_bench(preact=preact, layout=layout)
    return preact, postact


def gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # the input norm's row scale, deferred onto the accumulator, preact stored scaled and postact from it
    product = mm(A=A, B=B, coda_accumulator=coda_accumulator)
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
    # the down projection's dgrad through swiglu: B arrives (N, K), the block's w2.mT, so A @ B^T is dpostact, the
    # postact's cotangent; then swiglu's backward against the stored preact Z, the row dot ZdZ = sum(Z dZ), and the
    # postact recomputed for the down projection's wgrad
    dpostact = mm(A=A, B=B.mT, coda_accumulator=coda_accumulator)
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
    scale: float | None = None,
    layout: PairLayout = PairLayout.interleaved,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ZdZ in the model's dtype: it spans the whole row, so only dZ's layout differs
    dpostact = mm(A=A, B=B.mT)
    dZ, postact = _dswiglu_bench(preact=Z, grad=dpostact, layout=layout)
    Z_dZ = Z * dZ
    ZdZ = reduce(Z_dZ, "m n -> m", "sum")
    if scale is not None:
        ZdZ = ZdZ * scale
    return dZ, ZdZ, postact


def dswiglu_backward(preact: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    # swiglu's vjp alone, as the elementwise op returns it
    dpreact, _ = dswiglu(preact=preact, grad=grad_out)
    return dpreact


def dswiglu_backward_bench(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
    layout: PairLayout = PairLayout.interleaved,
) -> torch.Tensor:
    dpreact, _ = _dswiglu_bench(preact=preact, grad=grad_out, layout=layout)
    return dpreact


def linear_swiglu(x: torch.Tensor, weight: torch.Tensor, coda_accumulator: bool = False) -> torch.Tensor:
    # weight is the packed (2N, K) gate_up weight, gate and up rows interleaved; autograd of this is the backward's
    # reference
    _, postact = gemm_swiglu(A=x, B=weight.mT, coda_accumulator=coda_accumulator)
    return postact


def linear_swiglu_bench(
    x: torch.Tensor,
    weight: torch.Tensor,
    layout: PairLayout = PairLayout.interleaved,
) -> torch.Tensor:
    # weight is the (2N, K) gate_up weight, gate and up rows interleaved, or split, packed as [W_gate; W_up]
    _, postact = gemm_swiglu_bench(A=x, B=weight.mT, layout=layout)
    return postact


# ---- rmsnorm


def gemm_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    coda_accumulator: bool = False,
) -> torch.Tensor:
    # the input norm's row scale, deferred onto the accumulator of the GEMM that consumes the norm: B is (K, N)
    product = mm(A=A, B=B, coda_accumulator=coda_accumulator)
    rstd = rearrange(rstd, "m -> m 1")
    return product * rstd


def gemm_rmsnorm_bench(A: torch.Tensor, B: torch.Tensor, rstd: torch.Tensor) -> torch.Tensor:
    # the fp32 rstd promotes a 16-bit product, so the output goes back to the operands' dtype
    product = mm(A=A, B=B)
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
    # a GEMM joining the residual stream C, B (K, N), and the norm after it, partial: post = pre * weight, its rstd
    # left to the GEMM that consumes post (gemm_rmsnorm and its siblings), as a statistic of pre before rounding
    product = mm(A=A, B=B, coda_accumulator=coda_accumulator)
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
    # the dgrad of the GEMM that consumed the norm: B arrives (N, K), the block's w.mT, so A @ B^T is dnormed, the
    # grad on the norm's output, scaled by alpha (the loss scale at the LM head); then the norm's backward against
    # the stored pre and its forward rstd. A dX handed to the op is accumulated into: dX plus this dpre
    dnormed = mm(A=A, B=B.mT, coda_accumulator=coda_accumulator)
    if alpha is not None:
        dnormed = dnormed * alpha
    dpre, dweight, normed = drmsnorm(pre=pre, grad=dnormed, weight=weight, ZdZ=ZdZ, rstd=rstd)
    if dX is None:
        return dpre, dweight, normed
    accumulated = dX + dpre
    return accumulated, dweight, normed


def gemm_residual_partial_rmsnorm_bwd_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    weight: torch.Tensor,
    pre: torch.Tensor,
    ZdZ: torch.Tensor,
    rstd: torch.Tensor,
    alpha: torch.Tensor | None = None,
    dX: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # dpre and the norm's output back in pre's dtype, as a model's autograd hands them on; a dX, the residual's
    # incoming grad as block_post runs the op at the MLP, takes that rounded dpre out of place
    dnormed = mm(A=A, B=B.mT)
    if alpha is not None:
        dnormed = dnormed * alpha
    dpre, dweight, normed = drmsnorm(pre=pre, grad=dnormed, weight=weight, ZdZ=ZdZ, rstd=rstd)
    if dX is None:
        return dpre.to(dtype=pre.dtype), dweight, normed.to(dtype=pre.dtype)
    rounded_dpre = dpre.to(dtype=pre.dtype)
    accumulated = dX + rounded_dpre
    return accumulated, dweight, normed.to(dtype=pre.dtype)


# ---- xsa: each head of the attention output without its component along the head's value


def _dot(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    # a per-head row dot
    first_second = first * second
    return reduce(first_second, "... h d -> ... h 1", "sum")


def xsa_fwd(x: torch.Tensor, v: torch.Tensor, head_dim: int) -> torch.Tensor:
    # z = x - x_projection per head, x_projection = x_projection_scale v with x_projection_scale = <x, v> / <v, v>
    # and no epsilon, as lm-engine's _compute_xsa_output writes it for MHA
    x_heads = rearrange(x, "... (h d) -> ... h d", d=head_dim)
    v_heads = rearrange(v, "... (h d) -> ... h d", d=head_dim)
    x_dot_v = _dot(first=x_heads, second=v_heads)
    v_dot_v = _dot(first=v_heads, second=v_heads)
    x_projection_scale = x_dot_v / v_dot_v
    x_projection = x_projection_scale * v_heads
    z_heads = x_heads - x_projection
    return rearrange(z_heads, "... h d -> ... (h d)")


def xsa_fwd_bench(x: torch.Tensor, v: torch.Tensor, head_dim: int) -> torch.Tensor:
    # z in x's dtype, as lm-engine returns it
    z = xsa_fwd(x=x, v=v, head_dim=head_dim)
    return z.to(dtype=x.dtype)


def dxsa(
    grad: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xsa's vjp in closed form given grad, the grad dz on z, in the operands' dtype, so that a test can take dz from
    # the GEMM's own accumulator. Per head, with xsa_fwd's x_projection_scale and
    # dz_projection_scale = <dz, v> / <v, v>: dx = dz - dz_projection_scale v, and
    # dv = 2 dz_projection_scale x_projection_scale v - (x_projection_scale dz + dz_projection_scale x). Pinned to
    # fp64 autograd in test_refs
    dz_heads = rearrange(grad, "... (h d) -> ... h d", d=head_dim)
    x_heads = rearrange(x, "... (h d) -> ... h d", d=head_dim)
    v_heads = rearrange(v, "... (h d) -> ... h d", d=head_dim)
    v_dot_v = _dot(first=v_heads, second=v_heads)
    x_dot_v = _dot(first=x_heads, second=v_heads)
    x_projection_scale = x_dot_v / v_dot_v
    dz_dot_v = _dot(first=dz_heads, second=v_heads)
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
    # c_proj's dgrad through xsa: A the block's (M, K) dy and B = W^T, the (K, N) c_proj weight arriving (N, K), so
    # dz = A @ B^T = dy W is the grad on z
    dz = mm(A=A, B=B.mT, coda_accumulator=coda_accumulator)
    return dxsa(grad=dz, x=x, v=v, head_dim=head_dim)


def gemm_xsa_bwd_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # dz rounded by the unfused GEMM before the formula, which runs in the model's dtype
    dz = mm(A=A, B=B.mT)
    dx, dv = dxsa(grad=dz, x=x, v=v, head_dim=head_dim)
    return dx.to(dtype=x.dtype), dv.to(dtype=v.dtype)


# ---- cross entropy: the LM head's logits and their loss, its gradient formed in the forward


def _fused_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # the fused op's outputs from its logits, with return_zdz the row dot zdz = sum(logits dlogits) the final norm's
    # backward reads
    losses, dlogits = cross_entropy(logits=logits, target=target, ignore_index=ignore_index)
    zdz = None
    if return_zdz:
        logits_dlogits = logits * dlogits
        zdz = reduce(logits_dlogits, "m n -> m", "sum")
    return losses, dlogits, zdz


def gemm_cross_entropy_fwd_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool = False,
    coda_accumulator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # the logits A @ B, B (K, V). CODA forms zdz over its stored, rounded gradient, so a test reads it with that
    # gradient as dZ
    logits = mm(A=A, B=B, coda_accumulator=coda_accumulator)
    return _fused_cross_entropy(logits=logits, target=target, ignore_index=ignore_index, return_zdz=return_zdz)


def gemm_cross_entropy_fwd_bwd_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # a model never forms the gradient in the forward: this is the LM head's bf16 logits, as its GEMM stores them,
    # then the loss and the gradient its autograd would form
    logits = mm(A=A, B=B)
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
    # the final norm's row scale deferred onto the accumulator: the logits are (A @ B) rstd
    product = mm(A=A, B=B, coda_accumulator=coda_accumulator)
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
    # the final norm's row scale deferred onto the product, the logits back in the model's dtype before the loss
    # reads them
    logits = gemm_rmsnorm_bench(A=A, B=B, rstd=rstd)
    return _cross_entropy_bench(logits=logits, target=target, ignore_index=ignore_index, return_zdz=return_zdz)
