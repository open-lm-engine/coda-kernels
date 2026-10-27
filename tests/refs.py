import math
import torch
from einops import rearrange, reduce, repeat


# ---- primitives


def swiglu(preact: torch.Tensor) -> torch.Tensor:
    # gate and up in adjacent columns (2i, 2i + 1), the pairing CODA's gated epilogue emits. In bf16, eager rounds
    # silu(gate) before the product, as a model's eager swiglu does, and a compiled one does not
    pairs = rearrange(preact, "m (n two) -> m n two", two=2)
    gate, up = torch.unbind(pairs, dim=-1)
    silu = torch.nn.functional.silu(gate)
    return silu * up


swiglu_bench_interleaved = swiglu


def swiglu_bench(preact: torch.Tensor) -> torch.Tensor:
    # gate and up as the contiguous halves [gate | up] of a GEMM over a model's packed [W_gate; W_up]
    half = preact.shape[-1] // 2
    gate, up = torch.split(preact, (half, half), dim=-1)
    silu = torch.nn.functional.silu(gate)
    return silu * up


def dswiglu(preact: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # swiglu's vjp in closed form, interleaved like preact, and the postact it recomputes on the way: closed so that
    # it compiles under fullgraph as torch's arm, and pinned to fp64 autograd in test_refs
    pairs = rearrange(preact, "m (n two) -> m n two", two=2)
    gate, up = torch.unbind(pairs, dim=-1)
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


dswiglu_bench_interleaved = dswiglu


def dswiglu_bench(preact: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # dswiglu's closed form on the halves, its grads concatenated as the split preact holds them
    half = preact.shape[-1] // 2
    gate, up = torch.split(preact, (half, half), dim=-1)
    sigmoid = torch.nn.functional.sigmoid(gate)
    silu = gate * sigmoid
    grad_silu = grad * up
    one_minus_sigmoid = 1 - sigmoid
    silu_one_minus_sigmoid = silu * one_minus_sigmoid
    dsilu = sigmoid + silu_one_minus_sigmoid
    grad_gate = grad_silu * dsilu
    grad_up = grad * silu
    grad_preact = torch.cat((grad_gate, grad_up), dim=-1)
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
    positions_fp64 = positions.to(dtype=torch.float64)
    positions_column = rearrange(positions_fp64, "m -> m 1")
    inv_freq_fp64 = inv_freq.to(dtype=torch.float64)
    inv_freq_row = rearrange(inv_freq_fp64, "p -> 1 p")
    return positions_column * inv_freq_row


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


rope_bench_interleaved = rope


def rope_bench(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_dim: int) -> torch.Tensor:
    # the split layout a model's rotate_half uses: each head's halves (x0, x1) at columns (j, j + head_dim / 2)
    # rotated by theta_j, (x0 cos - x1 sin, x1 cos + x0 sin); -sin rotates by -theta_j, the transpose. The
    # (M, pairs) tables, all heads' or one head's, broadcast across the heads and are cast to x's dtype, so x sets
    # the precision
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
    rstd_per_head = rearrange(rstd_per_head, "m h -> m h 1")
    # gamma is [gamma_q | gamma_k]: its q half weights the first num_heads_q heads, its k half the rest
    gamma_q, gamma_k = torch.split(gamma, (head_dim, head_dim))
    weight_q = repeat(gamma_q, "d -> h d", h=num_heads_q)
    weight_k = repeat(gamma_k, "d -> h d", h=num_heads_k)
    weights = torch.cat((weight_q, weight_k))
    normed_heads = heads * rstd_per_head * weights
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
    # qknorm's vjp in closed form at the forward's statistic: with r = rsqrt(head_mean_sq + eps) and
    # dr = sum(gamma dn z) per head, dz = r gamma dn - r^3 z dr / head_dim, and dgamma sums dn z r over rows and
    # each half's heads. Also the heads' share of an input norm's row dot sum(z dz), in its closed form eps r^3 dr:
    # summed directly over a z rounded after the statistic, it misses by ~1000 eps
    heads_dn = rearrange(dn, "m (h d) -> m h d", d=head_dim)
    heads_z = rearrange(z, "m (h d) -> m h d", d=head_dim)
    rstd_per_head = torch.rsqrt(head_mean_sq + eps)
    rstd_per_head = rearrange(rstd_per_head, "m h -> m h 1")
    # gamma is [gamma_q | gamma_k]: its q half weights the first num_heads_q heads, its k half the rest
    gamma_q, gamma_k = torch.split(gamma, (head_dim, head_dim))
    weight_q = repeat(gamma_q, "d -> h d", h=num_heads_q)
    weight_k = repeat(gamma_k, "d -> h d", h=num_heads_k)
    weights = torch.cat((weight_q, weight_k))
    weighted = heads_dn * weights
    drstd = reduce(weighted * heads_z, "m h d -> m h 1", "sum")
    dz_direct = rstd_per_head * weighted
    rstd_cubed = rstd_per_head**3
    rstd_cubed_z_drstd = rstd_cubed * heads_z * drstd
    dz_statistic = rstd_cubed_z_drstd / head_dim
    dz = dz_direct - dz_statistic
    dgamma_terms = heads_dn * heads_z * rstd_per_head
    columns = reduce(dgamma_terms, "m h d -> h d", "sum")
    columns_q, columns_k = torch.split(columns, (num_heads_q, num_heads_k))
    dgamma_q = reduce(columns_q, "h d -> d", "sum")
    dgamma_k = reduce(columns_k, "h d -> d", "sum")
    dgamma = torch.cat((dgamma_q, dgamma_k))
    share_rstd_cubed = rstd_per_head**3
    zdz_share_terms = eps * share_rstd_cubed * drstd
    zdz_share = reduce(zdz_share_terms, "m h 1 -> m", "sum")
    return rearrange(dz, "m h d -> m (h d)"), dgamma, zdz_share


def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # each row's loss lse - z_t, and the gradient of the summed loss on the logits, softmax - onehot, whose target
    # entry is minus the off-target mass: on a confident row p_t - 1 cancels to 0 even in fp64. A row whose target
    # is ignore_index gets 0 for both
    kept = target != ignore_index
    safe_target = torch.where(condition=kept, input=target, other=0)
    rows = torch.arange(logits.shape[0], device=logits.device)
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits[rows, safe_target]
    losses = lse - target_logits
    dlogits = torch.softmax(logits, dim=-1)
    dlogits[rows, safe_target] = 0
    off_target_mass = reduce(dlogits, "m v -> m", "sum")
    dlogits[rows, safe_target] = -off_target_mass
    losses = torch.where(condition=kept, input=losses, other=0)
    kept = rearrange(kept, "m -> m 1")
    dlogits = torch.where(condition=kept, input=dlogits, other=0)
    return losses, dlogits
