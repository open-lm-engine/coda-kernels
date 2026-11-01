import math
import functools
import torch
from enum import Enum
from einops import rearrange, reduce, repeat
from quack.gemm_interface import gemm as quack_gemm


class PairLayout(Enum):
    # pairs at (2j, 2j + 1), or split into two halves
    interleaved = "interleaved"
    split = "split"


# ---- primitives


def swiglu(
    preact: torch.Tensor,
    layout: PairLayout,
) -> torch.Tensor:
    # silu(gate) * up, interleaved or split as [gate | up]
    if layout == PairLayout.interleaved:
        pairs = rearrange(preact, "... (n two) -> ... n two", two=2)
        gate, up = torch.unbind(pairs, dim=-1)
    else:
        assert layout == PairLayout.split
        half = preact.shape[-1] // 2
        gate, up = torch.split(preact, (half, half), dim=-1)

    silu = torch.nn.functional.silu(gate)
    return silu * up


def dswiglu(
    preact: torch.Tensor,
    grad: torch.Tensor,
    layout: PairLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    # hand-written derivative: autograd.grad would break fullgraph
    if layout == PairLayout.interleaved:
        pairs = rearrange(preact, "m (n two) -> m n two", two=2)
        gate, up = torch.unbind(pairs, dim=-1)
    else:
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

    if layout == PairLayout.interleaved:
        dpairs = torch.stack((dgate, dup), dim=-1)
        dpreact = rearrange(dpairs, "m n two -> m (n two)")
    else:
        dpreact = torch.cat((dgate, dup), dim=-1)

    postact = silu * up
    return dpreact, postact


def rmsnorm_rstd(
    x: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # each row's 1 / sqrt(mean(x^2) + eps)
    square = torch.square(x)
    mean_square = reduce(square, "... n -> ...", "mean")
    return torch.rsqrt(mean_square + eps)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # x * rstd * weight
    rstd = rmsnorm_rstd(
        x=x,
        eps=eps,
    )
    rstd = rearrange(rstd, "... -> ... 1")
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
    # rmsnorm's vjp, and normed recomputed for the next
    # wgrad; ZdZ = sum(Z dZ) / N over the next GEMM's output
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
    # (M, N / 2) cos and sin of 2 pi positions (hi + lo), where
    # frequencies holds each pair's inv_freq / 2 pi as fp32 (hi, lo);
    # in fp64 this is the angle CODA's kernels form
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
    # (M, head_dim / 2) positions * inv_freq: the truth's angle,
    # cast here so it is exact whatever the caller passes
    positions = positions.to(dtype=torch.float64)
    positions = rearrange(positions, "m -> m 1")
    inv_freq = inv_freq.to(dtype=torch.float64)
    inv_freq = rearrange(inv_freq, "p -> 1 p")
    return positions * inv_freq


def rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    layout: PairLayout,
) -> torch.Tensor:
    # (x0, x1) -> (x0 cos - x1 sin, x1 cos + x0 sin),
    # -sin rotating back; split pairs each head's halves
    if layout == PairLayout.interleaved:
        pairs = rearrange(
            x,
            "... (h p two) -> ... h p two",
            p=cos.shape[-1],
            two=2,
        )
        x0, x1 = torch.unbind(pairs, dim=-1)
        cos = cos.to(dtype=x.dtype)
        cos = rearrange(cos, "... p -> ... 1 p")
        sin = sin.to(dtype=x.dtype)
        sin = rearrange(sin, "... p -> ... 1 p")
    else:
        assert layout == PairLayout.split
        half = head_dim // 2
        halves = rearrange(
            x,
            "... (h two p) -> ... h two p",
            two=2,
            p=half,
        )
        x0, x1 = torch.unbind(halves, dim=-2)
        cos = cos.to(dtype=x.dtype)
        cos = rearrange(cos, "... (h p) -> ... h p", p=half)
        sin = sin.to(dtype=x.dtype)
        sin = rearrange(sin, "... (h p) -> ... h p", p=half)

    x0_cos = x0 * cos
    x1_sin = x1 * sin
    rotated_x0 = x0_cos - x1_sin
    x1_cos = x1 * cos
    x0_sin = x0 * sin
    rotated_x1 = x1_cos + x0_sin

    if layout == PairLayout.interleaved:
        rotated = torch.stack((rotated_x0, rotated_x1), dim=-1)
        return rearrange(rotated, "... h p two -> ... (h p two)")
    else:
        rotated = torch.stack((rotated_x0, rotated_x1), dim=-2)
        return rearrange(rotated, "... h two p -> ... (h two p)")


def head_mean_square(
    x: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    # (..., heads) mean square per head, qknorm's statistic
    x_heads = rearrange(x, "... (h d) -> ... h d", d=head_dim)
    square = torch.square(x_heads)
    return reduce(square, "... h d -> ... h", "mean")


def qknorm(
    x: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
) -> torch.Tensor:
    # each head times rsqrt(head_mean_sq + eps) and its gamma
    x_heads = rearrange(x, "... (h d) -> ... h d", d=head_dim)
    rstd_per_head = torch.rsqrt(head_mean_sq + eps)
    rstd_per_head = rearrange(rstd_per_head, "... h -> ... h 1")
    # gamma_q weights the first num_heads_q heads, gamma_k the rest
    gamma_q, gamma_k = torch.split(gamma, (head_dim, head_dim))
    gamma_q = repeat(gamma_q, "d -> h d", h=num_heads_q)
    gamma_k = repeat(gamma_k, "d -> h d", h=num_heads_k)
    gamma_heads = torch.cat((gamma_q, gamma_k))
    x_rstd = x_heads * rstd_per_head
    normed_heads = x_rstd * gamma_heads
    normed = rearrange(normed_heads, "... h d -> ... (h d)")
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
    # qknorm's vjp at the forward's statistic, per head:
    # dz = rstd dn gamma - rstd^3 z drstd / head_dim,
    # drstd = sum(dn gamma z); zdz_qk = eps rstd^3 drstd
    # in closed form, as a direct sum is ~1000 eps off
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
    # dlogits = softmax - onehot, rows with target == ignore_index
    # zeroed; the target op is minus the off-target mass, as
    # p_t - 1 rounds to 0 at p_t ~ 1
    non_ignored = target != ignore_index
    safe_target = torch.where(
        condition=non_ignored,
        input=target,
        other=0,
    )
    row_index = torch.arange(logits.shape[0], device=logits.device)
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits[row_index, safe_target]
    losses = lse - target_logits
    dlogits = torch.softmax(logits, dim=-1)
    dlogits[row_index, safe_target] = 0
    off_target_mass = reduce(dlogits, "m v -> m", "sum")
    dlogits_target = -off_target_mass
    dlogits[row_index, safe_target] = dlogits_target
    losses = torch.where(
        condition=non_ignored,
        input=losses,
        other=0,
    )
    non_ignored = rearrange(non_ignored, "m -> m 1")
    dlogits = torch.where(
        condition=non_ignored,
        input=dlogits,
        other=0,
    )
    return losses, dlogits


# ---- gemm


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    use_quack: bool = False,
) -> torch.Tensor:
    # with use_quack, the fp32 accumulator CODA's epilogues read;
    # untuned, since at split_k=1 every config gives the same bits
    if not use_quack:
        # B is the transposed view of an (N, K) weight, as nn.Linear stores
        # it, so torch.matmul runs the same cuBLAS kernel as F.linear, with
        # no copy, in eager and compiled; torch pays no layout cost
        return torch.matmul(
            input=A,
            other=B,
        )
    else:
        return quack_gemm(
            A=A,
            B=B,
            out_dtype=torch.float32,
            tuned=False,
            split_k=1,
        )


def gemm_symmetric(
    A: torch.Tensor,
    B: torch.Tensor | None = None,
    C: torch.Tensor | None = None,
    D_scale: float | None = None,
    C_scale: float | None = None,
    use_quack: bool = False,
) -> torch.Tensor:
    # D_scale A B + C_scale C, B defaulting to A^T
    if B is None:
        B = A.mT
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    if D_scale is not None:
        D = D_scale * D
    if C is None:
        return D
    else:
        if C_scale is not None:
            C = C_scale * C
        return D + C


def gemm_scalar_scale(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: torch.Tensor,
    use_quack: bool = False,
) -> torch.Tensor:
    # alpha is a 0-d fp32 tensor on the GPU; being 0-d, it does not change
    # the product's dtype, so D * alpha stays in D's dtype
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    return D * alpha


def gemm_add_dual_precision(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    C_scale: float | None = None,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # out in C's dtype, and again in A's as Newton-Schulz's next
    # operand; without use_quack, bf16 A B rounds before the add
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    if C_scale is not None:
        C = C_scale * C
    out = D + C
    return out, out.to(dtype=A.dtype)


def gemm_sigmoid(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    use_quack: bool = False,
) -> torch.Tensor:
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    return torch.nn.functional.sigmoid(D)


# ---- swiglu


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    layout: PairLayout,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # B (K, 2N), gate and up interleaved or split
    preact = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    postact = swiglu(
        preact=preact,
        layout=layout,
    )
    return preact, postact


def gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    layout: PairLayout,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # preact = (A @ B) * rstd, and postact = swiglu(preact)
    # rstd is fp32, so the product is cast back to the GEMM output's dtype
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    rstd = rearrange(rstd, "m -> m 1")
    preact = D * rstd
    preact = preact.to(dtype=D.dtype)
    postact = swiglu(
        preact=preact,
        layout=layout,
    )
    return preact, postact


def gemm_swiglu_bwd_zdz(
    A: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    layout: PairLayout,
    scale: float | None = None,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # the down projection's dgrad through swiglu at the stored
    # preact Z; ZdZ = scale * sum(Z dZ), and postact for the wgrad
    dpostact = gemm(
        A=A,
        B=B.mT,
        use_quack=use_quack,
    )
    dZ, postact = dswiglu(
        preact=Z,
        grad=dpostact,
        layout=layout,
    )
    Z_dZ = Z * dZ
    ZdZ = reduce(Z_dZ, "m n -> m", "sum")
    if scale is not None:
        ZdZ = ZdZ * scale
    return dZ, ZdZ, postact


def dswiglu_backward(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
    layout: PairLayout,
) -> torch.Tensor:
    # dpreact alone
    dpreact, _ = dswiglu(
        preact=preact,
        grad=grad_out,
        layout=layout,
    )
    return dpreact


# ---- rmsnorm


def gemm_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    use_quack: bool = False,
) -> torch.Tensor:
    # (A @ B) * rstd: the input norm's row scale, applied after the GEMM
    # rstd is fp32, so the product is cast back to the GEMM output's dtype
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    rstd = rearrange(rstd, "m -> m 1")
    scaled = D * rstd
    return scaled.to(dtype=D.dtype)


def gemm_residual_partial_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pre = A @ B + C, post = pre * weight, and rstd of the
    # unrounded pre, which the GEMM consuming post applies
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    pre = D + C
    post = pre * weight
    rstd = rmsnorm_rstd(
        x=pre,
        eps=eps,
    )
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
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # the next GEMM's dgrad times alpha, through drmsnorm
    # at the stored pre and rstd; dpre added to dX if given
    dnormed = gemm(
        A=A,
        B=B.mT,
        use_quack=use_quack,
    )
    if alpha is not None:
        dnormed = dnormed * alpha
    dpre, dweight, normed = drmsnorm(
        pre=pre,
        grad=dnormed,
        weight=weight,
        ZdZ=ZdZ,
        rstd=rstd,
    )
    if dX is None:
        return dpre, dweight, normed
    else:
        accumulated = dX + dpre
        return accumulated, dweight, normed


# ---- rope: q and k heads rotated, v passed through
# q | k and v return apart; the angles come as cached cos
# and sin tables, or as positions and frequencies


def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    positions: torch.Tensor | None,
    frequencies: torch.Tensor | None,
    size_q: int,
    size_k: int,
    size_v: int,
    head_dim: int,
    layout: PairLayout,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # rope(A @ B), N = [q | k | v]; the split layout relabels B's
    # q | k columns per head
    qkv = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    size_qk = size_q + size_k
    qk, v = torch.split(qkv, (size_qk, size_v), dim=1)
    if cos is None:
        cos, sin = rope_tables(
            positions=positions,
            frequencies=frequencies[:size_qk],
        )
    rotated_qk = rope(
        x=qk,
        cos=cos,
        sin=sin,
        layout=layout,
        head_dim=head_dim,
    )
    return rotated_qk, v


def gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    positions: torch.Tensor | None,
    frequencies: torch.Tensor | None,
    size_q: int,
    size_k: int,
    size_v: int,
    head_dim: int,
    layout: PairLayout,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # rope((A @ B) * rstd): the input norm's row scale, applied after the GEMM
    # rstd is fp32, so the product is cast back to the GEMM output's dtype
    # q | k and v are scaled separately, so that compiled, the scaled q | k
    # goes straight into rope without being stored
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    size_qk = size_q + size_k
    D_qk, D_v = torch.split(D, (size_qk, size_v), dim=1)
    rstd = rearrange(rstd, "m -> m 1")
    z_qk = D_qk * rstd
    z_qk = z_qk.to(dtype=D.dtype)
    z_v = D_v * rstd
    z_v = z_v.to(dtype=D.dtype)
    if cos is None:
        cos, sin = rope_tables(
            positions=positions,
            frequencies=frequencies[:size_qk],
        )
    rotated_qk = rope(
        x=z_qk,
        cos=cos,
        sin=sin,
        layout=layout,
        head_dim=head_dim,
    )
    return rotated_qk, z_v


def gemm_qknorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    positions: torch.Tensor | None,
    frequencies: torch.Tensor | None,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    layout: PairLayout,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # rope(qknorm(A @ B)) over [q | k], with preact and
    # head_mean_sq; split relabels B and gamma per head
    preact = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    head_mean_sq = head_mean_square(
        x=preact,
        head_dim=head_dim,
    )
    normed = qknorm(
        x=preact,
        head_mean_sq=head_mean_sq,
        gamma=weight,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
    )
    if cos is None:
        cos, sin = rope_tables(
            positions=positions,
            frequencies=frequencies,
        )
    rotated_normed = rope(
        x=normed,
        cos=cos,
        sin=sin,
        layout=layout,
        head_dim=head_dim,
    )
    return rotated_normed, preact, head_mean_sq


def gemm_rmsnorm_qknorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    positions: torch.Tensor | None,
    frequencies: torch.Tensor | None,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    layout: PairLayout,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # the rotated output and head_mean_sq use z = (A @ B) * rstd
    # rstd is fp32, so z is cast back to preact's dtype
    # preact is saved without rstd, and the backward rebuilds z from it
    preact = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    rstd = rearrange(rstd, "m -> m 1")
    z = preact * rstd
    z = z.to(dtype=preact.dtype)
    head_mean_sq = head_mean_square(
        x=z,
        head_dim=head_dim,
    )
    normed = qknorm(
        x=z,
        head_mean_sq=head_mean_sq,
        gamma=weight,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
    )
    if cos is None:
        cos, sin = rope_tables(
            positions=positions,
            frequencies=frequencies,
        )
    rotated_normed = rope(
        x=normed,
        cos=cos,
        sin=sin,
        layout=layout,
        head_dim=head_dim,
    )
    return rotated_normed, preact, head_mean_sq


def qknorm_rope_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    x: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    pos: torch.Tensor | None,
    freq: torch.Tensor | None,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    layout: PairLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    # gemm_qknorm_rope's backward, x its saved
    # preact; split relabels dq, dk, x and gamma
    if cos is None:
        cos, sin = rope_tables(
            positions=pos,
            frequencies=freq,
        )
    # TODO: check this
    dqk = torch.cat((dq, dk), dim=1)
    dnormed = rope(
        x=dqk,
        cos=cos,
        sin=-sin,
        layout=layout,
        head_dim=head_dim,
    )
    dx, dgamma, _ = dqknorm(
        dn=dnormed,
        z=x,
        head_mean_sq=head_mean_sq,
        gamma=gamma,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
    )
    return dx.to(dtype=x.dtype), dgamma


def rope_bwd_zdz(
    y: torch.Tensor,
    dy: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    pos: torch.Tensor | None,
    freq: torch.Tensor | None,
    scale: float,
    size_q: int,
    size_k: int,
    size_v: int,
    head_dim: int,
    layout: PairLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # dz = R^T dy for y = R z, v's dy as it came; R is
    # orthogonal, so zdz = scale * sum(z dz) = scale * sum(y dy)
    size_qk = size_q + size_k
    dy_qk, dy_v = torch.split(dy, (size_qk, size_v), dim=1)
    if cos is None:
        cos, sin = rope_tables(
            positions=pos,
            frequencies=freq[:size_qk],
        )
    dz_qk = rope(
        x=dy_qk,
        cos=cos,
        sin=-sin,
        layout=layout,
        head_dim=head_dim,
    )
    y_dy = y * dy
    zdz_unscaled = reduce(y_dy, "m n -> m", "sum")
    zdz = zdz_unscaled * scale
    return dz_qk, dy_v, zdz


def qknorm_rope_bwd_zdz(
    dy: torch.Tensor,
    preact: torch.Tensor,
    v: torch.Tensor,
    rstd: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    pos: torch.Tensor | None,
    freq: torch.Tensor | None,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    scale: float,
    layout: PairLayout,
    dv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    # backward at z = rstd preact, v's dz = dy + dv; zdz =
    # scale * sum(z dz), its q | k part from dqknorm
    size_qk = preact.shape[1]
    rstd = rearrange(rstd, "m -> m 1")
    z = rstd * preact
    if cos is None:
        cos, sin = rope_tables(
            positions=pos,
            frequencies=freq,
        )
    size_v = v.shape[1]
    dy_qk, dy_v = torch.split(dy, (size_qk, size_v), dim=1)
    dnormed = rope(
        x=dy_qk,
        cos=cos,
        sin=-sin,
        layout=layout,
        head_dim=head_dim,
    )
    dz_qk, dgamma, zdz_qk = dqknorm(
        dn=dnormed,
        z=z,
        head_mean_sq=head_mean_sq,
        gamma=gamma,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
    )
    dz_v = dy_v
    if dv is not None:
        dz_v = dz_v + dv
    v_dz_v = v * dz_v
    zdz_v = reduce(v_dz_v, "m n -> m", "sum")
    zdz_unscaled = zdz_v + zdz_qk
    zdz = zdz_unscaled * scale
    return dz_qk.to(dtype=dy.dtype), dz_v, zdz, dgamma


# ---- xsa: each head's output minus its component along v


def _dot_per_head(
    left_heads: torch.Tensor,
    right_heads: torch.Tensor,
) -> torch.Tensor:
    # per-head dot over d, kept (..., h, 1) to broadcast
    left_right = left_heads * right_heads
    return reduce(left_right, "... h d -> ... h 1", "sum")


def xsa_fwd(
    x: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    # per head z = x - (<x, v> / <v, v>) v, no epsilon
    x_heads = rearrange(x, "... (h d) -> ... h d", d=head_dim)
    v_heads = rearrange(v, "... (h d) -> ... h d", d=head_dim)
    x_dot_v = _dot_per_head(
        left_heads=x_heads,
        right_heads=v_heads,
    )
    v_dot_v = _dot_per_head(
        left_heads=v_heads,
        right_heads=v_heads,
    )
    x_projection_scale = x_dot_v / v_dot_v
    x_projection = x_projection_scale * v_heads
    z_heads = x_heads - x_projection
    z = rearrange(z_heads, "... h d -> ... (h d)")
    return z.to(dtype=x.dtype)


def dxsa(
    grad: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xsa's vjp, closed form so a test can feed dz from the GEMM;
    # per head, a = <x, v> / <v, v> and b = <dz, v> / <v, v>:
    # dx = dz - b v, dv = 2 a b v - (a dz + b x)
    dz_heads = rearrange(grad, "... (h d) -> ... h d", d=head_dim)
    x_heads = rearrange(x, "... (h d) -> ... h d", d=head_dim)
    v_heads = rearrange(v, "... (h d) -> ... h d", d=head_dim)
    v_dot_v = _dot_per_head(
        left_heads=v_heads,
        right_heads=v_heads,
    )
    x_dot_v = _dot_per_head(
        left_heads=x_heads,
        right_heads=v_heads,
    )
    x_projection_scale = x_dot_v / v_dot_v
    dz_dot_v = _dot_per_head(
        left_heads=dz_heads,
        right_heads=v_heads,
    )
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
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # the output projection's dgrad A @ B^T, through dxsa
    dz = gemm(
        A=A,
        B=B.mT,
        use_quack=use_quack,
    )
    dx, dv = dxsa(
        grad=dz,
        x=x,
        v=v,
        head_dim=head_dim,
    )
    return dx.to(dtype=x.dtype), dv.to(dtype=v.dtype)


# ---- cross entropy: the LM head's loss, its gradient formed in the forward
# the logits in logits_dtype: fp32 for torch, as CODA's epilogues read them
# off the accumulator, and fp64 for the oracle; zdz over dlogits as CODA
# stores them, in bf16


def cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    reduction: str,
    logits_dtype: torch.dtype,
) -> torch.Tensor:
    # the loss over the logits cast to logits_dtype, as in TorchTitan's
    # cross_entropy_loss (torchtitan/components/loss.py)
    return torch.nn.functional.cross_entropy(
        input=logits.to(dtype=logits_dtype),
        target=target,
        ignore_index=ignore_index,
        reduction=reduction,
    )


def gemm_lse(
    A: torch.Tensor,
    B: torch.Tensor,
    logits_dtype: torch.dtype,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # the logits A @ B and each row's lse
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    logits = D.to(dtype=logits_dtype)
    lses = torch.logsumexp(
        input=logits,
        dim=-1,
    )
    return logits, lses


def gemm_rmsnorm_lse(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    logits_dtype: torch.dtype,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # the logits (A @ B) rstd, the final norm's scale deferred, and
    # each row's lse
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    rstd = rearrange(rstd, "m -> m 1")
    scaled = D * rstd
    logits = scaled.to(dtype=logits_dtype)
    lses = torch.logsumexp(
        input=logits,
        dim=-1,
    )
    return logits, lses


def gemm_lse_select_logits(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    logits_dtype: torch.dtype,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # each row's loss lse - target logit, its lse and its target
    # logit; an ignored row's loss and logit 0
    logits, lses = gemm_lse(
        A=A,
        B=B,
        logits_dtype=logits_dtype,
        use_quack=use_quack,
    )
    non_ignored = target != ignore_index
    rows = torch.arange(
        A.shape[0],
        device=A.device,
    )
    # ignored rows have no target column, so they read column 0, and
    # multiplying by non_ignored zeroes them
    safe_target = torch.where(
        condition=non_ignored,
        input=target,
        other=0,
    )
    target_logits = logits[rows, safe_target] * non_ignored
    margins = lses - target_logits
    return margins * non_ignored, lses, target_logits


def cross_entropy_fwd_bwd(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # with return_zdz, also zdz: each row's sum of logits * dlogits, which
    # the final norm's backward needs
    losses, dlogits = cross_entropy(
        logits=logits,
        target=target,
        ignore_index=ignore_index,
    )
    if return_zdz:
        logits_dlogits = logits * dlogits
        zdz = reduce(logits_dlogits, "m v -> m", "sum")
    else:
        zdz = None
    return losses, dlogits, zdz


def cross_entropy_fwd_bwd_bench(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_zdz: bool,
    logits_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # torch's path in bf16: the loss over the logits cast to logits_dtype,
    # and autograd's gradient back in the logits' dtype (vjp, so it
    # compiles to one graph); target int64
    loss_fn = functools.partial(
        cross_entropy_loss,
        target=target,
        ignore_index=ignore_index,
        reduction="none",
        logits_dtype=logits_dtype,
    )
    losses, pullback = torch.func.vjp(loss_fn, logits)
    ones = torch.ones_like(losses)
    (dlogits,) = pullback(ones)
    if return_zdz:
        row_dot_terms = logits * dlogits
        row_dot = reduce(row_dot_terms, "m v -> m", "sum")
    else:
        row_dot = None
    return losses, dlogits, row_dot


def gemm_cross_entropy_fwd_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    logits_dtype: torch.dtype,
    return_zdz: bool,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # the loss over the logits A @ B
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    logits = D.to(dtype=logits_dtype)
    return cross_entropy_fwd_bwd(
        logits=logits,
        target=target,
        ignore_index=ignore_index,
        return_zdz=return_zdz,
    )


def gemm_cross_entropy_fwd_bwd_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    logits_dtype: torch.dtype,
    return_zdz: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # torch's LM head: the GEMM stores bf16 logits, then the loss over
    # them cast to logits_dtype and autograd's gradient
    logits = gemm(
        A=A,
        B=B,
    )
    return cross_entropy_fwd_bwd_bench(
        logits=logits,
        target=target,
        ignore_index=ignore_index,
        return_zdz=return_zdz,
        logits_dtype=logits_dtype,
    )


def gemm_rmsnorm_cross_entropy_fwd_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    logits_dtype: torch.dtype,
    return_zdz: bool,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # the loss over the logits (A @ B) rstd, the final norm's scale
    # deferred
    D = gemm(
        A=A,
        B=B,
        use_quack=use_quack,
    )
    rstd = rearrange(rstd, "m -> m 1")
    scaled = D * rstd
    logits = scaled.to(dtype=logits_dtype)
    return cross_entropy_fwd_bwd(
        logits=logits,
        target=target,
        ignore_index=ignore_index,
        return_zdz=return_zdz,
    )


def gemm_rmsnorm_cross_entropy_fwd_bwd_bench(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    logits_dtype: torch.dtype,
    return_zdz: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # the logits stored in bf16, then the loss over them cast to
    # logits_dtype and autograd's gradient
    logits = gemm_rmsnorm(
        A=A,
        B=B,
        rstd=rstd,
    )
    return cross_entropy_fwd_bwd_bench(
        logits=logits,
        target=target,
        ignore_index=ignore_index,
        return_zdz=return_zdz,
        logits_dtype=logits_dtype,
    )
