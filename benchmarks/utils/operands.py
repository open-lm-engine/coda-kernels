import math
import torch
from typing import NamedTuple
from einops import rearrange, reduce, repeat

from benchmarks.backends import torch_ops
from benchmarks.utils import data
from benchmarks.utils import models
from benchmarks.utils.data import Sampler, Shape
from coda.core.ops.rope_utils import make_rope_frequencies


class Normal(NamedTuple):
    # a Gaussian's mean and std
    mean: float
    std: float


class Block(NamedTuple):
    # a shifted block's forward outputs, each given a cotangent, and its
    # leaves
    outputs: tuple[str, ...]
    leaves: tuple[str, ...]


# rope's base, Llama-3's
ROPE_BASE = 500000.0


BLOCKS = {
    "transformer.block_pre": Block(
        outputs=("residual", "qkv", "v"),
        leaves=("x", "w", "wn", "wn2"),
    ),
    "transformer.block": Block(
        outputs=("x2", "qkv", "v"),
        leaves=("x0", "o0", "v0", "w0", "w1", "w2", "w3", "wn0", "wn1", "wn2"),
    ),
    "transformer.block_post": Block(
        outputs=("loss",),
        leaves=("x0", "o0", "v0", "w0", "w1", "w2", "w3", "wn0", "wn1"),
    ),
    "llama3.block_pre": Block(
        outputs=("residual", "qkv"),
        leaves=("x", "w", "wn"),
    ),
    "llama3.block": Block(
        outputs=("x2", "qkv"),
        leaves=("x0", "y0", "w0", "w1", "w2", "w3", "wn0", "wn1"),
    ),
    "llama3.block_post": Block(
        outputs=("loss",),
        leaves=("x0", "y0", "w0", "w1", "w2", "w3", "wn0", "wn1"),
    ),
}


def sequence_positions(shape: Shape) -> torch.Tensor:
    # rope positions for one sequence, starting at the shape's offset
    end = shape.position_offset + models.SEQUENCE_LENGTH
    return torch.arange(
        start=shape.position_offset,
        end=end,
        dtype=torch.int32,
        device="cuda",
    )


def token_positions(shape: Shape) -> torch.Tensor:
    # every token's rope position, (B * T,); each sequence starts again
    # at the shape's offset
    sequence = sequence_positions(shape=shape)
    positions = repeat(
        tensor=sequence,
        pattern="t -> (b t)",
        b=models.batch_size(shape=shape),
    )
    assert positions.is_contiguous()
    return positions


def inv_freq(shape: Shape) -> torch.Tensor:
    # rope's inverse frequency for each pair j of a head,
    # base^(-2j / head_dim), in fp32 as Llama-3 computes it
    even_indices = torch.arange(
        start=0,
        end=shape.head_dim,
        step=2,
        dtype=torch.float32,
        device="cuda",
    )
    exponents = even_indices / shape.head_dim
    powers = ROPE_BASE**exponents
    pair_frequencies = 1.0 / powers
    assert pair_frequencies.is_contiguous()
    return pair_frequencies


def batched_shape(
    shape: Shape,
    width: int,
) -> tuple[int, int, int]:
    # (B, T, width): the step's tokens as a batch of sequences, as a block
    # takes them
    return (
        models.batch_size(shape=shape),
        models.SEQUENCE_LENGTH,
        width,
    )


def split_layout_heads(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    # each head along the last axis, from the interleaved layout, pairs
    # (2j, 2j + 1), to the split layout, pairs (j, j + head_dim / 2)
    assert x.shape[-1] == num_heads * head_dim
    assert head_dim % 2 == 0
    half = head_dim // 2
    return rearrange(
        tensor=x,
        pattern="... (h p two) -> ... (h two p)",
        h=num_heads,
        two=2,
        p=half,
    ).contiguous()


def split_layout_qkv(
    x: torch.Tensor,
    size_qk: int,
    head_dim: int,
) -> torch.Tensor:
    # a [q | k | v] activation in the split layout; v has no rope pairs
    assert x.ndim == 2
    assert 0 < size_qk < x.shape[1]
    assert size_qk % head_dim == 0
    num_heads = size_qk // head_dim
    qk = x[:, :size_qk]
    v = x[:, size_qk:]
    split_qk = split_layout_heads(
        x=qk,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    split = torch.cat(
        tensors=(split_qk, v),
        dim=1,
    )
    assert split.is_contiguous()
    return split


def split_layout_rope_weight(
    B: torch.Tensor,
    size_qk: int,
    head_dim: int,
) -> torch.Tensor:
    # B is the transposed weight: its q and k rows move to the split
    # layout, any v rows stay, and it is transposed back to keep B's strides
    weight = B.mT
    assert weight.ndim == 2
    assert weight.is_contiguous()
    assert 0 < size_qk <= weight.shape[0]
    assert head_dim % 2 == 0
    weight_qk = weight[:size_qk]
    weight_v = weight[size_qk:]
    half = head_dim // 2
    weight_split_qk = rearrange(
        tensor=weight_qk,
        pattern="(h p two) k -> (h two p) k",
        two=2,
        p=half,
    )
    weight_split = torch.cat(
        tensors=(weight_split_qk, weight_v),
        dim=0,
    )
    assert weight_split.is_contiguous()
    return weight_split.mT


def split_layout_swiglu_weight(B: torch.Tensor) -> torch.Tensor:
    # B is the transposed weight: its gate and up rows are regrouped into
    # torch's split layout, all gates then all ups, and transposed back to
    # keep B's strides
    split = rearrange(
        tensor=B.mT,
        pattern="(pair two) k -> (two pair) k",
        two=2,
    ).contiguous()
    return split.mT


def cache_rope_cos_sin(
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # fp32 cos and sin tables of one head's rope angles; the angles are
    # computed in fp64, so the tables rotate by the oracle's angles
    assert frequencies.ndim == 1
    assert frequencies.shape[0] % head_dim == 0
    head_frequencies = frequencies[:head_dim]
    head_frequencies_fp64 = head_frequencies.to(dtype=torch.float64)
    positions_fp64 = positions.to(dtype=torch.float64)
    cos, sin = torch_ops.rope_tables(
        positions=positions_fp64,
        frequencies=head_frequencies_fp64,
    )
    return cos.to(dtype=torch.float32), sin.to(dtype=torch.float32)


def measure_head_norms(
    x: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    # each row's norm per head, (M, heads), in fp32
    x_fp32 = x.to(dtype=torch.float32)
    heads = rearrange(
        tensor=x_fp32,
        pattern="m (h d) -> m h d",
        d=head_dim,
    )
    return heads.norm(dim=-1)


def gemm_operands(
    shape: Shape,
    out_features: int,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # a linear layer's inputs: A (tokens, dim), and B, the transposed
    # weight, k-major as nn.Linear stores it
    weight = sampler.weight(
        shape=(out_features, shape.dim),
        dtype=dtype,
    )
    A = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    return {
        "A": A,
        "B": weight.mT,
    }


def rope_operands(shape: Shape) -> dict:
    # rope's positions, its per-column frequencies for [q | k | v] and for
    # [q | k], and the widths of q, k and v
    size_q = shape.heads * shape.head_dim
    size_k = shape.kv_heads * shape.head_dim
    size_v = shape.kv_heads * shape.head_dim
    pair_frequencies = inv_freq(shape=shape)
    qk_heads = shape.heads + shape.kv_heads
    frequencies_qk = make_rope_frequencies(
        inv_freq=pair_frequencies,
        num_heads=qk_heads,
    )

    # v's columns get frequency 0, which leaves them unrotated
    frequencies_v = torch.zeros(
        size=(size_v,),
        dtype=frequencies_qk.dtype,
        device=frequencies_qk.device,
    )
    frequencies_qkv = torch.cat(
        tensors=(frequencies_qk, frequencies_v),
        dim=0,
    )
    positions = token_positions(shape=shape)
    return {
        "positions": positions,
        "frequencies_qkv": frequencies_qkv,
        "frequencies_qk": frequencies_qk,
        "size_q": size_q,
        "size_k": size_k,
        "size_v": size_v,
    }


def muon_iterate(
    weight: torch.Tensor,
    steps: int,
    coefficients: tuple[float, float, float],
) -> torch.Tensor:
    # Muon's Newton-Schulz iterate after this many steps, computed in fp32:
    # the weight divided by its norm, then X = a X + (b G + c G G) X with
    # G = X X^T at each step
    a, b, c = coefficients
    weight_fp32 = weight.to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(weight_fp32)
    X = weight_fp32 / (norm + MUON_EPS)

    for _ in range(steps):
        gram = torch.mm(
            input=X,
            mat2=X.mT,
        )
        gram_squared = torch.mm(
            input=gram,
            mat2=gram,
        )
        polynomial = b * gram + c * gram_squared
        update = torch.mm(
            input=polynomial,
            mat2=X,
        )
        X = a * X + update

    return X


def lm_head_dlogits(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    # the cross-entropy gradient on the LM head's logits: each row sums to
    # zero, with most of its weight on the target
    assert A.ndim == 2
    M, K = A.shape
    assert B.ndim == 2
    assert B.shape[0] == K
    assert target.shape == (M,)
    assert M % DLOGITS_CHUNK_SIZE == 0
    dlogits_chunks = []

    for start in range(0, M, DLOGITS_CHUNK_SIZE):
        stop = start + DLOGITS_CHUNK_SIZE
        A_chunk = A[start:stop]
        target_chunk = target[start:stop]
        assert A_chunk.shape == (DLOGITS_CHUNK_SIZE, K)
        assert target_chunk.shape == (DLOGITS_CHUNK_SIZE,)
        logits = torch.mm(
            input=A_chunk,
            mat2=B,
            out_dtype=torch.float32,
        )
        _, dlogits = torch_ops.cross_entropy(
            logits=logits,
            target=target_chunk,
            ignore_index=data.IGNORE_INDEX,
            cross_entropy_numerics=torch_ops.CrossEntropyNumerics.precise,
        )
        dlogits_chunks.append(dlogits.to(dtype=dtype))

    return torch.cat(tensors=dlogits_chunks)


def qknorm_rope_bwd_zdz_operands(
    shape: Shape,
    preact: torch.Tensor,
    gamma: torch.Tensor,
    size_v: int,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # what the zdz backward reads besides the forward's saved tensors: dy,
    # the cotangent of [q | k | v] (from qk_cotangent on q and k, so zdz is
    # a well-conditioned sum, and random on v); v; and dv, v's gradient
    # from attention, which the backward adds to dy on v's columns
    random = sampler.randn(
        shape=preact.shape,
        dtype=dtype,
    )
    cotangent_qk = qk_cotangent(
        shape=shape,
        random=random,
        z=preact,
        gamma=gamma,
        sampler=sampler,
        dtype=dtype,
    )
    cotangent_v = sampler.randn(
        shape=(shape.tokens, size_v),
        dtype=dtype,
    )
    dy = torch.cat(
        tensors=(cotangent_qk, cotangent_v),
        dim=1,
    )
    v = sampler.randn(
        shape=(shape.tokens, size_v),
        dtype=dtype,
    )
    dv = sampler.randn(
        shape=(shape.tokens, size_v),
        dtype=dtype,
    )
    return {
        "dy": dy,
        "v": v,
        "dv": dv,
    }


def lm_head_operands(
    shape: Shape,
    rmsnorm: bool,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # the LM head's inputs: the final norm's output A, the weight
    # transposed as B, and targets with some ignored
    samples_gemm = gemm_operands(
        shape=shape,
        out_features=shape.vocab_size,
        sampler=sampler,
        dtype=dtype,
    )
    weight = samples_gemm["B"].mT
    target = sampler.targets(
        size=shape.tokens,
        vocab_size=shape.vocab_size,
    )

    A_fp32 = lm_head_input(
        shape=shape,
        random=samples_gemm["A"],
        weight=weight,
        target=target,
        sampler=sampler,
    )

    if rmsnorm:
        # the op normalizes A / gamma to unit rms per row, then multiplies
        # by B, the weight times gamma; B also carries the rows' median rms,
        # so each row's logits are the fit's scaled by 0.78 to 1.28, and the
        # loss distribution stays the plain op's
        gamma = sampler.gamma(
            size=shape.dim,
            dtype=torch.float32,
        )
        A_over_gamma = A_fp32 / gamma
        weight_times_gamma = weight * gamma
        samples_rmsnorm = rmsnorm_operands(
            shape=shape,
            x=A_over_gamma,
            rstd_nominal=shape.statistics.final_rstd,
            sampler=sampler,
            dtype=dtype,
        )
        return {
            "A": samples_rmsnorm["A"],
            "B": weight_times_gamma.to(dtype=dtype).mT,
            "rstd": samples_rmsnorm["rstd"],
            "target": target,
        }
    else:
        return {
            "A": A_fp32.to(dtype=dtype),
            "B": samples_gemm["B"],
            "target": target,
        }


def linear_qknorm_rope_operands(
    shape: Shape,
    size_q: int,
    size_k: int,
    size_v: int,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # a linear_qknorm_rope step's tensors: the [q | k | v] weight, x,
    # gamma, and the cotangents dq, dk and dv; dq and dk come from
    # qk_cotangent
    size_qk = size_q + size_k
    weight = sampler.weight(
        shape=(size_qk + size_v, shape.dim),
        dtype=dtype,
    )
    x = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    gamma = sampler.gamma(
        size=2 * shape.head_dim,
        dtype=dtype,
    )
    random_dq = sampler.randn(
        shape=(shape.tokens, size_q),
        dtype=dtype,
    )
    random_dk = sampler.randn(
        shape=(shape.tokens, size_k),
        dtype=dtype,
    )
    dv = sampler.randn(
        shape=(shape.tokens, size_v),
        dtype=dtype,
    )

    # q and k before the norm, as the GEMM's fp32 accumulator holds them
    weight_qk = weight[:size_qk]
    assert weight_qk.shape == (size_qk, shape.dim)
    accumulator = torch.mm(
        input=x,
        mat2=weight_qk.mT,
        out_dtype=torch.float32,
    )
    random_qk = torch.cat(
        tensors=(random_dq, random_dk),
        dim=1,
    )
    cotangent = qk_cotangent(
        shape=shape,
        random=random_qk,
        z=accumulator,
        gamma=gamma,
        sampler=sampler,
        dtype=dtype,
    )
    cotangent_q, cotangent_k = torch.split(
        tensor=cotangent,
        split_size_or_sections=(size_q, size_k),
        dim=1,
    )
    return {
        "weight": weight,
        "x": x,
        "gamma": gamma,
        "dq": cotangent_q.contiguous(),
        "dk": cotangent_k.contiguous(),
        "dv": dv,
    }


def gemm_residual_rmsnorm_gemm_operands(
    shape: Shape,
    in_features: int,
    out_features: int,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # gemm_residual_rmsnorm_gemm's leaves, drawn as block_input draws them:
    # the residual stream x; y and w0, the GEMM that adds onto it; w1, the
    # next GEMM's weight; and wn, the norm's weight between them
    x = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    y = sampler.randn(
        shape=(shape.tokens, in_features),
        dtype=dtype,
    )
    w0 = sampler.weight(
        shape=(shape.dim, in_features),
        dtype=dtype,
    )
    w1 = sampler.weight(
        shape=(out_features, shape.dim),
        dtype=dtype,
    )
    wn = sampler.gamma(
        size=shape.dim,
        dtype=dtype,
    )
    return {
        "x": ACTIVATION_STD * x,
        "y": ACTIVATION_STD * y,
        "w0": w0,
        "w1": w1,
        "wn": wn,
    }
