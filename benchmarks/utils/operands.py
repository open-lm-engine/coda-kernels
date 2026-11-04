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


def positions(shape: Shape) -> torch.Tensor:
    # every token's rope position, (B * T,); each sequence starts again
    # at the shape's offset
    sequence = sequence_positions(shape=shape)
    token_positions = repeat(
        tensor=sequence,
        pattern="t -> (b t)",
        b=models.batch_size(shape=shape),
    )
    assert token_positions.is_contiguous()
    return token_positions


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
    rope_positions = positions(shape=shape)
    return {
        "positions": rope_positions,
        "frequencies_qkv": frequencies_qkv,
        "frequencies_qk": frequencies_qk,
        "size_q": size_q,
        "size_k": size_k,
        "size_v": size_v,
    }


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


def split_layout(
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


def split_swiglu_weight(B: torch.Tensor) -> torch.Tensor:
    # B is the transposed weight: its gate and up rows are regrouped into
    # torch's split layout, all gates then all ups, and transposed back to
    # keep B's strides
    split = rearrange(
        tensor=B.mT,
        pattern="(pair two) k -> (two pair) k",
        two=2,
    ).contiguous()
    return split.mT


def linear_operands(
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


def lm_head(
    shape: Shape,
    rmsnorm: bool,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # the LM head's inputs: the final norm's output A, the weight
    # transposed as B, and targets with some ignored
    samples_linear = linear_operands(
        shape=shape,
        out_features=shape.vocab_size,
        sampler=sampler,
        dtype=dtype,
    )
    weight = samples_linear["B"].mT
    target = sampler.targets(
        size=shape.tokens,
        vocab_size=shape.vocab_size,
    )

    A = logit_rows(
        shape=shape,
        weight=weight,
        target=target,
        bulk=samples_linear["A"],
        sampler=sampler,
    )

    if rmsnorm:
        # the norm's weight folds into B and divides out of A, so the
        # logits stay near the plain op's
        gamma = sampler.gamma(
            size=shape.dim,
            dtype=torch.float32,
        )
    else:
        A = A.to(dtype=dtype)
        return {
            "A": A,
            "B": samples_linear["B"],
            "target": target,
        }


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


def gemm_residual_rmsnorm_gemm_operands(
    shape: Shape,
    in_features: int,
    out_features: int,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # gemm_residual_rmsnorm_gemm's leaves, drawn as block_leaf draws them:
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
