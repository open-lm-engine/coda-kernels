import math
import torch
from typing import NamedTuple
from einops import rearrange, reduce, repeat
from coda.core.ops.rope_utils import make_rope_frequencies

from benchmarks.backends import torch_ops
from benchmarks.utils import data
from benchmarks.utils import models
from benchmarks.utils.data import Sampler, Shape


class Normal(NamedTuple):
    # a Gaussian's mean and std
    mean: float
    std: float


class Block(NamedTuple):
    # a shifted block's forward outputs, each of which gets a cotangent,
    # and its leaves
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
    return repeat(
        tensor=sequence,
        pattern="t -> (b t)",
        b=models.batch_size(shape=shape),
    )


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
    return 1.0 / powers


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
