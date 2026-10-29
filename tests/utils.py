import math
import torch
from typing import NamedTuple
from einops import rearrange, reduce

from coda.core.ops.rope_utils import expand_frequencies
from tests import refs_ops


class Sampler(object):

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.generator = torch.Generator(device="cuda")
        self.generator.manual_seed(seed)

    def randn(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.randn(
            shape,
            generator=self.generator,
            device="cuda",
        )
        return values.to(dtype=dtype)

    def weight(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # an (out, in) weight at a model's init scale, 1 / sqrt(in)
        values = self.randn(
            shape=shape,
            dtype=torch.float32,
        )
        scaled = values * shape[1] ** -0.5
        return scaled.to(dtype=dtype)

    def tokens(
        self,
        shape: tuple[int, ...],
        vocab_size: int,
    ) -> torch.Tensor:
        # int64 token ids uniform over [0, vocab_size)
        return torch.randint(
            vocab_size,
            shape,
            generator=self.generator,
            device="cuda",
        )

    def targets(
        self,
        size: int,
        vocab_size: int,
    ) -> torch.Tensor:
        # int64 labels in [0, vocab_size), IGNORED_FRACTION of them
        # ignored, so that a mean divides by the rest
        values = self.tokens(
            shape=(size,),
            vocab_size=vocab_size,
        )
        ignored = self.rand(size=size) < IGNORED_FRACTION
        return values.masked_fill(
            mask=ignored,
            value=IGNORE_INDEX,
        )


class Levels(NamedTuple):
    # where a case's statistics sit, each in its real range across the layers of Llama-3.2-1B and Qwen3-0.6B on
    # wikitext: the input norm's rstd (0.05 to 50), the final norm's (0.04 to 3), a qk-norm head's mean square
    # (0.02 to 350), a block branch's rms over the residual stream's it joins (0.1 to 7)
    rstd: float
    final_rstd: float
    head_mean_sq: float
    branch: float


class Model(NamedTuple):
    # a Llama-3 size, 8 k and v heads at every one
    dim: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int


LLAMA3 = {
    "1b": Model(
        dim=2048,
        ffn=8192,
        heads=32,
        kv_heads=8,
        head_dim=64,
    ),
    "8b": Model(
        dim=4096,
        ffn=14336,
        heads=32,
        kv_heads=8,
        head_dim=128,
    ),
    "70b": Model(
        dim=8192,
        ffn=28672,
        heads=64,
        kv_heads=8,
        head_dim=128,
    ),
}
TOKENS = (2048, 8192, 32768)
VOCABS = (65536, 128256, 262144)
