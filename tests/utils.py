import math
import torch
from typing import NamedTuple
from einops import rearrange, reduce

from coda.core.ops.rope_utils import expand_frequencies
from tests import refs_ops

NORM_EPS = 1e-5
IGNORE_INDEX = -100
# pretraining ignores ~1 in 2048,
# SFT whole prompts
IGNORED_FRACTION = 0.05
ROPE_BASE = 500000.0


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
        return torch.randn(
            shape,
            generator=self.generator,
            dtype=dtype,
            device="cuda",
        )

    def weight(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # an (out, in) weight at a model's init
        # scale, 1 / sqrt(in)
        values = self.randn(
            shape=shape,
            dtype=torch.float32,
        )
        scaled = values * shape[1] ** -0.5
        return scaled.to(dtype=dtype)

    def rand(
        self,
        size: int,
    ) -> torch.Tensor:
        # fp32 in [0, 1)
        return torch.rand(
            size,
            generator=self.generator,
            dtype=torch.float32,
            device="cuda",
        )

    def log_uniform(
        self,
        size: int,
        low: float,
        high: float,
    ) -> torch.Tensor:
        # fp32 log-uniform over [low, high)
        uniform = self.rand(size=size)
        exponent = math.log(low) + uniform * math.log(high / low)
        return torch.exp(exponent)

    def spread(
        self,
        size: int,
        ratio: float,
    ) -> torch.Tensor:
        # fp32 factors log-uniform over
        # [1 / sqrt(ratio), sqrt(ratio)): a
        # statistic's spread within one tensor
        return self.log_uniform(
            size=size,
            low=ratio**-0.5,
            high=ratio**0.5,
        )

    def norm_weight(
        self,
        size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # a norm's weight, log-uniform over
        # [0.2, 4), as trained Llama-3 and
        # Qwen3 norm weights span
        values = self.log_uniform(
            size=size,
            low=0.2,
            high=4.0,
        )
        return values.to(dtype=dtype)

    def tokens(
        self,
        shape: tuple[int, ...],
        vocab_size: int,
    ) -> torch.Tensor:
        # int64 token ids, uniform
        # over [0, vocab_size)
        return torch.randint(
            vocab_size,
            shape,
            generator=self.generator,
            dtype=torch.int64,
            device="cuda",
        )

    def targets(
        self,
        size: int,
        vocab_size: int,
    ) -> torch.Tensor:
        # int64 labels in [0, vocab_size), IGNORED_FRACTION
        # of them ignored, so that a mean divides by the rest
        values = self.tokens(
            shape=(size,),
            vocab_size=vocab_size,
        )
        ignored = self.rand(size=size) < IGNORED_FRACTION
        return values.masked_fill(
            mask=ignored,
            value=IGNORE_INDEX,
        )


class Statistics(NamedTuple):
    # a case's input statistics, within the
    # ranges measured over Llama-3.2-1B's and
    # Qwen3-0.6B's layers on wikitext
    # the input norm's rstd: 0.05 to 50
    rstd: float
    # the final norm's rstd: 0.04 to 3
    final_rstd: float
    # a qk-norm head's mean square: 0.02 to 350
    head_mean_sq: float


class Shape(NamedTuple):
    dim: int
    head_dim: int
    vocab: int
    # the inputs' distribution: "random";
    # "outlier" (one feature many times the
    # rest, gradients along the outputs);
    # "floor" (norm inputs near eps,
    # confident CE rows); "stress" (blocks:
    # 50x every token)
    distribution: str = "random"
    # where its statistics sit, typical
    # ones by default
    statistics: Statistics
