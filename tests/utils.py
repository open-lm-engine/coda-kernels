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
# trained Llama-3 and Qwen3 norm
# weights span about 0.2 to 4
NORM_WEIGHT_LOW = 0.2
NORM_WEIGHT_HIGH = 4.0


class Pairs(NamedTuple):
    # where a tensor's rope or swiglu pairs sit
    # along dim
    dim: int
    # the unit whose halves pair up: a head,
    # or all of [gate | up]
    group: int


class QKVPairs(NamedTuple):
    # a [q | k | v]'s columns: q's and k's
    # heads paired, v's columns as they are
    head_dim: int
    size_q: int
    size_k: int
    size_v: int


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

    def multipliers(
        self,
        size: int,
        ratio: float,
    ) -> torch.Tensor:
        # fp32 random factors around 1, log-uniform
        # over [1 / sqrt(ratio), sqrt(ratio))
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
        # [NORM_WEIGHT_LOW, NORM_WEIGHT_HIGH)
        values = self.log_uniform(
            size=size,
            low=NORM_WEIGHT_LOW,
            high=NORM_WEIGHT_HIGH,
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
    M: int
    dim: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int

    # the inputs' distribution: "random";
    # "outlier" (one feature many times the
    # rest, gradients along the outputs);
    # "floor" (norm inputs near eps,
    # confident CE rows); "stress" (blocks:
    # 50x every token)
    distribution: str
    # where its statistics sit, typical
    # ones by default
    statistics: Statistics

    def per_head(
        self,
        dim: int,
    ) -> Pairs:
        # each head's pairs along dim: q | k
        # columns, or weight or gamma rows
        return Pairs(
            dim=dim,
            group=self.head_dim,
        )

    def qk_heads(self) -> QKVPairs:
        # q's and k's heads paired in [q | k | v]
        return QKVPairs(
            head_dim=self.head_dim,
            size_q=self.heads * self.head_dim,
            size_k=self.kv_heads * self.head_dim,
            size_v=self.kv_heads * self.head_dim,
        )

    def gate_up(
        self,
        dim: int,
    ) -> Pairs:
        # [gate | up] halves along dim, as
        # a model packing w1 | w3 holds them
        return Pairs(
            dim=dim,
            group=2 * self.ffn,
        )



class Model(NamedTuple):
    # a Llama-3 size
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
# small test shapes, every M odd so it
# overhangs every row tile; together
# their statistics span the real ranges,
# no norm's rstd near 1
TEST_SHAPES = {
    # GQA at head_dim 64; M one row past a
    # 1024-row tile; vocab a multiple of 8,
    # not of 128; a middle layer
    "gqa64": Shape(
        M=1025,
        dim=512,
        ffn=1152,
        heads=8,
        kv_heads=2,
        head_dim=64,
        vocab_size=1000,
        offset=0,
        distribution="random",
        statistics=Statistics(
            rstd=5.0,
            final_rstd=3.0,
            head_mean_sq=3.6,
            branch_to_residual=1.0,
        ),
    ),
    # MHA at head_dim 128; ffn 1408, half a
    # 256-wide tile over; rope from 2^20,
    # where a model's fp32 angle moves bf16
    # outputs across ties; a last layer
    "mha128-far": Shape(
        M=777,
        dim=1024,
        ffn=1408,
        heads=8,
        kv_heads=8,
        head_dim=128,
        vocab_size=4104,
        offset=2**20,
        distribution="random",
        statistics=Statistics(
            rstd=0.05,
            final_rstd=0.04,
            head_mean_sq=0.03,
            branch_to_residual=1.0,
        ),
    ),
    "mha128-outlier": Shape(
        M=1031,
        dim=1024,
        ffn=1408,
        heads=8,
        kv_heads=8,
        head_dim=128,
        vocab_size=4104,
        offset=0,
        distribution="outlier",
        statistics=Statistics(
            rstd=0.5,
            final_rstd=0.3,
            head_mean_sq=100.0,
            branch_to_residual=1.0,
        ),
    ),
    "mha128-floor": Shape(
        M=1027,
        dim=1024,
        ffn=1408,
        heads=8,
        kv_heads=8,
        head_dim=128,
        vocab_size=4104,
        offset=0,
        distribution="floor",
        statistics=Statistics(
            rstd=35.0,
            final_rstd=0.12,
            head_mean_sq=1.0,
            branch_to_residual=7.0,
        ),
    ),
    "mha128-stress": Shape(
        M=1029,
        dim=1024,
        ffn=1408,
        heads=8,
        kv_heads=8,
        head_dim=128,
        vocab_size=4104,
        offset=0,
        distribution="stress",
    ),
}

