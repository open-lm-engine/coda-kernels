import math
import torch
from typing import NamedTuple


NORM_EPS = 1e-5
IGNORE_INDEX = -100
# pretraining ignores about 1 label in 2048 and
# SFT ignores whole prompts; 5% is in between
IGNORED_FRACTION = 0.05
# trained Llama-3 and Qwen3 norm weights fall
# roughly between 0.2 and 4
NORM_WEIGHT_LOW = 0.2
NORM_WEIGHT_HIGH = 4.0


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
        # a random (out, in) weight at the usual
        # init scale, 1 / sqrt(in)
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
        # uniform in [0, 1)
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
        # log-uniform in [low, high)
        uniform = self.rand(size=size)
        exponent = math.log(low) + uniform * math.log(high / low)
        return torch.exp(exponent)

    def multipliers(
        self,
        size: int,
        ratio: float,
    ) -> torch.Tensor:
        # random factors near 1; the largest is at
        # most ratio times the smallest
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
        # a norm weight in the range trained
        # models use
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
        # random token ids in [0, vocab_size)
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
        # random labels, with IGNORED_FRACTION of
        # them set to IGNORE_INDEX
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
    # how large the inputs are; each range was
    # measured on Llama-3.2-1B and Qwen3-0.6B

    # rstd of the input norm, 0.05 to 50
    rstd: float
    # rstd of the final norm, 0.04 to 3
    final_rstd: float
    # mean square of a q or k head before
    # qk-norm, 0.02 to 350
    head_mean_sq: float
    # the scale of the residual C in A @ B + C,
    # relative to A @ B: 0.15 to 10 in real
    # models
    residual_scale: float


# a typical middle layer
TYPICAL = Statistics(
    rstd=2.6,
    final_rstd=1.7,
    head_mean_sq=3.6,
    residual_scale=1.0,
)


class Shape(NamedTuple):
    # tokens through a model of these sizes
    tokens: int
    dim: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    # rope loses precision at large positions,
    # where fp32 angles lose digits; this checks
    # that without a long sequence: the tokens
    # take the positions from position_offset up
    position_offset: int
    # the kind of inputs: "random" (plain),
    # "diffuse" (an LM head at initialization),
    # or a hard case: "outlier" (a few huge
    # features, gradients aligned with outputs),
    # "floor" (values near eps, confident
    # predictions) or "stress" (extreme values,
    # blocks only)
    distribution: str
    # how large the inputs are
    statistics: Statistics


# small shapes; tokens odd, so the last row tile
# is partial, and together they cover each
# statistic's real range
SHAPES = {
    # GQA with head_dim 64, a vocab that is not
    # a multiple of 128, and a middle layer
    "gqa64": Shape(
        tokens=1025,
        dim=512,
        ffn=1152,
        heads=8,
        kv_heads=2,
        head_dim=64,
        vocab_size=1000,
        position_offset=0,
        distribution="random",
        statistics=Statistics(
            rstd=5.0,
            final_rstd=3.0,
            head_mean_sq=3.6,
            residual_scale=1.0,
        ),
    ),
    # ffn not a multiple of 256, rope from
    # position 2^20 (where fp32 angles lose
    # precision), and a last layer
    "mha128-far": Shape(
        tokens=777,
        dim=1024,
        ffn=1408,
        heads=8,
        kv_heads=8,
        head_dim=128,
        vocab_size=4104,
        position_offset=2**20,
        distribution="diffuse",
        statistics=Statistics(
            rstd=0.05,
            final_rstd=0.04,
            head_mean_sq=0.03,
            residual_scale=1.0,
        ),
    ),
    "mha128-outlier": Shape(
        tokens=1031,
        dim=1024,
        ffn=1408,
        heads=8,
        kv_heads=8,
        head_dim=128,
        vocab_size=4104,
        position_offset=0,
        distribution="outlier",
        statistics=Statistics(
            rstd=0.5,
            final_rstd=0.3,
            head_mean_sq=100.0,
            residual_scale=1.0,
        ),
    ),
    # a first layer: activations so small that
    # eps is 2.4% of the mean square, and the
    # residual still small, 0.15 (Qwen3's)
    "mha128-floor": Shape(
        tokens=1027,
        dim=1024,
        ffn=1408,
        heads=8,
        kv_heads=8,
        head_dim=128,
        vocab_size=4104,
        position_offset=0,
        distribution="floor",
        statistics=Statistics(
            rstd=35.0,
            final_rstd=0.12,
            head_mean_sq=1.0,
            residual_scale=0.15,
        ),
    ),
    "mha128-stress": Shape(
        tokens=1029,
        dim=1024,
        ffn=1408,
        heads=8,
        kv_heads=8,
        head_dim=128,
        vocab_size=4104,
        position_offset=0,
        distribution="stress",
        statistics=TYPICAL,
    ),
}
