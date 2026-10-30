import math
import torch
from typing import NamedTuple
from einops import rearrange, reduce

from coda.core.ops.rope_utils import make_rope_frequencies
from tests import refs_ops

NORM_EPS = 1e-5
# fixed seeds, so every run sees the same inputs, one
# per builder; builders whose draws meet in one case
# take different seeds, so they never repeat values
SEED_PROJECTION = 0
SEED_FP8 = 1
SEED_ROW_MULTIPLIERS = 2
IGNORE_INDEX = -100
# pretraining ignores about 1 label in 2048 and
# SFT ignores whole prompts; 5% is in between
IGNORED_FRACTION = 0.05
ROPE_BASE = 500000.0
# trained Llama-3 and Qwen3 norm weights fall
# roughly between 0.2 and 4
NORM_WEIGHT_LOW = 0.2
NORM_WEIGHT_HIGH = 4.0
# a plain rstd draw, uniform over this range
RSTD_LOW = 0.5
RSTD_HIGH = 1.5
# fp8 operands are N(0, FP8_STD) rounded to e4m3, so
# about 5% land in its subnormals, where a kernel that
# flushes them to zero shows; FP8_SCALE is not a power
# of two, so a dropped or rounded scale shows too
FP8_STD = 0.2
FP8_SCALE = 0.91


class Pairs(NamedTuple):
    # where a tensor's rope or swiglu pairs sit, to
    # convert between CODA's and a model's layouts;
    # dim is the axis holding them
    dim: int
    # the unit whose two halves form the pairs:
    # one head, or all of [gate | up]
    group: int


class QKVPairs(NamedTuple):
    # the pairs of a [q | k | v] tensor along dim:
    # q's and k's heads hold pairs, v is left as it is
    dim: int
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

    def per_head(
        self,
        dim: int,
    ) -> Pairs:
        # pairs inside each head, along dim
        return Pairs(
            dim=dim,
            group=self.head_dim,
        )

    def qk_heads(
        self,
        dim: int,
    ) -> QKVPairs:
        # pairs inside q's and k's heads along dim;
        # v has none
        return QKVPairs(
            dim=dim,
            head_dim=self.head_dim,
            size_q=self.heads * self.head_dim,
            size_k=self.kv_heads * self.head_dim,
            size_v=self.kv_heads * self.head_dim,
        )

    def gate_up(
        self,
        dim: int,
    ) -> Pairs:
        # gate and up halves, along dim
        return Pairs(
            dim=dim,
            group=2 * self.ffn,
        )

    def positions(self) -> torch.Tensor:
        # rope positions of the tokens
        return torch.arange(
            self.position_offset,
            self.position_offset + self.tokens,
            dtype=torch.int32,
            device="cuda",
        )

    def inv_freq(self) -> torch.Tensor:
        # rope's inverse frequency for each pair of
        # a head, base^(-2j / head_dim) for pair j,
        # in fp32 as Llama-3 computes it
        exponents = torch.arange(
            start=0,
            end=self.head_dim,
            step=2,
            dtype=torch.float32,
            device="cuda",
        )
        exponents = exponents / self.head_dim
        return 1.0 / (ROPE_BASE**exponents)

    def rope(self) -> dict:
        # what the rope ops take, under both of CODA's
        # names (positions and frequencies in the
        # forward ops, pos and freq in the backward
        # ones), plus the widths and qk-norm's settings
        size_q = self.heads * self.head_dim
        size_k = self.kv_heads * self.head_dim
        size_v = self.kv_heads * self.head_dim
        # CODA's table has one entry per [q | k | v]
        # column: q's and k's heads get each pair's
        # frequency, and v's columns get 0, which
        # leaves them unrotated
        frequencies_qk = make_rope_frequencies(
            inv_freq=self.inv_freq(),
            num_heads=self.heads + self.kv_heads,
        )
        frequencies_v = torch.zeros(
            size_v,
            dtype=frequencies_qk.dtype,
            device=frequencies_qk.device,
        )
        frequencies = torch.cat(
            tensors=(frequencies_qk, frequencies_v),
            dim=0,
        )
        positions = self.positions()
        return {
            "positions": positions,
            "frequencies": frequencies,
            "pos": positions,
            "freq": frequencies,
            "head_dim": self.head_dim,
            "size_q": size_q,
            "size_k": size_k,
            "size_v": size_v,
            "num_heads_q": self.heads,
            "num_heads_k": self.kv_heads,
            "eps": NORM_EPS,
        }

    def rmsnorm_linear(
        self,
        out_features: int,
    ) -> dict:
        # a linear layer on the activations: A is
        # (tokens, dim), B the weight, (dim,
        # out_features) in nn.Linear's k-major
        # layout, and rstd an input norm's row scale
        sampler = Sampler(seed=SEED_PROJECTION)
        # nn.Linear's (out_features, dim) weight;
        # B is its k-major transpose
        weight = sampler.weight(
            shape=(out_features, self.dim),
            dtype=torch.bfloat16,
        )
        B = weight.mT
        A = sampler.randn(
            shape=(self.tokens, self.dim),
            dtype=torch.bfloat16,
        )
        uniform = sampler.rand(size=self.tokens)
        rstd = RSTD_LOW + (RSTD_HIGH - RSTD_LOW) * uniform
        return {
            "A": A,
            "B": B,
            "rstd": rstd,
        }

    def fp8(
        self,
        width: int,
    ) -> dict:
        # fp8 inputs for the GEMM, with CODA's single
        # scale and torch's per-input scales
        sampler = Sampler(seed=SEED_FP8)
        A = FP8_STD * sampler.randn(
            shape=(self.tokens, self.dim),
            dtype=torch.float32,
        )
        weight = FP8_STD * sampler.randn(
            shape=(width, self.dim),
            dtype=torch.float32,
        )
        scale = torch.full(
            (),
            FP8_SCALE,
            dtype=torch.float32,
            device="cuda",
        )
        fp8_weight = weight.to(dtype=torch.float8_e4m3fn)
        return {
            "A": A.to(dtype=torch.float8_e4m3fn),
            "B": fp8_weight.mT,
            "scale": scale,
            "scale_a": scale,
            "scale_b": torch.ones_like(scale),
        }


class Model(NamedTuple):
    # one Llama-3 model size
    dim: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int

    def shape(
        self,
        tokens: int,
        vocab_size: int,
        kv_heads: int,
    ) -> Shape:
        # this model at a token count, with plain
        # inputs at a typical layer's statistics
        return Shape(
            tokens=tokens,
            dim=self.dim,
            ffn=self.ffn,
            heads=self.heads,
            kv_heads=kv_heads,
            head_dim=self.head_dim,
            vocab_size=vocab_size,
            position_offset=0,
            distribution="random",
            statistics=TYPICAL,
        )


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
