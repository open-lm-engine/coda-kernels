import os
import json
import math
import torch
import argparse
import itertools
import operator
import statistics
from typing import NamedTuple
from unittest import mock

from benchmarks import arms as competitors
from benchmarks import bench_utils
from tests import registry
from tests.utils import compare
from tests.utils import data
from tests.utils import truth
from tests.utils.forms import Entry, Form


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
    ) -> data.Shape:
        # this model at a token count, with plain
        # inputs at a typical layer's statistics
        return data.Shape(
            tokens=tokens,
            dim=self.dim,
            ffn=self.ffn,
            heads=self.heads,
            kv_heads=kv_heads,
            head_dim=self.head_dim,
            vocab_size=vocab_size,
            position_offset=0,
            distribution="random",
            statistics=data.TYPICAL,
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
# Llama-3's own vocab size (128000 tokens and 256 special
# ones), the only one an entry that does not read the
# vocab runs at
LLAMA3_VOCAB_SIZE = 128256
# gpt-oss's and Qwen3.5's vocab sizes, from their config.json
GPT_OSS_VOCAB_SIZE = 201088
QWEN3_5_VOCAB_SIZE = 248320
# the vocab sizes an entry that reads the vocab runs at:
# powers of two, then Llama-3's, gpt-oss's and Qwen3.5's
VOCAB_SIZES = (
    2**16,
    2**17,
    2**18,
    LLAMA3_VOCAB_SIZE,
    GPT_OSS_VOCAB_SIZE,
    QWEN3_5_VOCAB_SIZE,
)


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CODA against torch: speed and accuracy")
    parser.add_argument(
        "--entries",
        nargs="+",
        default=None,
        help="registry entries to run, all benchmarked ones by default",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Llama-3 sizes to run, all of them by default",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=21,
        help="paired timing rounds",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="untimed launches per version and round",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="timed launches per version and round",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="file to append one JSON row per entry and shape to",
    )
    options = parser.parse_args()

    if options.entries is None:
        options.entries = list(registry.BENCHED)

    if options.models is None:
        options.models = list(LLAMA3)

    return options


def configure_torch() -> None:
    # one inductor cache per campaign: the directory alone
    # moved identical code by 2.9%
    assert "TORCHINDUCTOR_CACHE_DIR" in os.environ, "set TORCHINDUCTOR_CACHE_DIR"
    # fp32 matmuls stay fp32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    # never run eager in place of a compile: past dynamo's
    # recompile limit a compile without fullgraph falls
    # back to eager with only a warning, and with
    # suppress_errors a failed compile does too; raise
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch._dynamo.config.suppress_errors = False


def entry_vocab_sizes(entry_name: str) -> tuple[int, ...]:
    # every vocab size for an entry that reads the vocab,
    # and Llama-3's own for the rest
    if entry_name in registry.VOCAB_READERS:
        return VOCAB_SIZES
    else:
        return (LLAMA3_VOCAB_SIZE,)


def entry_kv_heads(
    entry_name: str,
    model: Model,
) -> int:
    # the model's own k and v heads, or as many as q heads
    # for the entries whose kernels take no fewer (qk-norm's)
    if entry_name in registry.MHA:
        return model.heads
    else:
        return model.kv_heads
