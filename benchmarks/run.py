import os
import json
import math
import torch
import argparse
import operator
import statistics
from typing import NamedTuple

from benchmarks import arms as competitors
from benchmarks import bench_utils
from tests import registry
from tests.core import compare
from tests.core import data
from tests.core import truth
from tests.core.forms import Entry, Form


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
