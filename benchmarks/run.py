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
VOCAB_SIZES = (65536, 128256, 262144)
