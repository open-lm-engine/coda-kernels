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
from tests.utils.forms import Case, Form


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
# ones), the only one an op that does not read the
# vocab runs at
LLAMA3_VOCAB_SIZE = 128256
# gpt-oss's and Qwen3.5's vocab sizes, from their config.json
GPT_OSS_VOCAB_SIZE = 201088
QWEN3_5_VOCAB_SIZE = 248320
# the vocab sizes an op that reads the vocab runs at:
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
        "--ops",
        nargs="+",
        default=None,
        help="registry ops to run, all benchmarked ones by default",
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
        help="file to append one JSON row per op and shape to",
    )
    options = parser.parse_args()

    if options.ops is None:
        options.ops = list(registry.BENCHED)

    if options.models is None:
        options.models = list(LLAMA3)

    return options


def configure_torch() -> None:
    # one inductor cache per campaign: the directory alone
    # moved identical code by 2.9%
    assert "TORCHINDUCTOR_CACHE_DIR" in os.environ
    # fp32 matmuls stay IEEE fp32, never TF32, through
    # the new fp32_precision API alone (mixing it with the
    # legacy allow_tf32 flags raises, and the environment's
    # override is a legacy setting)
    assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "1"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    # never run eager in place of a compile: past dynamo's
    # recompile limit a compile without fullgraph falls
    # back to eager with only a warning, and with
    # suppress_errors a failed compile does too; raise
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch._dynamo.config.suppress_errors = False


def op_vocab_sizes(op_name: str) -> tuple[int, ...]:
    # every vocab size for an op that reads the vocab,
    # and Llama-3's own for the rest
    if op_name in registry.VOCAB_READERS:
        return VOCAB_SIZES
    else:
        return (LLAMA3_VOCAB_SIZE,)


def op_kv_heads(
    op_name: str,
    model: Model,
) -> int:
    # the model's own k and v heads, or as many as q heads
    # for the ops whose kernels take no fewer (qk-norm's)
    if op_name in registry.MHA:
        return model.heads
    else:
        return model.kv_heads


def row_label(
    model_name: str,
    shape: data.Shape,
) -> str:
    # a row's label, e.g. "llama3 8b gqa8 tokens=2048 V=128256"
    if shape.kv_heads == shape.heads:
        attention = "mha"
    else:
        attention = f"gqa{shape.kv_heads}"

    return f"llama3 {model_name} {attention} tokens={shape.tokens} V={shape.vocab_size}"


def bench_shapes(
    op_name: str,
    model_names: list[str],
) -> dict[str, data.Shape]:
    # every bench shape of an op, by its row label: each
    # named model at every token count and vocab size
    shapes = {}
    for model_name in model_names:
        model = LLAMA3[model_name]
        sizes = itertools.product(
            TOKENS,
            op_vocab_sizes(op_name=op_name),
        )
        for tokens, vocab_size in sizes:
            shape = model.shape(
                tokens=tokens,
                vocab_size=vocab_size,
                kv_heads=op_kv_heads(
                    op_name=op_name,
                    model=model,
                ),
            )
            label = row_label(
                model_name=model_name,
                shape=shape,
            )
            shapes[label] = shape
    return shapes


def input_shapes(case: Case) -> tuple:
    # the shapes of an op's inputs: an op with no
    # token or vocab axis (an optimizer's weight) has the
    # same ones at every token count, and runs once
    found = []
    for value in case.inputs.values():
        found.append(str(getattr(value, "shape", value)))
    return tuple(found)
