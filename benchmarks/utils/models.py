import itertools
from typing import NamedTuple

from benchmarks.utils import data


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
# tokens per step
TOKENS = (8192, 16384, 32768)
# tokens per sequence: a step's tokens are split into sequences this long
SEQUENCE_LENGTH = 2048
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
# ops run at every vocab size
OPS_WITH_VOCAB = (
    "gemm_lse",
    "gemm_rmsnorm_lse",
    "gemm_lse_select_logits",
    "cross_entropy_fwd_bwd",
    "gemm_cross_entropy_fwd_bwd",
    "gemm_rmsnorm_cross_entropy_fwd_bwd",
    "linear_cross_entropy",
    "linear_cross_entropy_forward",
    "gemm_scalar_scale_lm_head",
    "transformer.block_post",
    "llama3.block_post",
    "chain.gemm_residual_rmsnorm_gemm_cross_entropy_fwd_bwd",
)
# ops run with as many k and v heads as q heads
OPS_WITH_MHA = (
    "gemm_qknorm_rope",
    "gemm_rmsnorm_qknorm_rope",
    "qknorm_rope_bwd",
    "qknorm_rope_bwd_zdz",
    "linear_qknorm_rope",
    "gemm_xsa_bwd",
    "transformer.block_pre",
    "transformer.block",
    "transformer.block_post",
    "llama3.block_pre",
    "llama3.block",
    "llama3.block_post",
    "chain.gemm_residual_rmsnorm_gemm_qknorm_rope",
)


def get_vocab_sizes(op_name: str) -> tuple[int, ...]:
    # all vocab sizes for an op with a vocab dimension, else Llama-3's
    if op_name in OPS_WITH_VOCAB:
        return VOCAB_SIZES
    else:
        return (LLAMA3_VOCAB_SIZE,)


def get_kv_heads(
    op_name: str,
    model: Model,
) -> int:
    # the q head count for an op that runs MHA, else the model's k and v heads
    if op_name in OPS_WITH_MHA:
        return model.heads
    else:
        return model.kv_heads


def batch_size(shape: data.Shape) -> int:
    # the sequences a shape's tokens make up, SEQUENCE_LENGTH each
    assert shape.tokens % SEQUENCE_LENGTH == 0
    return shape.tokens // SEQUENCE_LENGTH


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
            get_vocab_sizes(op_name=op_name),
        )
        for tokens, vocab_size in sizes:
            shape = model.shape(
                tokens=tokens,
                vocab_size=vocab_size,
                kv_heads=get_kv_heads(
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
