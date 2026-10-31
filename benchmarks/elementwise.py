import torch
from einops import rearrange
from coda.core.elementwise import functional as elementwise_functional

from benchmarks.backends import torch_ops
from benchmarks.backends.torch_ops import PairLayout
from benchmarks.utils.cases import Case, Implementation, make_oracle_fn
from benchmarks.utils.data import Sampler, Shape


def dswiglu_backward(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    preact = sampler.randn(
        shape=(shape.tokens, 2 * shape.ffn),
        dtype=dtype,
    )
    grad_out = sampler.randn(
        shape=(shape.tokens, shape.ffn),
        dtype=dtype,
    )
    # preact's columns reordered to all gates, then all ups
    split_preact = rearrange(
        tensor=preact,
        pattern="m (pair two) -> m (two pair)",
        two=2,
    ).contiguous()
    interleaved = {
        "preact": preact,
        "grad_out": grad_out,
        "layout": PairLayout.interleaved,
    }
    split = {
        "preact": split_preact,
        "grad_out": grad_out,
        "layout": PairLayout.split,
    }
    coda_inputs = {
        "preact": preact,
        "grad_out": grad_out,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.dswiglu_backward,
            in_dims={
                "preact": 0,
                "grad_out": 0,
            },
            out_dims=(0,),
            reduce_dims=(None,),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.dswiglu_backward,
            in_dims={
                "preact": 0,
                "grad_out": 0,
            },
            out_dims=(0,),
            reduce_dims=(None,),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=elementwise_functional.dswiglu_backward,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.dswiglu_backward,
                inputs=interleaved,
            ),
            "torch/split": Implementation(
                fn=torch_ops.dswiglu_backward,
                inputs=split,
            ),
        },
        oracles={
            "coda": interleaved_oracle_impl,
            "torch/interleaved": interleaved_oracle_impl,
            "torch/split": split_oracle_impl,
        },
        outputs=("grad_pre",),
    )


OPS = {
    "dswiglu_backward": dswiglu_backward,
}
