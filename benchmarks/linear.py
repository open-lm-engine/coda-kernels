import torch
import functools
from einops import rearrange
from coda.kernels.functional import cross_entropy as cross_entropy_functional
from coda.kernels.functional import qknorm_rope as qknorm_rope_functional
from coda.kernels.functional import sigmoid as sigmoid_functional
from coda.kernels.functional import swiglu as swiglu_functional

from benchmarks.backends import torch_linear
from benchmarks.backends.torch_ops import PairLayout
from benchmarks.utils import data
from benchmarks.utils import operands
from benchmarks.utils.cases import Case, Implementation, TrainingImplementation, make_oracle_fn, make_step_fn, mean_loss
from benchmarks.utils.data import Sampler, Shape


def linear_sigmoid(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    x = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    weight = sampler.weight(
        shape=(shape.heads, shape.dim),
        dtype=dtype,
    )
    grad_out = sampler.randn(
        shape=(shape.tokens, shape.heads),
        dtype=dtype,
    )
    leaves = ("x", "weight")
    cotangents = ("grad_out",)
    inputs = {
        "x": x,
        "weight": weight,
        "grad_out": grad_out,
    }
    # the oracle runs torch's step too, widened to fp64
    torch_step = make_step_fn(
        forward=torch_linear.linear_sigmoid,
        leaves=leaves,
        cotangents=cotangents,
        copies_cotangents=False,
    )
    # what every step returns, in order
    outputs = ("out", "dx", "dweight")
    # a weight's grad sums over tokens, so the oracle adds up its chunks
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_step,
            in_dims={
                "x": 0,
                "weight": None,
                "grad_out": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, 0),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": TrainingImplementation(
                forward=sigmoid_functional.linear_sigmoid,
                inputs=inputs,
                leaves=leaves,
                cotangents=cotangents,
                copies_cotangents=False,
            ),
            "torch": TrainingImplementation(
                forward=torch_linear.linear_sigmoid,
                inputs=inputs,
                leaves=leaves,
                cotangents=cotangents,
                copies_cotangents=False,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=outputs,
    )


def linear_swiglu(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    x = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    weight = sampler.weight(
        shape=(2 * shape.ffn, shape.dim),
        dtype=dtype,
    )
    grad_out = sampler.randn(
        shape=(shape.tokens, shape.ffn),
        dtype=dtype,
    )
    # the split layout: all gate rows, then all up rows
    split_weight = rearrange(
        tensor=weight,
        pattern="(pair two) k -> (two pair) k",
        two=2,
    ).contiguous()
    leaves = ("x", "weight")
    cotangents = ("grad_out",)
    coda_inputs = {
        "x": x,
        "weight": weight,
        "grad_out": grad_out,
    }
    interleaved = {
        "x": x,
        "weight": weight,
        "grad_out": grad_out,
        "layout": PairLayout.interleaved,
    }
    split = {
        "x": x,
        "weight": split_weight,
        "grad_out": grad_out,
        "layout": PairLayout.split,
    }
    # the oracles run torch's step too, widened to fp64
    torch_step = make_step_fn(
        forward=torch_linear.linear_swiglu,
        leaves=leaves,
        cotangents=cotangents,
        copies_cotangents=False,
    )
    # what every step returns, in order
    outputs = ("out", "dx", "dweight")
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_step,
            in_dims={
                "x": 0,
                "weight": None,
                "grad_out": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, 0),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_step,
            in_dims={
                "x": 0,
                "weight": None,
                "grad_out": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, 0),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": TrainingImplementation(
                forward=swiglu_functional.linear_swiglu,
                inputs=coda_inputs,
                leaves=leaves,
                cotangents=cotangents,
                copies_cotangents=False,
            ),
            "torch/interleaved": TrainingImplementation(
                forward=torch_linear.linear_swiglu,
                inputs=interleaved,
                leaves=leaves,
                cotangents=cotangents,
                copies_cotangents=False,
            ),
            "torch/split": TrainingImplementation(
                forward=torch_linear.linear_swiglu,
                inputs=split,
                leaves=leaves,
                cotangents=cotangents,
                copies_cotangents=False,
            ),
        },
        oracles={
            "coda": interleaved_oracle_impl,
            "torch/interleaved": interleaved_oracle_impl,
            "torch/split": split_oracle_impl,
        },
        outputs=outputs,
    )


OPS = {
    "linear_sigmoid": linear_sigmoid,
    "linear_swiglu": linear_swiglu,
    "linear_qknorm_rope": linear_qknorm_rope,
    "linear_cross_entropy": linear_cross_entropy,
    "linear_cross_entropy_forward": linear_cross_entropy_forward,
}
