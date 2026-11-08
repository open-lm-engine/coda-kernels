import torch
from coda.kernels.functional import cross_entropy as cross_entropy_functional
from coda.kernels.functional import qknorm_rope as qknorm_rope_functional
from coda.kernels.functional import sigmoid as sigmoid_functional
from coda.kernels.functional import swiglu as swiglu_functional

from benchmarks.backends import torch_linear
from benchmarks.backends.torch_ops import CrossEntropyNumerics
from benchmarks.backends.torch_ops import PairLayout
from benchmarks.utils import data
from benchmarks.utils import operands
from benchmarks.utils.cases import (
    Case,
    Implementation,
    ImplementationFwdBwd,
    make_batch_mean_loss_fn,
    make_oracle_fn,
    make_step_fn,
)
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
    # the oracle runs torch's forward and backward in fp64
    torch_step = make_step_fn(
        fn=torch_linear.linear_sigmoid,
        leaves=leaves,
        cotangents=cotangents,
    )
    outputs = ("out", "dx", "dweight")
    # the oracle runs in chunks of rows and sums dweight over them
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
            leaves=leaves,
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": ImplementationFwdBwd(
                fn=sigmoid_functional.linear_sigmoid,
                inputs=inputs,
                leaves=leaves,
                cotangents=cotangents,
            ),
            "torch": ImplementationFwdBwd(
                fn=torch_linear.linear_sigmoid,
                inputs=inputs,
                leaves=leaves,
                cotangents=cotangents,
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
    # reorder the weight for torch's split layout: all gates, then all ups
    # the split helpers work on (K, N) weights, the layout GEMMs use; this
    # weight is (N, K), so transpose it before and after the call
    split_weight = operands.split_layout_swiglu_weight(
        B=weight.mT,
    ).mT
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
    # the oracles run torch's forward and backward in fp64
    torch_step = make_step_fn(
        fn=torch_linear.linear_swiglu,
        leaves=leaves,
        cotangents=cotangents,
    )
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
            leaves=leaves,
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
            leaves=leaves,
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": ImplementationFwdBwd(
                fn=swiglu_functional.linear_swiglu,
                inputs=coda_inputs,
                leaves=leaves,
                cotangents=cotangents,
            ),
            "torch/interleaved": ImplementationFwdBwd(
                fn=torch_linear.linear_swiglu,
                inputs=interleaved,
                leaves=leaves,
                cotangents=cotangents,
            ),
            "torch/split": ImplementationFwdBwd(
                fn=torch_linear.linear_swiglu,
                inputs=split,
                leaves=leaves,
                cotangents=cotangents,
            ),
        },
        oracles={
            "coda": interleaved_oracle_impl,
            "torch/interleaved": interleaved_oracle_impl,
            "torch/split": split_oracle_impl,
        },
        outputs=outputs,
    )


def linear_qknorm_rope(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    size_q = samples_rope["size_q"]
    size_k = samples_rope["size_k"]
    size_v = samples_rope["size_v"]
    size_qk = size_q + size_k
    samples_linear = operands.linear_qknorm_rope_operands(
        shape=shape,
        size_q=size_q,
        size_k=size_k,
        size_v=size_v,
        sampler=sampler,
        dtype=dtype,
    )
    weight = samples_linear["weight"]
    x = samples_linear["x"]
    gamma = samples_linear["gamma"]
    dq = samples_linear["dq"]
    dk = samples_linear["dk"]
    dv = samples_linear["dv"]
    # only q and k are rotated, not v
    frequencies = samples_rope["frequencies_qk"]
    # reorder q's and k's weight rows, gamma, dq and dk for torch's split
    # layout, which pairs element j with j + head_dim / 2
    # the split helpers work on (K, N) weights, the layout GEMMs use; this
    # weight is (N, K), so transpose it before and after the call
    split_weight = operands.split_layout_rope_weight(
        B=weight.mT,
        size_qk=size_qk,
        head_dim=shape.head_dim,
    ).mT
    # gamma is one q head's norm weight, then one k head's
    split_gamma = operands.split_layout_heads(
        x=gamma,
        num_heads=2,
        head_dim=shape.head_dim,
    )
    split_dq = operands.split_layout_heads(
        x=dq,
        num_heads=shape.heads,
        head_dim=shape.head_dim,
    )
    split_dk = operands.split_layout_heads(
        x=dk,
        num_heads=shape.kv_heads,
        head_dim=shape.head_dim,
    )
    leaves = ("x", "weight", "gamma")
    cotangents = ("dq", "dk", "dv")
    # rope and qk-norm arguments shared by every implementation
    shared_inputs = {
        "positions": samples_rope["positions"],
        "frequencies": frequencies,
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
    }
    coda_inputs = {
        **shared_inputs,
        "x": x,
        "weight": weight,
        "gamma": gamma,
        "dq": dq,
        "dk": dk,
        "dv": dv,
    }
    # torch computes cos and sin from positions and frequencies
    interleaved = {
        **coda_inputs,
        "cos": None,
        "sin": None,
        "layout": PairLayout.interleaved,
    }
    split = {
        **shared_inputs,
        "x": x,
        "weight": split_weight,
        "gamma": split_gamma,
        "dq": split_dq,
        "dk": split_dk,
        "dv": dv,
        "cos": None,
        "sin": None,
        "layout": PairLayout.split,
    }
    # torch with its rope tables built once, outside the timed step
    cos, sin = operands.cache_rope_cos_sin(
        positions=samples_rope["positions"],
        frequencies=frequencies,
        head_dim=shape.head_dim,
    )
    interleaved_cached = {
        **interleaved,
        "positions": None,
        "frequencies": None,
        "cos": cos,
        "sin": sin,
    }
    split_cached = {
        **split,
        "positions": None,
        "frequencies": None,
        "cos": cos,
        "sin": sin,
    }
    # the oracles run torch's forward and backward in fp64
    torch_step = make_step_fn(
        fn=torch_linear.linear_qknorm_rope,
        leaves=leaves,
        cotangents=cotangents,
    )
    outputs = ("q", "k", "v", "dx", "dweight", "dgamma")
    oracle_fn = make_oracle_fn(
        fn=torch_step,
        in_dims={
            "x": 0,
            "weight": None,
            "gamma": None,
            "dq": 0,
            "dk": 0,
            "dv": 0,
            "positions": 0,
            "frequencies": None,
        },
        out_dims=(0, 0, 0, 0, 0, 0),
        reduce_dims=(None, None, None, None, 0, 0),
        leaves=leaves,
    )
    interleaved_oracle_impl = Implementation(
        fn=oracle_fn,
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=oracle_fn,
        inputs=split,
    )
    return Case(
        implementations={
            "coda": ImplementationFwdBwd(
                fn=qknorm_rope_functional.linear_qknorm_rope,
                inputs=coda_inputs,
                leaves=leaves,
                cotangents=cotangents,
            ),
            "torch/interleaved": ImplementationFwdBwd(
                fn=torch_linear.linear_qknorm_rope,
                inputs=interleaved,
                leaves=leaves,
                cotangents=cotangents,
            ),
            "torch/interleaved/cached": ImplementationFwdBwd(
                fn=torch_linear.linear_qknorm_rope,
                inputs=interleaved_cached,
                leaves=leaves,
                cotangents=cotangents,
            ),
            "torch/split": ImplementationFwdBwd(
                fn=torch_linear.linear_qknorm_rope,
                inputs=split,
                leaves=leaves,
                cotangents=cotangents,
            ),
            "torch/split/cached": ImplementationFwdBwd(
                fn=torch_linear.linear_qknorm_rope,
                inputs=split_cached,
                leaves=leaves,
                cotangents=cotangents,
            ),
        },
        oracles={
            "coda": interleaved_oracle_impl,
            "torch/interleaved": interleaved_oracle_impl,
            "torch/interleaved/cached": interleaved_oracle_impl,
            "torch/split": split_oracle_impl,
            "torch/split/cached": split_oracle_impl,
        },
        outputs=outputs,
    )


def linear_cross_entropy(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples = operands.lm_head_operands(
        shape=shape,
        rmsnorm=False,
        sampler=sampler,
        dtype=dtype,
    )
    x = samples["A"]
    # B is (dim, vocab); linear_cross_entropy takes the (vocab, dim) weight
    weight = samples["B"].mT
    target = samples["target"]
    # backward starts from a loss gradient of 1
    grad_loss = torch.ones(
        size=(),
        dtype=torch.float32,
        device="cuda",
    )
    leaves = ("x", "weight")
    cotangents = ("grad_loss",)
    inputs = {
        "x": x,
        "weight": weight,
        "target": target,
        "grad_loss": grad_loss,
    }
    # loss logits: fp32 for torch, following TorchTitan, and fp64 for the oracle
    torch_inputs = {
        **inputs,
        "logits_dtype": torch.float32,
        # the collapse-free loss and gradient CODA's kernel forms
        "cross_entropy_numerics": CrossEntropyNumerics.precise,
    }
    # torch's own cross entropy, reported apart
    torch_standard_ce_inputs = torch_inputs.copy()
    torch_standard_ce_inputs["cross_entropy_numerics"] = CrossEntropyNumerics.standard
    # the oracle sums its row chunks' losses, so each chunk divides by
    # the count of targets not ignored in the whole batch
    oracle_inputs = torch_inputs.copy()
    oracle_inputs["logits_dtype"] = torch.float64
    oracle_inputs["count"] = operands.count_non_ignored_targets(
        target=target,
    )
    oracle_forward = make_batch_mean_loss_fn(
        fn=torch_linear.linear_cross_entropy,
    )
    oracle_step = make_step_fn(
        fn=oracle_forward,
        leaves=leaves,
        cotangents=cotangents,
    )
    outputs = ("loss", "dx", "dweight")
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=oracle_step,
            in_dims={
                "x": 0,
                "weight": None,
                "target": 0,
                "grad_loss": None,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(0, None, 0),
            leaves=leaves,
        ),
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            "coda": ImplementationFwdBwd(
                fn=cross_entropy_functional.linear_cross_entropy,
                inputs=inputs,
                leaves=leaves,
                cotangents=cotangents,
            ),
            "torch": ImplementationFwdBwd(
                fn=torch_linear.linear_cross_entropy,
                inputs=torch_inputs,
                leaves=leaves,
                cotangents=cotangents,
            ),
            "torch/standard-ce": ImplementationFwdBwd(
                fn=torch_linear.linear_cross_entropy,
                inputs=torch_standard_ce_inputs,
                leaves=leaves,
                cotangents=cotangents,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
            "torch/standard-ce": oracle_impl,
        },
        outputs=outputs,
    )


def linear_cross_entropy_forward(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples = operands.lm_head_operands(
        shape=shape,
        rmsnorm=False,
        sampler=sampler,
        dtype=dtype,
    )
    x = samples["A"]
    # B is (dim, vocab); linear_cross_entropy takes the (vocab, dim) weight
    weight = samples["B"].mT
    target = samples["target"]
    inputs = {
        "x": x,
        "weight": weight,
        "target": target,
    }
    # loss logits: fp32 for torch, following TorchTitan, and fp64 for the oracle
    torch_inputs = {
        **inputs,
        "logits_dtype": torch.float32,
        # the collapse-free loss and gradient CODA's kernel forms
        "cross_entropy_numerics": CrossEntropyNumerics.precise,
    }
    # torch's own cross entropy, reported apart
    torch_standard_ce_inputs = torch_inputs.copy()
    torch_standard_ce_inputs["cross_entropy_numerics"] = CrossEntropyNumerics.standard
    # the oracle sums its row chunks' losses, so each chunk divides by
    # the count of targets not ignored in the whole batch
    oracle_inputs = torch_inputs.copy()
    oracle_inputs["logits_dtype"] = torch.float64
    oracle_inputs["count"] = operands.count_non_ignored_targets(
        target=target,
    )
    oracle_forward = make_batch_mean_loss_fn(
        fn=torch_linear.linear_cross_entropy,
    )
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=oracle_forward,
            in_dims={
                "x": 0,
                "weight": None,
                "target": 0,
            },
            out_dims=(0,),
            # sum the chunks' losses
            reduce_dims=(0,),
        ),
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=cross_entropy_functional.linear_cross_entropy_forward,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_linear.linear_cross_entropy,
                inputs=torch_inputs,
            ),
            "torch/standard-ce": Implementation(
                fn=torch_linear.linear_cross_entropy,
                inputs=torch_standard_ce_inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
            "torch/standard-ce": oracle_impl,
        },
        outputs=("loss",),
    )


OPS = {
    "linear_sigmoid": linear_sigmoid,
    "linear_swiglu": linear_swiglu,
    "linear_qknorm_rope": linear_qknorm_rope,
    "linear_cross_entropy": linear_cross_entropy,
    "linear_cross_entropy_forward": linear_cross_entropy_forward,
}
