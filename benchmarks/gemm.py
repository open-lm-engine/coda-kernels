import torch
from einops import rearrange, reduce
from coda.core.gemm import functional as gemm_functional

from benchmarks.backends import torch_ops
from benchmarks.backends.torch_ops import PairLayout
from benchmarks.utils import data
from benchmarks.utils import operands
from benchmarks.utils.cases import Case, Implementation, make_oracle_fn
from benchmarks.utils.data import Sampler, Shape


def gemm_sigmoid_fp32(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    # N is the head count, one gate per head
    samples = operands.linear_operands(
        shape=shape,
        out_features=shape.heads,
        sampler=sampler,
        dtype=dtype,
    )
    A = samples["A"]
    B = samples["B"]
    # both CODA and torch return the gates in fp32
    out_dtype = torch.float32
    out = sampler.randn(
        shape=(shape.tokens, shape.heads),
        dtype=out_dtype,
    )
    inputs = {
        "A": A,
        "B": B,
    }
    torch_inputs = {
        "A": A,
        "B": B,
        "out_dtype": out_dtype,
    }
    coda_inputs = {
        "A": A,
        "B": B,
        "out": out,
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_sigmoid,
            in_dims={
                "A": 0,
                "B": None,
            },
            out_dims=(0,),
            reduce_dims=(None,),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_sigmoid,
                inputs=coda_inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_sigmoid,
                inputs=torch_inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("out",),
    )


def gemm_rmsnorm(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_linear = operands.linear_operands(
        shape=shape,
        out_features=shape.dim,
        sampler=sampler,
        dtype=dtype,
    )
    # rescale A to the size a real norm input has, and compute its rstd
    samples_rmsnorm = operands.rmsnorm_operands(
        shape=shape,
        x=samples_linear["A"],
        rstd_nominal=shape.statistics.rstd,
        sampler=sampler,
        dtype=dtype,
    )
    A = samples_rmsnorm["A"]
    B = samples_linear["B"]
    rstd = samples_rmsnorm["rstd"]
    inputs = {
        "A": A,
        "B": B,
        "rstd": rstd,
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rmsnorm,
            in_dims={
                "A": 0,
                "B": None,
                "rstd": 0,
            },
            out_dims=(0,),
            reduce_dims=(None,),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_rmsnorm,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_rmsnorm,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("out",),
    )


def gemm_residual_partial_rmsnorm(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    weight = sampler.weight(
        shape=(shape.dim, shape.ffn),
        dtype=dtype,
    )
    A = sampler.randn(
        shape=(shape.tokens, shape.ffn),
        dtype=dtype,
    )
    residual = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=torch.float32,
    )
    gamma = sampler.gamma(
        size=shape.dim,
        dtype=dtype,
    )
    # scale the residual against A @ B
    residual_scaled = residual * shape.statistics.residual_scale
    C = residual_scaled.to(dtype=dtype)
    B = weight.mT
    inputs = {
        "A": A,
        "B": B,
        "C": C,
        "weight": gamma,
        "eps": data.NORM_EPS,
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_residual_partial_rmsnorm,
            in_dims={
                "A": 0,
                "B": None,
                "C": 0,
                "weight": None,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_residual_partial_rmsnorm,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_residual_partial_rmsnorm,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("pre", "post", "rstd"),
    )


def gemm_residual_partial_rmsnorm_bwd(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    weight = sampler.weight(
        shape=(2 * shape.ffn, shape.dim),
        dtype=dtype,
    )
    A = sampler.randn(
        shape=(shape.tokens, 2 * shape.ffn),
        dtype=dtype,
    )
    pre = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    gamma = sampler.gamma(
        size=shape.dim,
        dtype=dtype,
    )
    pre_fp32 = pre.to(dtype=torch.float32)
    rstd = torch_ops.rmsnorm_rstd(
        x=pre_fp32,
        eps=data.NORM_EPS,
    )
    # ZdZ, a per-row mean the rmsnorm backward needs, must agree with
    # the other inputs, so derive it from them rather than draw it
    dnormed = torch.mm(
        input=A,
        mat2=weight,
        out_dtype=torch.float32,
    )
    rstd_column = rearrange(
        tensor=rstd,
        pattern="m -> m 1",
    )
    # each element of dnormed times the norm's output,
    # pre * rstd * gamma
    ZdZ_terms = dnormed * pre_fp32 * rstd_column * gamma
    ZdZ = reduce(
        tensor=ZdZ_terms,
        pattern="m n -> m",
        reduction="mean",
    )
    # B is the projection's weight transposed, as the forward GEMM reads it
    B = weight.mT
    inputs = {
        "A": A,
        "B": B,
        "weight": gamma,
        "pre": pre,
        "ZdZ": ZdZ,
        "rstd": rstd,
    }
    outputs = ("dpre", "dW", "post")
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_residual_partial_rmsnorm_bwd,
            in_dims={
                "A": 0,
                "B": None,
                "weight": None,
                "pre": 0,
                "ZdZ": 0,
                "rstd": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, 0, None),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_residual_partial_rmsnorm_bwd,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_residual_partial_rmsnorm_bwd,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=outputs,
    )


def gemm_swiglu(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples = operands.linear_operands(
        shape=shape,
        out_features=2 * shape.ffn,
        sampler=sampler,
        dtype=dtype,
    )
    A = samples["A"]
    B = samples["B"]
    # two layouts of the gate and up columns: interleaved, as CODA takes
    # them, and split, all gates first, then all ups; torch runs both
    split_B = operands.split_swiglu_weight(
        B=B,
    )
    interleaved = {
        "A": A,
        "B": B,
        "layout": PairLayout.interleaved,
    }
    split = {
        "A": A,
        "B": split_B,
        "layout": PairLayout.split,
    }
    coda_inputs = {
        "A": A,
        "B": B,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_swiglu,
            in_dims={
                "A": 0,
                "B": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_swiglu,
            in_dims={
                "A": 0,
                "B": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_swiglu,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.gemm_swiglu,
                inputs=interleaved,
            ),
            "torch/split": Implementation(
                fn=torch_ops.gemm_swiglu,
                inputs=split,
            ),
        },
        oracles={
            "coda": interleaved_oracle_impl,
            "torch/interleaved": interleaved_oracle_impl,
            "torch/split": split_oracle_impl,
        },
        outputs=("preact", "postact"),
    )


def gemm_rmsnorm_swiglu(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_linear = operands.linear_operands(
        shape=shape,
        out_features=2 * shape.ffn,
        sampler=sampler,
        dtype=dtype,
    )
    samples_rmsnorm = operands.rmsnorm_operands(
        shape=shape,
        x=samples_linear["A"],
        rstd_nominal=shape.statistics.rstd,
        sampler=sampler,
        dtype=dtype,
    )
    A = samples_rmsnorm["A"]
    B = samples_linear["B"]
    rstd = samples_rmsnorm["rstd"]
    # reorder the weight for torch's split layout, as in gemm_swiglu
    split_B = operands.split_swiglu_weight(
        B=B,
    )
    interleaved = {
        "A": A,
        "B": B,
        "rstd": rstd,
        "layout": PairLayout.interleaved,
    }
    split = {
        "A": A,
        "B": split_B,
        "rstd": rstd,
        "layout": PairLayout.split,
    }
    coda_inputs = {
        "A": A,
        "B": B,
        "rstd": rstd,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rmsnorm_swiglu,
            in_dims={
                "A": 0,
                "B": None,
                "rstd": 0,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rmsnorm_swiglu,
            in_dims={
                "A": 0,
                "B": None,
                "rstd": 0,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_rmsnorm_swiglu,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.gemm_rmsnorm_swiglu,
                inputs=interleaved,
            ),
            "torch/split": Implementation(
                fn=torch_ops.gemm_rmsnorm_swiglu,
                inputs=split,
            ),
        },
        oracles={
            "coda": interleaved_oracle_impl,
            "torch/interleaved": interleaved_oracle_impl,
            "torch/split": split_oracle_impl,
        },
        outputs=("preact", "postact"),
    )


def gemm_swiglu_bwd_zdz(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    down = sampler.weight(
        shape=(shape.dim, shape.ffn),
        dtype=dtype,
    )
    A = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    Z = sampler.randn(
        shape=(shape.tokens, 2 * shape.ffn),
        dtype=dtype,
    )
    # B is the down weight transposed, as the forward GEMM reads it
    B = down.mT
    # ZdZ sums over Z's columns but divides by dim, the rmsnorm's width
    scale = 1.0 / shape.dim
    # torch's split layout needs Z's columns as all gates, then all ups
    split_Z = rearrange(
        tensor=Z,
        pattern="m (pair two) -> m (two pair)",
        two=2,
    ).contiguous()
    interleaved = {
        "A": A,
        "B": B,
        "Z": Z,
        "scale": scale,
        "layout": PairLayout.interleaved,
    }
    split = {
        "A": A,
        "B": B,
        "Z": split_Z,
        "scale": scale,
        "layout": PairLayout.split,
    }
    coda_inputs = {
        "A": A,
        "B": B,
        "Z": Z,
        "scale": scale,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_swiglu_bwd_zdz,
            in_dims={
                "A": 0,
                "B": None,
                "Z": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_swiglu_bwd_zdz,
            in_dims={
                "A": 0,
                "B": None,
                "Z": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_swiglu_bwd_zdz,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.gemm_swiglu_bwd_zdz,
                inputs=interleaved,
            ),
            "torch/split": Implementation(
                fn=torch_ops.gemm_swiglu_bwd_zdz,
                inputs=split,
            ),
        },
        oracles={
            "coda": interleaved_oracle_impl,
            "torch/interleaved": interleaved_oracle_impl,
            "torch/split": split_oracle_impl,
        },
        outputs=("dZ", "ZdZ", "out"),
    )


def gemm_xsa_bwd(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    inputs = operands.xsa_operands(
        shape=shape,
        sampler=sampler,
        dtype=dtype,
    )
    # torch takes v as a tensor of its own; CODA reads it in place
    torch_inputs = inputs.copy()
    torch_inputs["v"] = inputs["v"].contiguous()
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_xsa_bwd,
            in_dims={
                "x": 0,
                "v": 0,
                "A": 0,
                "B": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_xsa_bwd,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_xsa_bwd,
                inputs=torch_inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("dx", "dv"),
    )


OPS = {
    "gemm_sigmoid_fp32": gemm_sigmoid_fp32,
    "gemm_rmsnorm": gemm_rmsnorm,
    "gemm_residual_partial_rmsnorm": gemm_residual_partial_rmsnorm,
    "gemm_residual_partial_rmsnorm_bwd": gemm_residual_partial_rmsnorm_bwd,
    "gemm_swiglu": gemm_swiglu,
    "gemm_rmsnorm_swiglu": gemm_rmsnorm_swiglu,
    "gemm_swiglu_bwd_zdz": gemm_swiglu_bwd_zdz,
    "gemm_xsa_bwd": gemm_xsa_bwd,
}
