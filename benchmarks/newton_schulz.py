import torch
from einops import rearrange
from coda.core.gemm import functional as gemm_functional

from benchmarks.backends import torch_ops
from benchmarks.utils.cases import Case, Implementation, make_oracle_fn
from benchmarks.utils.data import Sampler, Shape


def gemm_symmetric(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    X = sampler.weight(
        shape=(shape.dim, shape.ffn),
        dtype=dtype,
    )
    # a stack of one, as Newton-Schulz calls it on stacks of matrices
    X_stack = rearrange(
        tensor=X,
        pattern="m k -> 1 m k",
    )
    inputs = {"A": X_stack}
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_symmetric,
            in_dims={"A": None},
            out_dims=(None,),
            reduce_dims=(None,),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_symmetric,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_symmetric,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("out",),
    )


def gemm_symmetric_polynomial(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    X = sampler.weight(
        shape=(shape.dim, shape.ffn),
        dtype=dtype,
    )
    # a stack of one, as Newton-Schulz calls it on stacks of matrices
    X_stack = rearrange(
        tensor=X,
        pattern="m k -> 1 m k",
    )
    A = gemm_functional.gemm_symmetric(A=X_stack)
    inputs = {
        "A": A,
        "B": A,
        "C": A,
        "D_scale": NS_C,
        "C_scale": NS_B,
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_symmetric,
            in_dims={
                "A": None,
                "B": None,
                "C": None,
            },
            out_dims=(None,),
            reduce_dims=(None,),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_symmetric,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_symmetric,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("out",),
    )



OPS = {
    "gemm_symmetric": gemm_symmetric,
    "gemm_symmetric[polynomial]": gemm_symmetric_polynomial,
    "gemm_add_dual_precision": gemm_add_dual_precision,
}
