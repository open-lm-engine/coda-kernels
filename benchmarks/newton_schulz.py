import torch
from einops import rearrange
from coda.core.gemm import functional as gemm_functional

from benchmarks.backends import torch_ops
from benchmarks.utils import operands
from benchmarks.utils.cases import Case, Implementation, make_oracle_fn
from benchmarks.utils.data import Sampler, Shape

# Muon's Newton-Schulz coefficients (a, b, c), as in the original Muon
# (github.com/KellerJordan/Muon) and torch.optim.Muon's defaults
NS_A = 3.4445
NS_B = -4.7750
NS_C = 2.0315
# X is the matrix Muon's fourth Newton-Schulz step works on, made by
# running the first three
X_STEPS = 3


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
    # a batch of one matrix, as the op takes batches
    X_batched = rearrange(
        tensor=X,
        pattern="m k -> 1 m k",
    )
    inputs = {"A": X_batched}
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
    # a batch of one matrix, as the op takes batches
    X_batched = rearrange(
        tensor=X,
        pattern="m k -> 1 m k",
    )
    # the Gram matrix G = X X^T, from which the op forms b G + c G G
    A = gemm_functional.gemm_symmetric(A=X_batched)
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


def gemm_add_dual_precision(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    weight = sampler.weight(
        shape=(shape.dim, shape.ffn),
        dtype=torch.float32,
    )
    iterate = operands.muon_iterate(
        weight=weight,
        steps=X_STEPS,
        coefficients=(NS_A, NS_B, NS_C),
    )
    # the step's polynomial P = b G + c G G with G = X X^T, from the fp64
    # iterate
    gram = torch.mm(
        input=iterate,
        mat2=iterate.mT,
    )
    gram_squared = torch.mm(
        input=gram,
        mat2=gram,
    )
    polynomial = NS_B * gram + NS_C * gram_squared
    X = iterate.to(dtype=torch.float32)
    # the op computes a X + P X: P and X enter the GEMM in dtype, and X is
    # added in fp32
    A = polynomial.to(dtype=dtype)
    B = X.to(dtype=dtype)
    # batches of one matrix, as the op takes batches
    A_batched = rearrange(
        tensor=A,
        pattern="m k -> 1 m k",
    )
    B_batched = rearrange(
        tensor=B,
        pattern="k n -> 1 k n",
    )
    C_batched = rearrange(
        tensor=X,
        pattern="m n -> 1 m n",
    )
    inputs = {
        "A": A_batched,
        "B": B_batched,
        "C": C_batched,
        "C_scale": NS_A,
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_add_dual_precision,
            in_dims={
                "A": None,
                "B": None,
                "C": None,
            },
            out_dims=(None, None),
            reduce_dims=(None, None),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_add_dual_precision,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_add_dual_precision,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("out", "out_auxiliary"),
    )


OPS = {
    "gemm_symmetric": gemm_symmetric,
    "gemm_symmetric[polynomial]": gemm_symmetric_polynomial,
    "gemm_add_dual_precision": gemm_add_dual_precision,
}
