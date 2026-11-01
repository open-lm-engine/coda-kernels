import torch
import functools
from coda.core.elementwise import functional as elementwise_functional
from coda.core.gemm import functional as gemm_functional
from coda.kernels.functional import cross_entropy as cross_entropy_functional

from benchmarks.backends import torch_composites
from benchmarks.backends import torch_ops
from benchmarks.utils import data
from benchmarks.utils import operands
from benchmarks.utils.cases import Case, Implementation, make_oracle_fn
from benchmarks.utils.data import Sampler, Shape


def gemm_lse(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples = operands.lm_head(
        shape=shape,
        rmsnorm=False,
        sampler=sampler,
        dtype=dtype,
    )
    inputs = {
        "A": samples["A"],
        "B": samples["B"],
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_lse,
            in_dims={
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
                fn=gemm_functional.gemm_lse,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_lse,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("logits", "lses"),
    )


def gemm_rmsnorm_lse(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples = operands.lm_head(
        shape=shape,
        rmsnorm=True,
        sampler=sampler,
        dtype=dtype,
    )
    inputs = {
        "A": samples["A"],
        "B": samples["B"],
        "rstd": samples["rstd"],
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rmsnorm_lse,
            in_dims={
                "A": 0,
                "B": None,
                "rstd": 0,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_rmsnorm_lse,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_rmsnorm_lse,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("logits", "lses"),
    )


def gemm_lse_select_logits(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples = operands.lm_head(
        shape=shape,
        rmsnorm=False,
        sampler=sampler,
        dtype=dtype,
    )
    inputs = {
        "A": samples["A"],
        "B": samples["B"],
        "target": samples["target"],
        "ignore_index": data.IGNORE_INDEX,
    }
    # CODA returns the lse only when asked for it
    coda_inputs = inputs.copy()
    coda_inputs["return_lse"] = True
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_lse_select_logits,
            in_dims={
                "A": 0,
                "B": None,
                "target": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_lse_select_logits,
                inputs=coda_inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_lse_select_logits,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("losses", "lses", "target_logits"),
    )


OPS = {
    "gemm_lse": gemm_lse,
    "gemm_rmsnorm_lse": gemm_rmsnorm_lse,
    "gemm_lse_select_logits": gemm_lse_select_logits,
    "cross_entropy_fwd_bwd": cross_entropy_fwd_bwd,
    "gemm_cross_entropy_fwd_bwd": gemm_cross_entropy_fwd_bwd,
    "gemm_rmsnorm_cross_entropy_fwd_bwd": gemm_rmsnorm_cross_entropy_fwd_bwd,
    "gemm_scalar_scale_lm_head": gemm_scalar_scale_lm_head,
    "linear_cross_entropy_forward": linear_cross_entropy_forward,
}
