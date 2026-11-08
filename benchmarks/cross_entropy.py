import torch
from coda.core.elementwise import functional as elementwise_functional
from coda.core.gemm import functional as gemm_functional
from coda.kernels.functional import cross_entropy as cross_entropy_functional

from benchmarks.backends import torch_ops
from benchmarks.backends.torch_ops import CrossEntropyNumerics
from benchmarks.utils import data
from benchmarks.utils import operands
from benchmarks.utils.cases import Case, Implementation, make_oracle_fn
from benchmarks.utils.data import Sampler, Shape

# the scale of the LM head backward, a mean loss's gradient; bf16 cannot
# hold 1/3072 exactly, so rounding it shows up as error
DLOSS = 1.0 / 3072


def gemm_lse(
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
    inputs = {
        "A": samples["A"],
        "B": samples["B"],
    }
    # loss logits: fp32 for torch, following TorchTitan, and fp64 for the oracle
    torch_inputs = {
        **inputs,
        "logits_dtype": torch.float32,
    }
    oracle_inputs = torch_inputs.copy()
    oracle_inputs["logits_dtype"] = torch.float64
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
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_lse,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_lse,
                inputs=torch_inputs,
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
    samples = operands.lm_head_operands(
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
    # loss logits: fp32 for torch, following TorchTitan, and fp64 for the oracle
    torch_inputs = {
        **inputs,
        "logits_dtype": torch.float32,
    }
    oracle_inputs = torch_inputs.copy()
    oracle_inputs["logits_dtype"] = torch.float64
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
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_rmsnorm_lse,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_rmsnorm_lse,
                inputs=torch_inputs,
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
    samples = operands.lm_head_operands(
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
    coda_inputs = {
        **inputs,
        "return_lse": True,
    }
    # loss logits: fp32 for torch, following TorchTitan, and fp64 for the oracle
    torch_inputs = {
        **inputs,
        "logits_dtype": torch.float32,
    }
    oracle_inputs = torch_inputs.copy()
    oracle_inputs["logits_dtype"] = torch.float64
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
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_lse_select_logits,
                inputs=coda_inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_lse_select_logits,
                inputs=torch_inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("losses", "lses", "target_logits"),
    )


def cross_entropy_fwd_bwd(
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
    # realistic logits, in fp32 as gemm_lse stores them
    logits = torch.mm(
        input=samples["A"],
        mat2=samples["B"],
        out_dtype=torch.float32,
    )
    # their lse, computed in fp64 to be exact, then rounded once to fp32
    logits_fp64 = logits.to(dtype=torch.float64)
    exact_lses = torch.logsumexp(
        input=logits_fp64,
        dim=-1,
    )
    lses = exact_lses.to(dtype=torch.float32)
    # CODA takes int32 targets, as its GEMM epilogue passes them on; the
    # torch reference takes the int64 ones
    target = samples["target"]
    target_int32 = target.to(dtype=torch.int32)
    coda_inputs = {
        "logits": logits,
        "lses": lses,
        "target": target_int32,
        "ignore_index": data.IGNORE_INDEX,
        "dlogits_dtype": dtype,
        "return_zdz": True,
    }
    # torch's loss logits in fp32, following TorchTitan
    torch_inputs = {
        "logits": logits,
        "target": target,
        "ignore_index": data.IGNORE_INDEX,
        "return_zdz": True,
        "logits_dtype": torch.float32,
        # the collapse-free loss and gradient CODA's kernel forms
        "cross_entropy_numerics": CrossEntropyNumerics.precise,
    }
    # torch's own cross entropy, reported apart
    torch_standard_ce_inputs = torch_inputs.copy()
    torch_standard_ce_inputs["cross_entropy_numerics"] = CrossEntropyNumerics.standard
    # the oracle runs on the logits in fp64, so it takes no logits_dtype
    oracle_inputs = {
        "logits": logits,
        "target": target_int32,
        "ignore_index": data.IGNORE_INDEX,
        "return_zdz": True,
        "cross_entropy_numerics": CrossEntropyNumerics.precise,
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.cross_entropy_fwd_bwd,
            in_dims={
                "logits": 0,
                "target": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=elementwise_functional.cross_entropy_fwd_bwd,
                inputs=coda_inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.cross_entropy_fwd_bwd_bench,
                inputs=torch_inputs,
            ),
            "torch/standard-ce": Implementation(
                fn=torch_ops.cross_entropy_fwd_bwd_bench,
                inputs=torch_standard_ce_inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
            "torch/standard-ce": oracle_impl,
        },
        outputs=("losses", "dlogits", "zdz"),
    )


def gemm_cross_entropy_fwd_bwd(
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
    inputs = {
        "A": samples["A"],
        "B": samples["B"],
        "target": samples["target"],
        "ignore_index": data.IGNORE_INDEX,
        "return_zdz": True,
    }
    # loss logits: fp32 for torch, following TorchTitan, and fp64 for the oracle
    torch_inputs = {
        **inputs,
        "logits_dtype": torch.float32,
        # the collapse-free loss and gradient CODA's kernel forms
        "cross_entropy_numerics": CrossEntropyNumerics.precise,
    }
    oracle_inputs = torch_inputs.copy()
    oracle_inputs["logits_dtype"] = torch.float64
    # torch's own cross entropy, reported apart
    torch_standard_ce_inputs = torch_inputs.copy()
    torch_standard_ce_inputs["cross_entropy_numerics"] = CrossEntropyNumerics.standard
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_cross_entropy_fwd_bwd,
            in_dims={
                "A": 0,
                "B": None,
                "target": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=cross_entropy_functional.gemm_cross_entropy_fwd_bwd,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_cross_entropy_fwd_bwd_bench,
                inputs=torch_inputs,
            ),
            "torch/standard-ce": Implementation(
                fn=torch_ops.gemm_cross_entropy_fwd_bwd_bench,
                inputs=torch_standard_ce_inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
            "torch/standard-ce": oracle_impl,
        },
        outputs=("losses", "dlogits", "zdz"),
    )


def gemm_rmsnorm_cross_entropy_fwd_bwd(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples = operands.lm_head_operands(
        shape=shape,
        rmsnorm=True,
        sampler=sampler,
        dtype=dtype,
    )
    inputs = {
        "A": samples["A"],
        "B": samples["B"],
        "rstd": samples["rstd"],
        "target": samples["target"],
        "ignore_index": data.IGNORE_INDEX,
        "return_zdz": True,
    }
    # loss logits: fp32 for torch, following TorchTitan, and fp64 for the oracle
    torch_inputs = {
        **inputs,
        "logits_dtype": torch.float32,
        # the collapse-free loss and gradient CODA's kernel forms
        "cross_entropy_numerics": CrossEntropyNumerics.precise,
    }
    oracle_inputs = torch_inputs.copy()
    oracle_inputs["logits_dtype"] = torch.float64
    # torch's own cross entropy, reported apart
    torch_standard_ce_inputs = torch_inputs.copy()
    torch_standard_ce_inputs["cross_entropy_numerics"] = CrossEntropyNumerics.standard
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rmsnorm_cross_entropy_fwd_bwd,
            in_dims={
                "A": 0,
                "B": None,
                "rstd": 0,
                "target": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=cross_entropy_functional.gemm_rmsnorm_cross_entropy_fwd_bwd,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_rmsnorm_cross_entropy_fwd_bwd_bench,
                inputs=torch_inputs,
            ),
            "torch/standard-ce": Implementation(
                fn=torch_ops.gemm_rmsnorm_cross_entropy_fwd_bwd_bench,
                inputs=torch_standard_ce_inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
            "torch/standard-ce": oracle_impl,
        },
        outputs=("losses", "dlogits", "zdz"),
    )


def gemm_scalar_scale_lm_head(
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
    A = operands.lm_head_dlogits(
        A=samples["A"],
        B=samples["B"],
        target=samples["target"],
        dtype=dtype,
    )
    # the LM head's (vocab, dim) weight, stored row-major
    B = samples["B"].mT
    # DLOSS on the GPU in fp32; being 0-d, it keeps torch's product in bf16
    alpha = torch.full(
        size=(),
        fill_value=DLOSS,
        dtype=torch.float32,
        device="cuda",
    )
    inputs = {
        "A": A,
        "B": B,
        "alpha": alpha,
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_scalar_scale,
            in_dims={
                "A": 0,
                "B": None,
                "alpha": None,
            },
            out_dims=(0,),
            reduce_dims=(None,),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_scalar_scale,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_ops.gemm_scalar_scale,
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
    "gemm_lse": gemm_lse,
    "gemm_rmsnorm_lse": gemm_rmsnorm_lse,
    "gemm_lse_select_logits": gemm_lse_select_logits,
    "cross_entropy_fwd_bwd": cross_entropy_fwd_bwd,
    "gemm_cross_entropy_fwd_bwd": gemm_cross_entropy_fwd_bwd,
    "gemm_rmsnorm_cross_entropy_fwd_bwd": gemm_rmsnorm_cross_entropy_fwd_bwd,
    "gemm_scalar_scale_lm_head": gemm_scalar_scale_lm_head,
}
