import time
import math
import torch
import triton
import argparse
from typing import NamedTuple
from quack import cache_utils
from quack import gemm_interface as quack_gemm
from quack.autotuner import autotune, AutotuneConfig

from coda.core.examples.gemm import gemm_op

from coda.kernels.gens import gpt as gens
# `gpt2` is more optimized and less precise
from coda.kernels.refs import gpt2 as refs
from coda.kernels.tests import gpt as tests

from coda.models import ops
from coda.models import ops2

from benchmarks.deprecated import quack_utils
from benchmarks.deprecated import trainstation_utils
from benchmarks import bench_utils

cache_utils.CACHE_ENABLED = False
torch._dynamo.config.capture_scalar_outputs = True


class UseQuackConfig(NamedTuple):
    use_quack: bool


UseQuackConfigOptions = [
    AutotuneConfig(
        config=UseQuackConfig(
            use_quack=qck,
        ),
    )
    for qck in [False, True]
]


@torch.compile(fullgraph=True, dynamic=False)
def gemm(A: torch.Tensor, B: torch.Tensor, quack: bool, backward: bool = False) -> torch.Tensor:
    if backward:
        B = B.mT

    if not quack:
        return torch.mm(A, B)
    else:
        return quack_gemm.gemm(A, B, tuned=True)


def benchmark_gemm(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
    transpose: bool = False,
) -> None:

    flops = 2 * M * N * K
    A = torch.randn(M, K, dtype=dtype, device=device) / math.sqrt(K)
    if transpose:
        B = torch.randn(N, K, dtype=dtype, device=device).mT
    else:
        B = torch.randn(K, N, dtype=dtype, device=device)

    fn0 = lambda: torch.mm(A, B)
    fn1 = lambda: gemm_op(A, B, use_tuned=True)
    fn2 = lambda: quack_gemm.gemm(A, B, tuned=True)

    for i, fn in enumerate([fn0, fn1, fn2]):
        time.sleep(0.5)
        t = triton.testing.do_bench(fn, warmup=warmup, rep=repeats)
        time.sleep(0.5)
        tflops = flops / (t * 1e9)  # Convert to TFlops
        print(f"[{i}] Average time: {t:.3f} ms, TFLOPS: {tflops:.1f}")


def benchmark_gemm_residual_partial_rmsnorm(
    M: int,
    N: int,
    K: int,
    block_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_residual_partial_rmsnorm(
        M=M,
        N=N,
        K=K,
        block_size=block_size,
        dtype=dtype,
    )
    inputs, inputs_ref, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )
    fn_dict = {
        "rapier": lambda: gens.gemm_residual_partial_rmsnorm(
            A=inputs["A"],
            B=inputs["B"],
            C=inputs["C"],
            W=inputs["W"],
            block_size=block_size,
        ),
        "ref": lambda: refs.gemm_residual_partial_rmsnorm(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            C=inputs_ref["C"],
            W=inputs_ref["W"],
            block_size=block_size,
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
        ),
        "expert": lambda: quack_utils.gemm_rms(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            C=inputs_ref["C"],
            norm_weight=inputs_ref["W"],
            block_size=block_size,
            tuned=True,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gemm_rmsnorm(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_rmsnorm(
        M=M,
        N=N,
        K=K,
        dtype=dtype,
    )
    inputs, inputs_ref, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )
    fn_dict = {
        "rapier": lambda: gens.gemm_rmsnorm(
            A=inputs["A"],
            B=inputs["B"],
            R=inputs["R"],
        ),
        "ref": lambda: refs.gemm_rmsnorm(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
        ),
        "expert": lambda: quack_gemm.gemm_norm_act(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            rstd=inputs_ref["R"],
            activation=None,
            store_preact=False,
            tuned=True,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gemm_rmsnorm_swiglu(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_rmsnorm_swiglu(
        M=M,
        N=N,
        K=K,
        dtype=dtype,
    )
    inputs, inputs_ref, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )
    fn_dict = {
        "rapier": lambda: gens.gemm_rmsnorm_swiglu(
            A=inputs["A"],
            B=inputs["B"],
            R=inputs["R"],
        ),
        "ref": lambda: refs.gemm_rmsnorm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
        ),
        "expert": lambda: quack_gemm.gemm_norm_act(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            rstd=inputs_ref["R"],
            activation="swiglu",
            store_preact=True,
            tuned=True,
        ),
        "liger": lambda: ops2.gemm_rmsnorm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            backend="liger",
            use_compile=False,
        ),
        "liger-compile": lambda: ops2.gemm_rmsnorm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            backend="liger",
            use_compile=True,
        ),
        "finfer": lambda: ops2.gemm_rmsnorm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            backend="flashinfer",
            use_compile=False,
        ),
        "finfer-compile": lambda: ops2.gemm_rmsnorm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            backend="flashinfer",
            use_compile=True,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gemm_rmsnorm_rope(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_rmsnorm_rope(
        M=M,
        N=N,
        K=K,
        dtype=dtype,
    )
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        rope=True,
    )
    cos_sin_finfer = torch.cat(
        [
            inputs_ref_fp32["cos"],
            inputs_ref_fp32["sin"],
        ],
        dim=-1,
    ).contiguous()
    fn_dict = {
        "rapier": lambda: gens.gemm_rmsnorm_rope(
            A=inputs["A"],
            B=inputs["B"],
            R=inputs["R"],
            cos_sin=inputs["cos_sin"],
        ),
        "ref": lambda: refs.gemm_rmsnorm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            cos_sin=inputs_ref["cos_sin"],
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
        ),
        "liger": lambda: ops2.gemm_rmsnorm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="liger",
            use_compile=False,
        ),
        "liger-compile": lambda: ops2.gemm_rmsnorm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="liger",
            use_compile=True,
        ),
        "finfer": lambda: ops2.gemm_rmsnorm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="flashinfer",
            use_compile=False,
        ),
        "finfer-compile": lambda: ops2.gemm_rmsnorm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="flashinfer",
            use_compile=True,
        ),
        "finfer2": lambda: ops2.gemm_rmsnorm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="flashinfer2",
            use_compile=False,
        ),
        "finfer2-compile": lambda: ops2.gemm_rmsnorm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="flashinfer2",
            use_compile=True,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gemm_rmsnorm_partial_cross_entropy(
    M: int,
    K: int,
    block_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_rmsnorm_partial_cross_entropy(
        M=M,
        K=K,
        block_size=block_size,
        vocab_size=vocab_size,
        dtype=dtype,
    )
    inputs, inputs_ref, _, auxiliary = tests.create_gpt_inputs(
        M=M,
        N=vocab_size,
        K=K,
        vocab_size=vocab_size,
        dtype=dtype,
    )
    fn_dict = {
        "rapier": lambda: gens.gemm_rmsnorm_partial_cross_entropy(
            A=inputs["A"],
            B=inputs["B"],
            R=inputs["R"],
            targets=auxiliary["targets"],
            block_size=block_size,
        ),
        "ref": lambda: refs.gemm_rmsnorm_partial_cross_entropy(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            R=inputs_ref["R"],
            targets=auxiliary["targets"],
            block_size=block_size,
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gemm_residual_partial_rmsnorm_bwd(
    M: int,
    N: int,
    K: int,
    block_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_residual_partial_rmsnorm_bwd(
        M=M,
        N=N,
        K=K,
        block_size=block_size,
        dtype=dtype,
    )
    inputs, inputs_ref, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        backward=True,
        transpose_B=True,
    )
    fn_dict = {
        "rapier": lambda: gens.gemm_residual_partial_rmsnorm_bwd(
            A=inputs["A"],
            B=inputs["B"],
            C=inputs["C"],
            W=inputs["W"],
            R=inputs["R"],
            ZdZ=inputs["ZdZ"],
            O=inputs["O"],
            block_size=block_size,
        ),
        "ref": lambda: refs.gemm_residual_partial_rmsnorm_bwd(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            C=inputs_ref["C"],
            W=inputs_ref["W"],
            R=inputs_ref["R"],
            ZdZ=inputs_ref["ZdZ"],
            O=inputs_ref["O"],
            block_size=block_size,
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
            backward=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
            backward=True,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gemm_partial_swiglu_bwd(
    M: int,
    N: int,
    K: int,
    block_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_partial_swiglu_bwd(
        M=M,
        N=N,
        K=K,
        block_size=block_size,
        dtype=dtype,
    )
    inputs, inputs_ref, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        backward=True,
        transpose_B=True,
    )
    # liger uses transposed weight
    B_transposed = inputs_ref["B"].mT.contiguous()
    fn_dict = {
        "rapier": lambda: gens.gemm_partial_swiglu_bwd(
            A=inputs["A"],
            B=inputs["B"],
            Z=inputs["Z"],
            block_size=block_size,
        ),
        "ref": lambda: refs.gemm_partial_swiglu_bwd(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            Z=inputs_ref["Z"],
            block_size=block_size,
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
            backward=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
            backward=True,
        ),
        "liger": lambda: ops2.gemm_partial_swiglu_bwd(
            A=inputs_ref["A"],
            B=B_transposed,
            Z=inputs_ref["Z"],
            block_size=block_size,
            backend="liger",
            use_compile=False,
        ),
        "liger-compile": lambda: ops2.gemm_partial_swiglu_bwd(
            A=inputs_ref["A"],
            B=B_transposed,
            Z=inputs_ref["Z"],
            block_size=block_size,
            backend="liger",
            use_compile=True,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gemm_rope(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_rope(
        M=M,
        N=N,
        K=K,
        dtype=dtype,
    )
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        rope=True,
    )
    cos_sin_finfer = torch.cat(
        [
            inputs_ref_fp32["cos"],
            inputs_ref_fp32["sin"],
        ],
        dim=-1,
    ).contiguous()
    fn_dict = {
        "rapier": lambda: gens.gemm_rope(
            A=inputs["A"],
            B=inputs["B"],
            cos_sin=inputs["cos_sin"],
        ),
        "ref": lambda: refs.gemm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            cos_sin=inputs_ref["cos_sin"],
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
        ),
        "liger": lambda: ops2.gemm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="liger",
            use_compile=False,
        ),
        "liger-compile": lambda: ops2.gemm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="liger",
            use_compile=True,
        ),
        "finfer": lambda: ops2.gemm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="flashinfer",
            use_compile=False,
        ),
        "finfer-compile": lambda: ops2.gemm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="flashinfer",
            use_compile=True,
        ),
        "finfer2": lambda: ops2.gemm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="flashinfer2",
            use_compile=False,
        ),
        "finfer2-compile": lambda: ops2.gemm_rope(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            cos=inputs_ref["cos"],
            sin=inputs_ref["sin"],
            cos_sin=cos_sin_finfer,
            positions=auxiliary["positions"],
            backend="flashinfer2",
            use_compile=True,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gemm_swiglu(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_swiglu(
        M=M,
        N=N,
        K=K,
        dtype=dtype,
    )
    inputs, inputs_ref, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )
    fn_dict = {
        "rapier": lambda: gens.gemm_swiglu(
            A=inputs["A"],
            B=inputs["B"],
        ),
        "ref": lambda: refs.gemm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
        ),
        "expert": lambda: quack_gemm.gemm_act(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            activation="swiglu",
            store_preact=True,
            tuned=True,
        ),
        "liger": lambda: ops2.gemm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            backend="liger",
            use_compile=False,
        ),
        "liger-compile": lambda: ops2.gemm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            backend="liger",
            use_compile=True,
        ),
        "finfer": lambda: ops2.gemm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            backend="flashinfer",
            use_compile=False,
        ),
        "finfer-compile": lambda: ops2.gemm_swiglu(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            backend="flashinfer",
            use_compile=True,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gemm_cross_entropy(
    M: int,
    K: int,
    block_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    tests.test_gemm_partial_cross_entropy(
        M=M,
        K=K,
        block_size=block_size,
        vocab_size=vocab_size,
        dtype=dtype,
    )
    inputs, inputs_ref, _, auxiliary = tests.create_gpt_inputs(
        M=M,
        N=vocab_size,
        K=K,
        vocab_size=vocab_size,
        dtype=dtype,
    )
    # liger cross entropy used transposed weight
    B_transposed = inputs_ref["B"].mT.contiguous()

    @autotune(configs=UseQuackConfigOptions, key=["block_size_"], cache_results=False)
    def _gens_gemm_cross_entropy(A: torch.Tensor, B: torch.Tensor, targets: torch.Tensor, block_size_: int, config: UseQuackConfig) -> torch.Tensor:
        _, logits_tgt, logits_lse = gens.gemm_partial_cross_entropy(
            A=A,
            B=B,
            targets=targets,
            block_size=block_size_,
        )
        return ops.cross_entropy_forward(logits_tgt=logits_tgt, logits_lse=logits_lse, targets=targets, use_quack=config.use_quack)

    @autotune(configs=UseQuackConfigOptions, key=["block_size_"], cache_results=False)
    def _refs_gemm_cross_entropy(A: torch.Tensor, B: torch.Tensor, targets: torch.Tensor, block_size_: int, config: UseQuackConfig) -> torch.Tensor:
        _, logits_tgt, logits_lse = refs.gemm_partial_cross_entropy(
            A=A,
            B=B,
            targets=targets,
            block_size=block_size_,
        )
        return ops.cross_entropy_forward(logits_tgt=logits_tgt, logits_lse=logits_lse, targets=targets, use_quack=config.use_quack)

    fn_dict = {
        "rapier": lambda: _gens_gemm_cross_entropy(
            A=inputs["A"],
            B=inputs["B"],
            targets=auxiliary["targets"],
            block_size_=block_size,
        ),
        "ref": lambda: _refs_gemm_cross_entropy(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            targets=auxiliary["targets"],
            block_size_=block_size,
        ),
        "quack": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=True,
        ),
        "cublas": lambda: gemm(
            A=inputs["A"],
            B=inputs["B"],
            quack=False,
        ),
        "torch": lambda: ops2.gemm_cross_entropy(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            targets=auxiliary["targets"],
            backend="torch",
            use_compile=True,
        ),
        "liger": lambda: ops2.gemm_cross_entropy(
            A=inputs_ref["A"],
            B=B_transposed,
            targets=auxiliary["targets"],
            backend="liger",
            use_compile=False,
        ),
        "liger-compile": lambda: ops2.gemm_cross_entropy(
            A=inputs_ref["A"],
            B=B_transposed,
            targets=auxiliary["targets"],
            backend="liger",
            use_compile=True,
        ),
        "liger2": lambda: ops2.gemm_cross_entropy(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            targets=auxiliary["targets"],
            backend="liger2",
            use_compile=False,
        ),
        "liger2-compile": lambda: ops2.gemm_cross_entropy(
            A=inputs_ref["A"],
            B=inputs_ref["B"],
            targets=auxiliary["targets"],
            backend="liger2",
            use_compile=True,
        ),
    }
    return bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_gpt(
    M: int,
    N: int,
    K: int,
    block_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    N_rope = N * 3
    results = {}

    time.sleep(0.5)
    results["gemm_residual_partial_rmsnorm"] = benchmark_gemm_residual_partial_rmsnorm(
        M=M,
        N=N,
        K=K,
        block_size=block_size,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    time.sleep(0.5)
    results["gemm_rmsnorm"] = benchmark_gemm_rmsnorm(
        M=M,
        N=N,
        K=K,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    time.sleep(0.5)
    results["gemm_rmsnorm_swiglu"] = benchmark_gemm_rmsnorm_swiglu(
        M=M,
        N=N,
        K=K,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    time.sleep(0.5)
    results["gemm_rmsnorm_rope"] = benchmark_gemm_rmsnorm_rope(
        M=M,
        N=N_rope,
        K=K,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    time.sleep(0.5)
    results["gemm_rmsnorm_partial_cross_entropy"] = benchmark_gemm_rmsnorm_partial_cross_entropy(
        M=M,
        K=K,
        block_size=block_size,
        vocab_size=vocab_size,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    time.sleep(0.5)
    results["gemm_residual_partial_rmsnorm_bwd"] = benchmark_gemm_residual_partial_rmsnorm_bwd(
        M=M,
        N=N,
        K=K,
        block_size=block_size,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    time.sleep(0.5)
    results["gemm_partial_swiglu_bwd"] = benchmark_gemm_partial_swiglu_bwd(
        M=M,
        N=N,
        K=K,
        block_size=block_size,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    return results


def benchmark_gpt2(
    M: int,
    N: int,
    K: int,
    block_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    N_rope = N * 3
    results = {}

    time.sleep(0.5)
    results["gemm_swiglu"] = benchmark_gemm_swiglu(
        M=M,
        N=N,
        K=K,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    time.sleep(0.5)
    results["gemm_rope"] = benchmark_gemm_rope(
        M=M,
        N=N_rope,
        K=K,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    time.sleep(0.5)
    results["gemm_cross_entropy"] = benchmark_gemm_cross_entropy(
        M=M,
        K=K,
        block_size=block_size,
        vocab_size=vocab_size,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )

    return results


def benchmark_trainstation(
    M: int,
    N: int,
    K: int,
    block_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    N_rope = N * 3
    results = {}

    time.sleep(0.5)
    inputs_0, _, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )
    time.sleep(0.5)
    results["gemm_residual_partial_rmsnorm"] = bench_utils.do_bench_dict(
        fn_dict={
            "human": lambda: trainstation_utils.gemm_rms(
                A=inputs_0["A"],
                B=inputs_0["B"],
                C=inputs_0["C"],
                norm_weight=inputs_0["W"],
                skip_final_reduction=True,
                tuned=True,
                tile_n=block_size,
            ),
        },
        warmup=warmup,
        repeats=repeats,
    )
    time.sleep(0.5)

    time.sleep(0.5)
    inputs_1, _, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )
    time.sleep(0.5)
    results["gemm_rmsnorm"] = bench_utils.do_bench_dict(
        fn_dict={
            "human": lambda: trainstation_utils.gemm_rstd_norm_fwd(
                A=inputs_1["A"],
                B=inputs_1["B"],
                colvec_rstd=inputs_1["R"],
                tuned=True,
            ),
        },
        warmup=warmup,
        repeats=repeats,
    )
    time.sleep(0.5)

    time.sleep(0.5)
    inputs_2, _, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )
    time.sleep(0.5)
    results["gemm_rmsnorm_swiglu"] = bench_utils.do_bench_dict(
        fn_dict={
            "human": lambda: trainstation_utils.gemm_rstd_norm_fwd(
                A=inputs_2["A"],
                B=inputs_2["B"],
                colvec_rstd=inputs_2["R"],
                use_gated_act=True,
                activation="swiglu",
                tuned=True,
            ),
        },
        warmup=warmup,
        repeats=repeats,
    )
    time.sleep(0.5)

    time.sleep(0.5)
    inputs_3, _, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N_rope,
        K=K,
        vocab_size=None,
        dtype=dtype,
        rope=True,
    )
    time.sleep(0.5)
    results["gemm_rmsnorm_rope"] = bench_utils.do_bench_dict(
        fn_dict={
            "human": lambda: trainstation_utils.gemm_rstd_norm_fwd(
                A=inputs_3["A"],
                B=inputs_3["B"],
                colvec_rstd=inputs_3["R"],
                use_rope=True,
                cos_sin=inputs_3["cos_sin"],
                tuned=True,
            ),
        },
        warmup=warmup,
        repeats=repeats,
    )
    time.sleep(0.5)

    time.sleep(0.5)
    inputs_4, _, _, auxiliary_4 = tests.create_gpt_inputs(
        M=M,
        N=vocab_size,
        K=K,
        vocab_size=vocab_size,
        dtype=dtype,
    )
    time.sleep(0.5)
    results["gemm_rmsnorm_partial_cross_entropy"] = bench_utils.do_bench_dict(
        fn_dict={
            "human": lambda: trainstation_utils.gemm_rstd_norm_fwd(
                A=inputs_4["A"],
                B=inputs_4["B"],
                colvec_rstd=inputs_4["R"],
                use_lse=True,
                target_idx=auxiliary_4["targets"],
                tuned=True,
                tile_n=block_size,
            ),
        },
        warmup=warmup,
        repeats=repeats,
    )
    time.sleep(0.5)

    time.sleep(0.5)
    inputs_5, _, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        backward=True,
        transpose_B=True,
    )
    # `B` is transposed by default for bwd, so it is (N, K), transpose to (K, N)
    BT5 = inputs_5["B"].mT.contiguous()
    time.sleep(0.5)
    results["gemm_residual_partial_rmsnorm_bwd"] = bench_utils.do_bench_dict(
        fn_dict={
            "human": lambda: trainstation_utils.gemm_rms_bwd(
                dPreAct=inputs_5["A"],
                W=BT5,
                prenorm=inputs_5["C"],
                rstd=inputs_5["R"],
                preact_dpreact=inputs_5["ZdZ"],
                dprenorm_in=inputs_5["O"],
                norm_weight=inputs_5["W"],
                skip_final_reduction=True,
                tuned=True,
                tile_m=block_size,
            ),
        },
        warmup=warmup,
        repeats=repeats,
    )
    time.sleep(0.5)

    time.sleep(0.5)
    inputs_6, _, _, _ = tests.create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        backward=True,
        transpose_B=True,
    )
    # `B` is transposed by default for bwd, so it is (N, K), transpose to (K, N)
    BT6 = inputs_6["B"].mT.contiguous()
    time.sleep(0.5)
    results["gemm_partial_swiglu_bwd"] = bench_utils.do_bench_dict(
        fn_dict={
            "human": lambda: trainstation_utils.gemm_dgated_zdz(
                A=inputs_6["A"],
                B=BT6,
                PreAct=inputs_6["Z"],
                activation="swiglu",
                colvec_reduce=True,
                skip_final_reduction=True,
                tuned=True,
                tile_n=block_size,
            ),
        },
        warmup=warmup,
        repeats=repeats,
    )
    time.sleep(0.5)
    return results


def benchmark_gpt_shapes(
    num: int,
    block_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> list[dict[int, dict]]:
    all_results = []
    for i in range(num + 1):

        results_dict = {}
        for mnk in [2048, 4096, 8192]:
            results_dict[mnk] = {
                "gpt": benchmark_gpt(
                    M=mnk,
                    N=mnk,
                    K=mnk,
                    block_size=block_size,
                    vocab_size=vocab_size,
                    dtype=dtype,
                    warmup=warmup,
                    repeats=repeats,
                ),
                "gpt2": benchmark_gpt2(
                    M=mnk,
                    N=mnk,
                    K=mnk,
                    block_size=block_size,
                    vocab_size=vocab_size,
                    dtype=dtype,
                    warmup=warmup,
                    repeats=repeats,
                ),
                "trainstation": benchmark_trainstation(
                    M=mnk,
                    N=mnk,
                    K=mnk,
                    block_size=block_size,
                    vocab_size=vocab_size,
                    dtype=dtype,
                    warmup=warmup,
                    repeats=repeats,
                ),
            }

        if i == 0:
            time.sleep(60)
        else:
            all_results.append(results_dict)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    DEFAULT_BLOCK_SIZE = 128
    DEFAULT_VOCAB_SIZE = 32768
    DEFAULT_DTYPE = torch.bfloat16
    DEFAULT_WARMUP = 5
    DEFAULT_REPEATS = 30

    results = benchmark_gpt_shapes(
        num=args.num,
        block_size=DEFAULT_BLOCK_SIZE,
        vocab_size=DEFAULT_VOCAB_SIZE,
        dtype=DEFAULT_DTYPE,
        warmup=DEFAULT_WARMUP,
        repeats=DEFAULT_REPEATS,
    )
    torch.save(results, args.output)
