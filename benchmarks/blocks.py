import torch
from typing import Callable
from coda.core.gemm import functional as gemm_functional
from coda.kernels.functional import cross_entropy as cross_entropy_functional
from coda.kernels.blocks import llama3
from coda.kernels.blocks import transformer

from benchmarks.backends import torch_blocks
from benchmarks.backends import torch_ops
from benchmarks.backends.torch_ops import PairLayout
from benchmarks.utils import data
from benchmarks.utils import operands
from benchmarks.utils.cases import (
    Case,
    Implementation,
    TrainingImplementation,
    make_batch_mean_loss_fn,
    make_oracle_fn,
    make_step_fn,
)
from benchmarks.utils.data import Sampler, Shape


def coda_gemm_residual_rmsnorm_gemm(
    x: torch.Tensor,
    y: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    wn: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # the residual x + y @ w0^T, its RMSNorm without the row scale, and
    # each row's rstd, which gemm_rmsnorm's epilogue applies
    residual, normed_without_rstd, rstd = gemm_functional.gemm_residual_partial_rmsnorm(
        A=y,
        B=w0.mT,
        C=x,
        weight=wn,
        eps=eps,
    )
    out = gemm_functional.gemm_rmsnorm(
        A=normed_without_rstd,
        B=w1.mT,
        rstd=rstd,
    )
    return residual, out


def coda_gemm_residual_rmsnorm_gemm_swiglu(
    x: torch.Tensor,
    y: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    wn: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual, normed_without_rstd, rstd = gemm_functional.gemm_residual_partial_rmsnorm(
        A=y,
        B=w0.mT,
        C=x,
        weight=wn,
        eps=eps,
    )
    preact, postact = gemm_functional.gemm_rmsnorm_swiglu(
        A=normed_without_rstd,
        B=w1.mT,
        rstd=rstd,
    )
    return residual, preact, postact


def coda_gemm_residual_rmsnorm_gemm_rope(
    x: torch.Tensor,
    y: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    wn: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    size_qk: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual, normed_without_rstd, rstd = gemm_functional.gemm_residual_partial_rmsnorm(
        A=y,
        B=w0.mT,
        C=x,
        weight=wn,
        eps=eps,
    )
    out = gemm_functional.gemm_rmsnorm_rope(
        A=normed_without_rstd,
        B=w1.mT,
        rstd=rstd,
        positions=positions,
        frequencies=frequencies,
    )
    # CODA writes one [q | k | v] on purpose and never needs to split it;
    # torch returns qk and v apart, so two column views line up with those
    qk = out[:, :size_qk]
    v = out[:, size_qk:]
    return residual, qk, v


def coda_gemm_residual_rmsnorm_gemm_qknorm_rope(
    x: torch.Tensor,
    y: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    wn: torch.Tensor,
    gamma_qk: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual, normed_without_rstd, rstd = gemm_functional.gemm_residual_partial_rmsnorm(
        A=y,
        B=w0.mT,
        C=x,
        weight=wn,
        eps=eps,
    )
    # q and k are normed per head and rotated; v is a GEMM of its own
    size_qk = head_dim * (num_heads_q + num_heads_k)
    size_v = w1.shape[0] - size_qk

    qkv = torch.empty(
        size=(residual.shape[0], w1.shape[0]),
        dtype=residual.dtype,
        device=residual.device,
    )
    qk, v = torch.split(qkv, (size_qk, size_v), dim=1)
    w1_qk, w1_v = torch.split(w1, (size_qk, size_v))
    # also written for the backward: the preact (the raw accumulator,
    # before rstd) and each head's mean square. torch's preact is its
    # GEMM's output after the norm, and its per-head statistics stay
    # inside F.rms_norm, so neither is returned
    gemm_functional.gemm_rmsnorm_qknorm_rope(
        A=normed_without_rstd,
        B=w1_qk.mT,
        rstd=rstd,
        weight=gamma_qk,
        positions=positions,
        frequencies=frequencies,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        out=qk,
    )
    gemm_functional.gemm_rmsnorm(
        A=normed_without_rstd,
        B=w1_v.mT,
        rstd=rstd,
        out=v,
    )
    return residual, qk, v


def coda_gemm_residual_rmsnorm_gemm_cross_entropy_fwd_bwd(
    x: torch.Tensor,
    y: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    wn: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual, normed_without_rstd, rstd = gemm_functional.gemm_residual_partial_rmsnorm(
        A=y,
        B=w0.mT,
        C=x,
        weight=wn,
        eps=eps,
    )
    # zdz, the row dot the final norm's backward takes, is formed as
    # CODA's training step forms it, and paid for here, but not
    # returned: torch's path never forms it
    losses, dlogits, _ = cross_entropy_functional.gemm_rmsnorm_cross_entropy_fwd_bwd(
        A=normed_without_rstd,
        B=w1.mT,
        rstd=rstd,
        target=target,
        ignore_index=ignore_index,
        return_zdz=True,
    )
    return residual, losses, dlogits


def gemm_residual_rmsnorm_gemm(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    # attention's output projection onto the residual stream, the MLP's
    # norm, and its gate and up projection with no epilogue
    sampler = Sampler(seed=seed)
    samples = operands.gemm_residual_rmsnorm_gemm_operands(
        shape=shape,
        in_features=shape.dim,
        out_features=2 * shape.ffn,
        sampler=sampler,
        dtype=dtype,
    )
    inputs = {
        **samples,
        "eps": data.NORM_EPS,
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_blocks.gemm_residual_rmsnorm_gemm,
            in_dims={
                "x": 0,
                "y": 0,
                "w0": None,
                "w1": None,
                "wn": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=coda_gemm_residual_rmsnorm_gemm,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_blocks.gemm_residual_rmsnorm_gemm,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("residual", "out"),
    )


def gemm_residual_rmsnorm_gemm_swiglu(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    # attention's output projection onto the residual stream, the MLP's
    # norm, and its gate and up projection with swiglu
    sampler = Sampler(seed=seed)
    samples = operands.gemm_residual_rmsnorm_gemm_operands(
        shape=shape,
        in_features=shape.dim,
        out_features=2 * shape.ffn,
        sampler=sampler,
        dtype=dtype,
    )
    inputs = {
        **samples,
        "eps": data.NORM_EPS,
    }
    # torch's split layout takes w1's rows as all gates, then all ups
    # the split helpers work on (K, N) weights, the layout GEMMs use; w1 is
    # (N, K), so transpose it before and after the call
    split_w1 = operands.split_swiglu_weight(
        B=samples["w1"].mT,
    ).mT
    interleaved = {
        **inputs,
        "layout": PairLayout.interleaved,
    }
    split = inputs.copy()
    split["w1"] = split_w1
    split["layout"] = PairLayout.split
    in_dims = {
        "x": 0,
        "y": 0,
        "w0": None,
        "w1": None,
        "wn": None,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_blocks.gemm_residual_rmsnorm_gemm_swiglu,
            in_dims=in_dims,
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_blocks.gemm_residual_rmsnorm_gemm_swiglu,
            in_dims=in_dims,
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=coda_gemm_residual_rmsnorm_gemm_swiglu,
                inputs=inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_blocks.gemm_residual_rmsnorm_gemm_swiglu,
                inputs=interleaved,
            ),
            "torch/split": Implementation(
                fn=torch_blocks.gemm_residual_rmsnorm_gemm_swiglu,
                inputs=split,
            ),
        },
        oracles={
            "coda": interleaved_oracle_impl,
            "torch/interleaved": interleaved_oracle_impl,
            "torch/split": split_oracle_impl,
        },
        outputs=("residual", "preact", "postact"),
    )


def gemm_residual_rmsnorm_gemm_rope(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    # the MLP's down projection onto the residual stream, the next
    # attention's norm, and its [q | k | v] projection with rope
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    size_qk = samples_rope["size_q"] + samples_rope["size_k"]
    width = size_qk + samples_rope["size_v"]
    samples = operands.gemm_residual_rmsnorm_gemm_operands(
        shape=shape,
        in_features=shape.ffn,
        out_features=width,
        sampler=sampler,
        dtype=dtype,
    )
    inputs = {
        **samples,
        "eps": data.NORM_EPS,
    }
    coda_inputs = {
        **inputs,
        "positions": samples_rope["positions"],
        "frequencies": samples_rope["frequencies_qkv"],
        "size_qk": size_qk,
    }
    interleaved = {
        **inputs,
        "cos": None,
        "sin": None,
        "positions": samples_rope["positions"],
        "frequencies": samples_rope["frequencies_qkv"],
        "size_q": samples_rope["size_q"],
        "size_k": samples_rope["size_k"],
        "size_v": samples_rope["size_v"],
        "head_dim": shape.head_dim,
        "layout": PairLayout.interleaved,
    }
    # the split helpers work on (K, N) weights, the layout GEMMs use; w1 is
    # (N, K), so transpose it before and after the call
    split_w1 = operands.split_rope_weight(
        B=samples["w1"].mT,
        size_qk=size_qk,
        head_dim=shape.head_dim,
    ).mT
    split = interleaved.copy()
    split["w1"] = split_w1
    split["layout"] = PairLayout.split
    interleaved_cached = operands.cache_rope_cos_sin(
        uncached=interleaved,
        position_name="positions",
        frequency_name="frequencies",
    )
    split_cached = operands.cache_rope_cos_sin(
        uncached=split,
        position_name="positions",
        frequency_name="frequencies",
    )
    in_dims = {
        "x": 0,
        "y": 0,
        "w0": None,
        "w1": None,
        "wn": None,
        "positions": 0,
        "frequencies": None,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_blocks.gemm_residual_rmsnorm_gemm_rope,
            in_dims=in_dims,
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_blocks.gemm_residual_rmsnorm_gemm_rope,
            in_dims=in_dims,
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=coda_gemm_residual_rmsnorm_gemm_rope,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_blocks.gemm_residual_rmsnorm_gemm_rope,
                inputs=interleaved,
            ),
            "torch/interleaved/cached": Implementation(
                fn=torch_blocks.gemm_residual_rmsnorm_gemm_rope,
                inputs=interleaved_cached,
            ),
            "torch/split": Implementation(
                fn=torch_blocks.gemm_residual_rmsnorm_gemm_rope,
                inputs=split,
            ),
            "torch/split/cached": Implementation(
                fn=torch_blocks.gemm_residual_rmsnorm_gemm_rope,
                inputs=split_cached,
            ),
        },
        oracles={
            "coda": interleaved_oracle_impl,
            "torch/interleaved": interleaved_oracle_impl,
            "torch/interleaved/cached": interleaved_oracle_impl,
            "torch/split": split_oracle_impl,
            "torch/split/cached": split_oracle_impl,
        },
        outputs=("residual", "qk", "v"),
    )


def gemm_residual_rmsnorm_gemm_qknorm_rope(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    # the rope chain of the transformer block, MHA: q and k normed per
    # head by gamma_qk before rope; torch runs the block's own path
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    size_qk = samples_rope["size_q"] + samples_rope["size_k"]
    width = size_qk + samples_rope["size_v"]
    samples = operands.gemm_residual_rmsnorm_gemm_operands(
        shape=shape,
        in_features=shape.ffn,
        out_features=width,
        sampler=sampler,
        dtype=dtype,
    )
    # qk-norm's weight: one head's worth for q, then one for k
    gamma_qk = sampler.gamma(
        size=2 * shape.head_dim,
        dtype=dtype,
    )
    tensors = {
        **samples,
        "gamma_qk": gamma_qk,
    }
    coda_inputs = {
        **tensors,
        "positions": samples_rope["positions"],
        "frequencies": samples_rope["frequencies_qk"],
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
    }
    # torch takes fp32 angles; the oracle takes exact fp64 ones, as in
    # transformer.block
    pair_frequencies = operands.inv_freq(shape=shape)
    angles = torch_ops.rope_angles(
        positions=samples_rope["positions"],
        inv_freq=pair_frequencies,
    )
    angles_fp32 = angles.to(dtype=torch.float32)
    torch_inputs = {
        **tensors,
        "angles": angles_fp32,
        "head_dim": shape.head_dim,
        "eps": data.NORM_EPS,
    }
    oracle_inputs = torch_inputs.copy()
    oracle_inputs["angles"] = angles
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_blocks.gemm_residual_rmsnorm_gemm_qknorm_rope,
            in_dims={
                "x": 0,
                "y": 0,
                "w0": None,
                "w1": None,
                "wn": None,
                "gamma_qk": None,
                "angles": 0,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=coda_gemm_residual_rmsnorm_gemm_qknorm_rope,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_blocks.gemm_residual_rmsnorm_gemm_qknorm_rope,
                inputs=torch_inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch/interleaved": oracle_impl,
        },
        outputs=("residual", "qk", "v"),
    )


def gemm_residual_rmsnorm_gemm_cross_entropy_fwd_bwd(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    # the last MLP's down projection onto the residual stream, the final
    # norm, and the LM head w1 with the loss and its gradient
    sampler = Sampler(seed=seed)
    samples = operands.gemm_residual_rmsnorm_gemm_operands(
        shape=shape,
        in_features=shape.ffn,
        out_features=shape.vocab_size,
        sampler=sampler,
        dtype=dtype,
    )
    inputs = {
        **samples,
        "target": sampler.targets(
            size=shape.tokens,
            vocab_size=shape.vocab_size,
        ),
        "ignore_index": data.IGNORE_INDEX,
        "eps": data.NORM_EPS,
    }
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_blocks.gemm_residual_rmsnorm_gemm_cross_entropy_fwd_bwd,
            in_dims={
                "x": 0,
                "y": 0,
                "w0": None,
                "w1": None,
                "wn": None,
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
                fn=coda_gemm_residual_rmsnorm_gemm_cross_entropy_fwd_bwd,
                inputs=inputs,
            ),
            "torch": Implementation(
                fn=torch_blocks.gemm_residual_rmsnorm_gemm_cross_entropy_fwd_bwd_bench,
                inputs=inputs,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch": oracle_impl,
        },
        outputs=("residual", "losses", "dlogits"),
    )


def llama3_block_pre(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples = operands.block_operands(
        shape=shape,
        name="llama3.block_pre",
        sampler=sampler,
        dtype=dtype,
    )
    leaves = ("x", "w", "wn")
    cotangents = ("grad_residual", "grad_qkv")
    # the forward outputs, then one grad per leaf
    forward_outputs = ("residual", "qkv")
    grads = ("dx", "dw", "dwn")
    outputs = (*forward_outputs, *grads)
    # every implementation takes the same leaves and cotangents
    shared_inputs = {}
    for name in (*leaves, *cotangents):
        shared_inputs[name] = samples[name]

    coda_inputs = {
        **shared_inputs,
        "positions": samples["positions"],
        "frequencies": samples["frequencies"],
        "eps": data.NORM_EPS,
    }
    # torch takes fp32 angles; the oracle takes exact fp64 ones
    pair_frequencies = operands.inv_freq(shape=shape)
    angles = torch_ops.rope_angles(
        positions=samples["positions"],
        inv_freq=pair_frequencies,
    )
    angles_fp32 = angles.to(dtype=torch.float32)
    torch_inputs = {
        **shared_inputs,
        "angles": angles_fp32,
        "head_dim": shape.head_dim,
        "eps": data.NORM_EPS,
    }
    oracle_inputs = {
        **shared_inputs,
        "angles": angles,
        "head_dim": shape.head_dim,
        "eps": data.NORM_EPS,
    }

    coda_forward = make_unbatched_fn(
        op=llama3.block_pre,
        activations=("x",),
    )
    oracle_step = make_step_fn(
        forward=torch_blocks.block_pre,
        leaves=leaves,
        cotangents=cotangents,
        copies_cotangents=False,
    )
    oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=oracle_step,
            in_dims={
                "x": 0,
                "w": None,
                "wn": None,
                "grad_residual": 0,
                "grad_qkv": 0,
                "angles": 0,
            },
            out_dims=(0, 0, 0, 0, 0),
            reduce_dims=(None, None, None, 0, 0),
        ),
        inputs=oracle_inputs,
    )
    return Case(
        implementations={
            # CODA's backward overwrites its cotangents, so its step copies them
            "coda": TrainingImplementation(
                forward=coda_forward,
                inputs=coda_inputs,
                leaves=leaves,
                cotangents=cotangents,
                copies_cotangents=True,
            ),
            "torch/interleaved": TrainingImplementation(
                forward=torch_blocks.block_pre_bench,
                inputs=torch_inputs,
                leaves=leaves,
                cotangents=cotangents,
                copies_cotangents=False,
            ),
        },
        oracles={
            "coda": oracle_impl,
            "torch/interleaved": oracle_impl,
        },
        outputs=outputs,
    )


OPS = {
    "transformer.block_pre": transformer_block_pre,
    "transformer.block": transformer_block,
    "transformer.block_post": transformer_block_post,
    "llama3.block_pre": llama3_block_pre,
    "llama3.block": llama3_block,
    "llama3.block_post": llama3_block_post,
    "chain.gemm_residual_rmsnorm_gemm": gemm_residual_rmsnorm_gemm,
    "chain.gemm_residual_rmsnorm_gemm_swiglu": gemm_residual_rmsnorm_gemm_swiglu,
    "chain.gemm_residual_rmsnorm_gemm_rope": gemm_residual_rmsnorm_gemm_rope,
    "chain.gemm_residual_rmsnorm_gemm_qknorm_rope": gemm_residual_rmsnorm_gemm_qknorm_rope,
    "chain.gemm_residual_rmsnorm_gemm_cross_entropy_fwd_bwd": gemm_residual_rmsnorm_gemm_cross_entropy_fwd_bwd,
}
