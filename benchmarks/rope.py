import torch
from coda.core.elementwise import functional as elementwise_functional
from coda.core.gemm import functional as gemm_functional

from benchmarks.backends import torch_ops
from benchmarks.backends.torch_ops import PairLayout
from benchmarks.utils import data
from benchmarks.utils import operands
from benchmarks.utils.cases import Case, Implementation, make_oracle_fn
from benchmarks.utils.data import Sampler, Shape


def coda_gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    size_qk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # CODA returns q, k and v in one tensor, while torch returns [q | k]
    # and v separately; two views of CODA's tensor give the same outputs
    # without copying
    out = gemm_functional.gemm_rope(
        A=A,
        B=B,
        positions=positions,
        frequencies=frequencies,
    )
    qk = out[:, :size_qk]
    v = out[:, size_qk:]
    return qk, v


def coda_gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    rstd: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    size_qk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # the same split of CODA's output, for the version with rmsnorm
    out = gemm_functional.gemm_rmsnorm_rope(
        A=A,
        B=B,
        rstd=rstd,
        positions=positions,
        frequencies=frequencies,
    )
    qk = out[:, :size_qk]
    v = out[:, size_qk:]
    return qk, v


def coda_rope_bwd_zdz(
    y: torch.Tensor,
    dy: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    scale: float,
    size_qk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # the same split, for rope's backward: CODA returns the gradient of
    # [q | k | v] in one tensor
    dz, zdz = elementwise_functional.rope_bwd_zdz(
        y=y,
        dy=dy,
        pos=pos,
        freq=freq,
        scale=scale,
    )
    dz_qk = dz[:, :size_qk]
    dz_v = dz[:, size_qk:]
    return dz_qk, dz_v, zdz


def coda_qknorm_rope_bwd_zdz(
    dy: torch.Tensor,
    preact: torch.Tensor,
    v: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    scale: float,
    dv: torch.Tensor,
    size_qk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # the same split, for the backward of qk-norm followed by rope
    dz, zdz, dgamma = elementwise_functional.qknorm_rope_bwd_zdz(
        dy=dy,
        preact=preact,
        v=v,
        rstd=rstd,
        gamma=gamma,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        scale=scale,
        dv=dv,
    )
    dz_qk = dz[:, :size_qk]
    dz_v = dz[:, size_qk:]
    return dz_qk, dz_v, zdz, dgamma


def gemm_rope(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    size_qk = samples_rope["size_q"] + samples_rope["size_k"]
    width = size_qk + samples_rope["size_v"]
    samples_gemm = operands.gemm_operands(
        shape=shape,
        out_features=width,
        sampler=sampler,
        dtype=dtype,
    )
    A = samples_gemm["A"]
    B = samples_gemm["B"]
    positions = samples_rope["positions"]
    # CODA gets frequencies for q, k and v (zero on v); torch for q and k only
    frequencies_qk = samples_rope["frequencies_qk"]
    frequencies_qkv = samples_rope["frequencies_qkv"]
    interleaved = {
        "A": A,
        "B": B,
        "cos": None,
        "sin": None,
        "positions": positions,
        "frequencies": frequencies_qk,
        "size_q": samples_rope["size_q"],
        "size_k": samples_rope["size_k"],
        "size_v": samples_rope["size_v"],
        "head_dim": shape.head_dim,
        "layout": PairLayout.interleaved,
    }
    split = interleaved.copy()
    split["B"] = operands.split_layout_rope_weight(
        B=B,
        size_qk=size_qk,
        head_dim=shape.head_dim,
    )
    split["layout"] = PairLayout.split
    # torch with its rope tables built once, outside the timed step
    cos, sin = operands.cache_rope_cos_sin(
        positions=positions,
        frequencies=frequencies_qk,
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
    coda_inputs = {
        "A": A,
        "B": B,
        "positions": positions,
        "frequencies": frequencies_qkv,
        "size_qk": size_qk,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rope,
            in_dims={
                "A": 0,
                "B": None,
                "positions": 0,
                "frequencies": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rope,
            in_dims={
                "A": 0,
                "B": None,
                "positions": 0,
                "frequencies": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=coda_gemm_rope,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.gemm_rope,
                inputs=interleaved,
            ),
            "torch/interleaved/cached": Implementation(
                fn=torch_ops.gemm_rope,
                inputs=interleaved_cached,
            ),
            "torch/split": Implementation(
                fn=torch_ops.gemm_rope,
                inputs=split,
            ),
            "torch/split/cached": Implementation(
                fn=torch_ops.gemm_rope,
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
        outputs=("qk", "v"),
    )


def gemm_rmsnorm_rope(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    size_qk = samples_rope["size_q"] + samples_rope["size_k"]
    width = size_qk + samples_rope["size_v"]
    samples_gemm = operands.gemm_operands(
        shape=shape,
        out_features=width,
        sampler=sampler,
        dtype=dtype,
    )
    samples_rmsnorm = operands.rmsnorm_operands(
        shape=shape,
        x=samples_gemm["A"],
        rstd_scale=shape.statistics.rstd,
        sampler=sampler,
        dtype=dtype,
    )
    A = samples_rmsnorm["A"]
    B = samples_gemm["B"]
    rstd = samples_rmsnorm["rstd"]
    positions = samples_rope["positions"]
    # CODA gets frequencies for q, k and v (zero on v); torch for q and k only
    frequencies_qk = samples_rope["frequencies_qk"]
    frequencies_qkv = samples_rope["frequencies_qkv"]
    interleaved = {
        "A": A,
        "B": B,
        "rstd": rstd,
        "cos": None,
        "sin": None,
        "positions": positions,
        "frequencies": frequencies_qk,
        "size_q": samples_rope["size_q"],
        "size_k": samples_rope["size_k"],
        "size_v": samples_rope["size_v"],
        "head_dim": shape.head_dim,
        "layout": PairLayout.interleaved,
    }
    split = interleaved.copy()
    split["B"] = operands.split_layout_rope_weight(
        B=B,
        size_qk=size_qk,
        head_dim=shape.head_dim,
    )
    split["layout"] = PairLayout.split
    # torch with its rope tables built once, outside the timed step
    cos, sin = operands.cache_rope_cos_sin(
        positions=positions,
        frequencies=frequencies_qk,
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
    coda_inputs = {
        "A": A,
        "B": B,
        "rstd": rstd,
        "positions": positions,
        "frequencies": frequencies_qkv,
        "size_qk": size_qk,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rmsnorm_rope,
            in_dims={
                "A": 0,
                "B": None,
                "rstd": 0,
                "positions": 0,
                "frequencies": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rmsnorm_rope,
            in_dims={
                "A": 0,
                "B": None,
                "rstd": 0,
                "positions": 0,
                "frequencies": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=coda_gemm_rmsnorm_rope,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.gemm_rmsnorm_rope,
                inputs=interleaved,
            ),
            "torch/interleaved/cached": Implementation(
                fn=torch_ops.gemm_rmsnorm_rope,
                inputs=interleaved_cached,
            ),
            "torch/split": Implementation(
                fn=torch_ops.gemm_rmsnorm_rope,
                inputs=split,
            ),
            "torch/split/cached": Implementation(
                fn=torch_ops.gemm_rmsnorm_rope,
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
        outputs=("qk", "v"),
    )


def rope_bwd_zdz(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    size_qk = samples_rope["size_q"] + samples_rope["size_k"]
    width = size_qk + samples_rope["size_v"]
    y = sampler.randn(
        shape=(shape.tokens, width),
        dtype=dtype,
    )
    dy = sampler.randn(
        shape=(shape.tokens, width),
        dtype=dtype,
    )
    positions = samples_rope["positions"]
    # CODA gets frequencies for q, k and v (zero on v); torch for q and k only
    frequencies_qk = samples_rope["frequencies_qk"]
    frequencies_qkv = samples_rope["frequencies_qkv"]
    # zdz = sum(z * dz) / dim per row, as the RMSNorm backward uses it
    scale = 1.0 / shape.dim
    interleaved = {
        "y": y,
        "dy": dy,
        "cos": None,
        "sin": None,
        "pos": positions,
        "freq": frequencies_qk,
        "scale": scale,
        "size_q": samples_rope["size_q"],
        "size_k": samples_rope["size_k"],
        "size_v": samples_rope["size_v"],
        "head_dim": shape.head_dim,
        "layout": PairLayout.interleaved,
    }
    split = interleaved.copy()
    split["y"] = operands.split_layout_qkv(
        x=y,
        size_qk=size_qk,
        head_dim=shape.head_dim,
    )
    split["dy"] = operands.split_layout_qkv(
        x=dy,
        size_qk=size_qk,
        head_dim=shape.head_dim,
    )
    split["layout"] = PairLayout.split
    # torch with its rope tables built once, outside the timed step
    cos, sin = operands.cache_rope_cos_sin(
        positions=positions,
        frequencies=frequencies_qk,
        head_dim=shape.head_dim,
    )
    interleaved_cached = {
        **interleaved,
        "pos": None,
        "freq": None,
        "cos": cos,
        "sin": sin,
    }
    split_cached = {
        **split,
        "pos": None,
        "freq": None,
        "cos": cos,
        "sin": sin,
    }
    coda_inputs = {
        "y": y,
        "dy": dy,
        "pos": positions,
        "freq": frequencies_qkv,
        "scale": scale,
        "size_qk": size_qk,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.rope_bwd_zdz,
            in_dims={
                "y": 0,
                "dy": 0,
                "pos": 0,
                "freq": None,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.rope_bwd_zdz,
            in_dims={
                "y": 0,
                "dy": 0,
                "pos": 0,
                "freq": None,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=coda_rope_bwd_zdz,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.rope_bwd_zdz,
                inputs=interleaved,
            ),
            "torch/interleaved/cached": Implementation(
                fn=torch_ops.rope_bwd_zdz,
                inputs=interleaved_cached,
            ),
            "torch/split": Implementation(
                fn=torch_ops.rope_bwd_zdz,
                inputs=split,
            ),
            "torch/split/cached": Implementation(
                fn=torch_ops.rope_bwd_zdz,
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
        outputs=("dz_qk", "dz_v", "zdz"),
    )


def gemm_qknorm_rope(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    size_qk = samples_rope["size_q"] + samples_rope["size_k"]
    samples_gemm = operands.gemm_operands(
        shape=shape,
        out_features=size_qk,
        sampler=sampler,
        dtype=dtype,
    )
    A = samples_gemm["A"]
    B = samples_gemm["B"]
    # qk-norm's weight: one head's worth for q, then one for k
    gamma = sampler.gamma(
        size=operands.NUM_QK_NORM_WEIGHTS * shape.head_dim,
        dtype=dtype,
    )
    positions = samples_rope["positions"]
    frequencies = samples_rope["frequencies_qk"]
    interleaved = {
        "A": A,
        "B": B,
        "weight": gamma,
        "cos": None,
        "sin": None,
        "positions": positions,
        "frequencies": frequencies,
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
        "layout": PairLayout.interleaved,
    }
    split = interleaved.copy()
    split["B"] = operands.split_layout_rope_weight(
        B=B,
        size_qk=size_qk,
        head_dim=shape.head_dim,
    )
    split["weight"] = operands.split_layout_heads(
        x=gamma,
        num_heads=operands.NUM_QK_NORM_WEIGHTS,
        head_dim=shape.head_dim,
    )
    split["layout"] = PairLayout.split
    # torch with its rope tables built once, outside the timed step
    cos, sin = operands.cache_rope_cos_sin(
        positions=positions,
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
    coda_inputs = {
        "A": A,
        "B": B,
        "weight": gamma,
        "positions": positions,
        "frequencies": frequencies,
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_qknorm_rope,
            in_dims={
                "A": 0,
                "B": None,
                "weight": None,
                "positions": 0,
                "frequencies": None,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_qknorm_rope,
            in_dims={
                "A": 0,
                "B": None,
                "weight": None,
                "positions": 0,
                "frequencies": None,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_qknorm_rope,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.gemm_qknorm_rope,
                inputs=interleaved,
            ),
            "torch/interleaved/cached": Implementation(
                fn=torch_ops.gemm_qknorm_rope,
                inputs=interleaved_cached,
            ),
            "torch/split": Implementation(
                fn=torch_ops.gemm_qknorm_rope,
                inputs=split,
            ),
            "torch/split/cached": Implementation(
                fn=torch_ops.gemm_qknorm_rope,
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
        outputs=("out", "preact", "head_mean_sq"),
    )


def gemm_rmsnorm_qknorm_rope(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    size_qk = samples_rope["size_q"] + samples_rope["size_k"]
    samples_gemm = operands.gemm_operands(
        shape=shape,
        out_features=size_qk,
        sampler=sampler,
        dtype=dtype,
    )
    samples_rmsnorm = operands.rmsnorm_operands(
        shape=shape,
        x=samples_gemm["A"],
        rstd_scale=shape.statistics.rstd,
        sampler=sampler,
        dtype=dtype,
    )
    A = samples_rmsnorm["A"]
    B = samples_gemm["B"]
    rstd = samples_rmsnorm["rstd"]
    # qk-norm's weight: one head's worth for q, then one for k
    gamma = sampler.gamma(
        size=operands.NUM_QK_NORM_WEIGHTS * shape.head_dim,
        dtype=dtype,
    )
    positions = samples_rope["positions"]
    frequencies = samples_rope["frequencies_qk"]
    interleaved = {
        "A": A,
        "B": B,
        "rstd": rstd,
        "weight": gamma,
        "cos": None,
        "sin": None,
        "positions": positions,
        "frequencies": frequencies,
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
        "layout": PairLayout.interleaved,
    }
    split = interleaved.copy()
    split["B"] = operands.split_layout_rope_weight(
        B=B,
        size_qk=size_qk,
        head_dim=shape.head_dim,
    )
    split["weight"] = operands.split_layout_heads(
        x=gamma,
        num_heads=operands.NUM_QK_NORM_WEIGHTS,
        head_dim=shape.head_dim,
    )
    split["layout"] = PairLayout.split
    # torch with its rope tables built once, outside the timed step
    cos, sin = operands.cache_rope_cos_sin(
        positions=positions,
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
    coda_inputs = {
        "A": A,
        "B": B,
        "rstd": rstd,
        "weight": gamma,
        "positions": positions,
        "frequencies": frequencies,
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rmsnorm_qknorm_rope,
            in_dims={
                "A": 0,
                "B": None,
                "rstd": 0,
                "weight": None,
                "positions": 0,
                "frequencies": None,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.gemm_rmsnorm_qknorm_rope,
            in_dims={
                "A": 0,
                "B": None,
                "rstd": 0,
                "weight": None,
                "positions": 0,
                "frequencies": None,
            },
            out_dims=(0, 0, 0),
            reduce_dims=(None, None, None),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=gemm_functional.gemm_rmsnorm_qknorm_rope,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.gemm_rmsnorm_qknorm_rope,
                inputs=interleaved,
            ),
            "torch/interleaved/cached": Implementation(
                fn=torch_ops.gemm_rmsnorm_qknorm_rope,
                inputs=interleaved_cached,
            ),
            "torch/split": Implementation(
                fn=torch_ops.gemm_rmsnorm_qknorm_rope,
                inputs=split,
            ),
            "torch/split/cached": Implementation(
                fn=torch_ops.gemm_rmsnorm_qknorm_rope,
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
        outputs=("out", "preact", "head_mean_sq"),
    )


def qknorm_rope_bwd(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    size_q = samples_rope["size_q"]
    size_k = samples_rope["size_k"]
    samples_qknorm = operands.qknorm_rope_bwd_operands(
        shape=shape,
        rmsnorm=False,
        sampler=sampler,
        dtype=dtype,
    )
    x = samples_qknorm["preact"]
    gamma = samples_qknorm["gamma"]
    head_mean_sq = samples_qknorm["head_mean_sq"]
    # dq and dk, the gradients attention's backward hands back, drawn as
    # one cotangent for the rotated q and k heads and then split into two
    # separate tensors, as attention returns them
    size_qk = size_q + size_k
    random = sampler.randn(
        shape=(shape.tokens, size_qk),
        dtype=dtype,
    )
    cotangent = operands.calibrate_qk_cotangent(
        shape=shape,
        random_cotangent=random,
        qk_preact=x,
        gamma=gamma,
        sampler=sampler,
        dtype=dtype,
    )
    dq = cotangent[:, :size_q].contiguous()
    dk = cotangent[:, size_q:].contiguous()
    positions = samples_rope["positions"]
    frequencies = samples_rope["frequencies_qk"]
    interleaved = {
        "dq": dq,
        "dk": dk,
        "x": x,
        "head_mean_sq": head_mean_sq,
        "gamma": gamma,
        "cos": None,
        "sin": None,
        "pos": positions,
        "freq": frequencies,
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
        "layout": PairLayout.interleaved,
    }
    split = interleaved.copy()
    split["dq"] = operands.split_layout_heads(
        x=dq,
        num_heads=shape.heads,
        head_dim=shape.head_dim,
    )
    split["dk"] = operands.split_layout_heads(
        x=dk,
        num_heads=shape.kv_heads,
        head_dim=shape.head_dim,
    )
    split["x"] = operands.split_layout_heads(
        x=x,
        num_heads=shape.heads + shape.kv_heads,
        head_dim=shape.head_dim,
    )
    split["gamma"] = operands.split_layout_heads(
        x=gamma,
        num_heads=operands.NUM_QK_NORM_WEIGHTS,
        head_dim=shape.head_dim,
    )
    split["layout"] = PairLayout.split
    # torch with its rope tables built once, outside the timed step
    cos, sin = operands.cache_rope_cos_sin(
        positions=positions,
        frequencies=frequencies,
        head_dim=shape.head_dim,
    )
    interleaved_cached = {
        **interleaved,
        "pos": None,
        "freq": None,
        "cos": cos,
        "sin": sin,
    }
    split_cached = {
        **split,
        "pos": None,
        "freq": None,
        "cos": cos,
        "sin": sin,
    }
    coda_inputs = {
        "dq": dq,
        "dk": dk,
        "x": x,
        "gamma": gamma,
        "pos": positions,
        "freq": frequencies,
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.qknorm_rope_bwd,
            in_dims={
                "dq": 0,
                "dk": 0,
                "x": 0,
                "head_mean_sq": 0,
                "gamma": None,
                "pos": 0,
                "freq": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, 0),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.qknorm_rope_bwd,
            in_dims={
                "dq": 0,
                "dk": 0,
                "x": 0,
                "head_mean_sq": 0,
                "gamma": None,
                "pos": 0,
                "freq": None,
            },
            out_dims=(0, 0),
            reduce_dims=(None, 0),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=elementwise_functional.qknorm_rope_bwd,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.qknorm_rope_bwd,
                inputs=interleaved,
            ),
            "torch/interleaved/cached": Implementation(
                fn=torch_ops.qknorm_rope_bwd,
                inputs=interleaved_cached,
            ),
            "torch/split": Implementation(
                fn=torch_ops.qknorm_rope_bwd,
                inputs=split,
            ),
            "torch/split/cached": Implementation(
                fn=torch_ops.qknorm_rope_bwd,
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
        outputs=("dx", "dgamma"),
    )


def qknorm_rope_bwd_zdz(
    shape: Shape,
    seed: int,
    dtype: torch.dtype,
) -> Case:
    sampler = Sampler(seed=seed)
    samples_rope = operands.rope_operands(shape=shape)
    samples_qknorm = operands.qknorm_rope_bwd_operands(
        shape=shape,
        rmsnorm=True,
        sampler=sampler,
        dtype=dtype,
    )
    preact = samples_qknorm["preact"]
    gamma = samples_qknorm["gamma"]
    rstd = samples_qknorm["rstd"]
    head_mean_sq = samples_qknorm["head_mean_sq"]
    size_qk = preact.shape[1]
    size_v = samples_rope["size_v"]
    samples_bwd = operands.qknorm_rope_bwd_zdz_operands(
        shape=shape,
        preact=preact,
        gamma=gamma,
        size_v=size_v,
        sampler=sampler,
        dtype=dtype,
    )
    dy = samples_bwd["dy"]
    v = samples_bwd["v"]
    dv = samples_bwd["dv"]
    positions = samples_rope["positions"]
    frequencies = samples_rope["frequencies_qk"]
    # zdz = sum(z * dz) / dim per row, as the RMSNorm backward uses it
    scale = 1.0 / shape.dim
    interleaved = {
        "dy": dy,
        "preact": preact,
        "v": v,
        "rstd": rstd,
        "head_mean_sq": head_mean_sq,
        "gamma": gamma,
        "cos": None,
        "sin": None,
        "pos": positions,
        "freq": frequencies,
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
        "scale": scale,
        "layout": PairLayout.interleaved,
        "dv": dv,
    }
    split = interleaved.copy()
    split["dy"] = operands.split_layout_qkv(
        x=dy,
        size_qk=size_qk,
        head_dim=shape.head_dim,
    )
    split["preact"] = operands.split_layout_heads(
        x=preact,
        num_heads=shape.heads + shape.kv_heads,
        head_dim=shape.head_dim,
    )
    split["gamma"] = operands.split_layout_heads(
        x=gamma,
        num_heads=operands.NUM_QK_NORM_WEIGHTS,
        head_dim=shape.head_dim,
    )
    split["layout"] = PairLayout.split
    # torch with its rope tables built once, outside the timed step
    cos, sin = operands.cache_rope_cos_sin(
        positions=positions,
        frequencies=frequencies,
        head_dim=shape.head_dim,
    )
    interleaved_cached = {
        **interleaved,
        "pos": None,
        "freq": None,
        "cos": cos,
        "sin": sin,
    }
    split_cached = {
        **split,
        "pos": None,
        "freq": None,
        "cos": cos,
        "sin": sin,
    }
    coda_inputs = {
        "dy": dy,
        "preact": preact,
        "v": v,
        "rstd": rstd,
        "gamma": gamma,
        "pos": positions,
        "freq": frequencies,
        "head_dim": shape.head_dim,
        "num_heads_q": shape.heads,
        "num_heads_k": shape.kv_heads,
        "eps": data.NORM_EPS,
        "scale": scale,
        "dv": dv,
        "size_qk": size_qk,
    }
    interleaved_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.qknorm_rope_bwd_zdz,
            in_dims={
                "dy": 0,
                "preact": 0,
                "v": 0,
                "rstd": 0,
                "head_mean_sq": 0,
                "gamma": None,
                "pos": 0,
                "freq": None,
                "dv": 0,
            },
            out_dims=(0, 0, 0, 0),
            reduce_dims=(None, None, None, 0),
        ),
        inputs=interleaved,
    )
    split_oracle_impl = Implementation(
        fn=make_oracle_fn(
            fn=torch_ops.qknorm_rope_bwd_zdz,
            in_dims={
                "dy": 0,
                "preact": 0,
                "v": 0,
                "rstd": 0,
                "head_mean_sq": 0,
                "gamma": None,
                "pos": 0,
                "freq": None,
                "dv": 0,
            },
            out_dims=(0, 0, 0, 0),
            reduce_dims=(None, None, None, 0),
        ),
        inputs=split,
    )
    return Case(
        implementations={
            "coda": Implementation(
                fn=coda_qknorm_rope_bwd_zdz,
                inputs=coda_inputs,
            ),
            "torch/interleaved": Implementation(
                fn=torch_ops.qknorm_rope_bwd_zdz,
                inputs=interleaved,
            ),
            "torch/interleaved/cached": Implementation(
                fn=torch_ops.qknorm_rope_bwd_zdz,
                inputs=interleaved_cached,
            ),
            "torch/split": Implementation(
                fn=torch_ops.qknorm_rope_bwd_zdz,
                inputs=split,
            ),
            "torch/split/cached": Implementation(
                fn=torch_ops.qknorm_rope_bwd_zdz,
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
        outputs=("dz_qk", "dz_v", "zdz", "dgamma"),
    )


OPS = {
    "gemm_rope": gemm_rope,
    "gemm_rmsnorm_rope": gemm_rmsnorm_rope,
    "rope_bwd_zdz": rope_bwd_zdz,
    "gemm_qknorm_rope": gemm_qknorm_rope,
    "gemm_rmsnorm_qknorm_rope": gemm_rmsnorm_qknorm_rope,
    "qknorm_rope_bwd": qknorm_rope_bwd,
    "qknorm_rope_bwd_zdz": qknorm_rope_bwd_zdz,
}
