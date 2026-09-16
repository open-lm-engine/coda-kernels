# Copyright (c) 2025, Tri Dao
# v0.4.1
from typing import Optional, Tuple
from functools import partial

import torch
from torch import Tensor

from quack.gemm_interface import (
    GemmConfig,
    AutotuneConfig,
    autotune,
    default_config,
    get_all_configs,
    get_device_capacity,
    _prune_gemm_rms_configs,
    gemm_sq_reduce_dispatch,
)


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["dynamic_scheduler"],
    prune_configs_by={"early_config_prune": _prune_gemm_rms_configs},
)
def _gemm_rms_tuned(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, N) or (L, K, N)
    out: Tensor,  # (M, N) or (L, M, N)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    norm_weight: Optional[Tensor] = None,  # (N,) or (L, N)
    premult_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) — pre-norm_weight snapshot
    eps: float = 1e-6,
    block_size: int | None = None,
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> Tensor:
    if config is None:
        config = default_config(A.device)
    og_ndim_2 = A.ndim == 2
    N = B.shape[-1]
    if A.ndim == 2:
        A = A.unsqueeze(0)
    B = B.mT
    if B.ndim == 2:
        B = B.unsqueeze(0)
    if out.ndim == 2:
        out = out.unsqueeze(0)
    if C is not None and C.ndim == 2:
        C = C.unsqueeze(0)
    if norm_weight is not None and norm_weight.ndim == 1:
        norm_weight = norm_weight.unsqueeze(0)  # (L, N)
    if premult_out is not None and premult_out.ndim == 2:
        premult_out = premult_out.unsqueeze(0)
    # Allocate partial reduction buffer
    tile_n = config.tile_n
    assert block_size is not None
    assert tile_n == block_size
    n_tiles = (N + tile_n - 1) // tile_n
    colvec_reduce = torch.empty(
        (A.shape[0], A.shape[1], n_tiles), dtype=torch.float32, device=A.device
    )
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    gemm_sq_reduce_dispatch(
        A,
        B,
        out,
        C,
        colvec_reduce,
        tile_count_semaphore,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        rowvec=norm_weight,
        aux_out=premult_out,
    )
    return colvec_reduce


def _gemm_rms_out(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    C: Optional[Tensor] = None,
    norm_weight: Optional[Tensor] = None,
    premult_out: Optional[Tensor] = None,
    eps: float = 1e-6,
    block_size: int | None = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> Tensor:
    """GEMM + RMS + optional rowvec scaling.

    D_raw = A @ B (+ C), rstd = rsqrt(mean(D_raw^2) + eps), D_out = D_raw * norm_weight.
    If premult_out is provided, D_raw (the pre-norm_weight value) is also written to it.
    """
    fn = _gemm_rms_tuned if tuned else partial(_gemm_rms_tuned.fn, config=None)
    return fn(
        A,
        B,
        out,
        C=C,
        norm_weight=norm_weight,
        premult_out=premult_out,
        eps=eps,
        block_size=block_size,
        dynamic_scheduler=dynamic_scheduler,
    )


def gemm_rms(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, N) or (L, K, N)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    norm_weight: Optional[Tensor] = None,  # (N,) or (L, N)
    out: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    out_dtype: Optional[torch.dtype] = None,
    premult_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) — pre-norm_weight snapshot
    eps: float = 1e-6,
    block_size: int | None = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> Tuple[Tensor, Tensor]:
    """GEMM + RMS statistics + optional rowvec scaling.

    D_raw = A @ B (+ C), rstd = rsqrt(mean(D_raw^2) + eps), D_out = D_raw * norm_weight.
    If premult_out is provided, D_raw (the pre-norm_weight value) is also written to it.
    Returns (D_out, rstd).
    """
    out_dtype = A.dtype if out_dtype is None else out_dtype
    N = B.shape[-1]
    if out is None:
        out_shape = (*A.shape[:-1], N)
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    # Empty-input fast path. Skipping the kernel also avoids a torch.library
    # adinplaceorview_impl IndexError that fires on empty inputs because
    # premult_out's positional slot isn't materialized in the boxed args tuple.
    # K=0 with no C reduces the matmul to zero, so D = 0 and rstd = rsqrt(eps).
    if out.numel() == 0 or A.numel() == 0:
        raise ValueError
    rstd = _gemm_rms_out(
        A,
        B,
        out,
        C=C,
        norm_weight=norm_weight,
        premult_out=premult_out,
        eps=eps,
        block_size=block_size,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
    )
    return out, rstd
