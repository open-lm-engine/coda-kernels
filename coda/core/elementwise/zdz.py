import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from typing import Callable

from quack.cache import jit_cache
from quack.cute_dsl_utils import torch2cute_dtype_map

from coda.core.ops import math_utils
from coda.core.ops import misc_utils
from coda.core.ops import constants
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils
from coda.core.ops import reduction_utils


@cute.kernel
def rope_bwd_zdz_kernel(
    mY_packed: cute.Tensor,
    mDY_packed: cute.Tensor,
    mDZ_packed: cute.Tensor,
    mZdZ: cute.Tensor,
    mPos: cute.Tensor,
    mFreq: cute.Tensor,
    scale: cutlass.Constexpr[float],
    dtype: type[cute.Numeric],
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    thr_m: cutlass.Constexpr[int],
    thr_n: cutlass.Constexpr[int],
    val_m: cutlass.Constexpr[int],
    vector_size: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    idY = cute.make_identity_tensor(mY_packed.shape)


@cute.jit
def _rope_bwd_zdz(
    mY: cute.Tensor,
    mDY: cute.Tensor,
    mDZ: cute.Tensor,
    mZdZ: cute.Tensor,
    mPos: cute.Tensor,
    mFreq: cute.Tensor,
    scale: cutlass.Constexpr[float],
    thr_m: cutlass.Constexpr[int],
    thr_n: cutlass.Constexpr[int],
    val_m: cutlass.Constexpr[int],
    stream: cuda.CUstream,
) -> int:
    mY_packed = layout_utils.recast_tensor(mY, dtype=cute.Int32)
    mDY_packed = layout_utils.recast_tensor(mDY, dtype=cute.Int32)
    mDZ_packed = layout_utils.recast_tensor(mDZ, dtype=cute.Int32)
    vector_size = cutlass.const_expr(constants.NUM_BITS_PER_COPY // mY_packed.element_type.width)
    misc_utils.static_assert(len(mY_packed.shape) == 2)
    misc_utils.static_assert(len(mDY_packed.shape) == 2)
    misc_utils.static_assert(len(mDZ_packed.shape) == 2)
    misc_utils.static_assert(len(mZdZ.shape) == 1)
    misc_utils.static_assert(len(mPos.shape) == 1)
    misc_utils.static_assert(len(mFreq.shape) == 1)
    misc_utils.static_assert(mY_packed.shape[1] == mDY_packed.shape[1])
    misc_utils.static_assert(mY_packed.shape[1] == mDZ_packed.shape[1])
    misc_utils.static_assert(mY_packed.shape[1] % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mDY_packed.shape[1] % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mDZ_packed.shape[1] % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mFreq.shape[0] == mY.shape[1])
    tiler_mn, tv_layout = layout_utils.make_layout_tv_from_shape(
        thread_shape=(thr_m, thr_n),
        thread_order="row",
        value_shape=(val_m, vector_size),
        value_order="row",
    )

    num_blocks = cute.ceil_div(mY_packed.shape[0], tiler_mn[0])
    num_threads = cute.size(tv_layout, mode=[0])
    kernel = rope_bwd_zdz_kernel(
        mY_packed=mY_packed,
        mDY_packed=mDY_packed,
        mDZ_packed=mDZ_packed,
        mZdZ=mZdZ,
        mPos=mPos,
        mFreq=mFreq,
        scale=scale,
        dtype=mY.element_type,
        tiler_mn=tiler_mn,
        tv_layout=tv_layout,
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        vector_size=vector_size,
    )
    kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[num_threads, 1, 1],
        cluster=None,
        smem=None,
        stream=stream,
    )
    return kernel.smem_usage()


@jit_cache
def _compile_rope_bwd_zdz(
    size: int,
    scale: float,
    dtype: type[cute.Numeric],
    pos_dtype: type[cute.Numeric],
    freq_dtype: type[cute.Numeric],
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> Callable:
    m = cute.sym_int()
    vector_size = cutlass.const_expr(constants.NUM_BITS_PER_COPY // dtype.width)
    mY = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mDY = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mDZ = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mZdZ = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(m,),
        stride=(1,),
        assumed_align=4,
    )
    mPos = cute.runtime.make_fake_tensor(
        dtype=pos_dtype,
        shape=(m,),
        stride=(1,),
        assumed_align=pos_dtype.width // 8,
    )
    mFreq = cute.runtime.make_fake_tensor(
        dtype=freq_dtype,
        shape=(size,),
        stride=(1,),
        assumed_align=freq_dtype.width // 8,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _rope_bwd_zdz,
        mY=mY,
        mDY=mDY,
        mDZ=mDZ,
        mZdZ=mZdZ,
        mPos=mPos,
        mFreq=mFreq,
        scale=scale,
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        stream=stream,
        options="--enable-tvm-ffi",
    )


def rope_bwd_zdz_(
    y: torch.Tensor,
    dy: torch.Tensor,
    dz: torch.Tensor,
    zdz: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    scale: float,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    fn = _compile_rope_bwd_zdz(
        size=y.shape[1],
        scale=scale,
        dtype=torch2cute_dtype_map[y.dtype],
        pos_dtype=torch2cute_dtype_map[pos.dtype],
        freq_dtype=torch2cute_dtype_map[freq.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    fn(
        y,
        dy,
        dz,
        zdz,
        pos,
        freq,
    )
