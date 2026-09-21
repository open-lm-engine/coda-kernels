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

    idY_packed = cute.make_identity_tensor(mY_packed.shape)
    config = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mY_packed.element_type,
        num_bits_per_copy=mY_packed.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )

    # the row's running sum of y * dy
    rZdZ = creation_utils.allocate_tensor_from_shape(
        shape=(val_m,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    rZdZ.fill(value=0.0)
    rPos = creation_utils.allocate_tensor_from_shape(
        shape=(val_m,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    # per tile: dz = R^T dy, and the tile's share of sum(y * dy)
    misc_utils.static_assert(mY_packed.shape[1] % tiler_mn[1] == 0)
    for tile_index in cutlass.range_constexpr(misc_utils.ceil_div(mY_packed.shape[1], tiler_mn[1])):
        gY_packed = cute.local_tile(mY_packed, tiler_mn, (bidx, tile_index))
        gDY_packed = cute.local_tile(mDY_packed, tiler_mn, (bidx, tile_index))
        gDZ_packed = cute.local_tile(mDZ_packed, tiler_mn, (bidx, tile_index))
        cY_packed = cute.local_tile(idY_packed, tiler_mn, (bidx, tile_index))
        copy_outputs_Y = memory_utils.copy(
            src=gY_packed,
            dst="rmem",
            crd=cY_packed,
            shape=mY_packed.shape,
            config=config,
            thread_index=tidx,
            smem_allocator=allocator,
        )
        copy_outputs_DY = memory_utils.copy(
            src=gDY_packed,
            dst="rmem",
            crd=cY_packed,
            shape=mDY_packed.shape,
            config=config,
            thread_index=tidx,
            smem_allocator=allocator,
        )
        tYrY_packed = copy_outputs_Y.dst_thread
        tYrDY_packed = copy_outputs_DY.dst_thread
        tYcY_packed = copy_outputs_Y.crd_thread
        if cutlass.const_expr(tile_index == 0):
            for row_index in cutlass.range_constexpr(val_m):
                row_coord, _ = tYcY_packed[row_index * vector_size]
                # a row past M is never stored, but its pos read must stay in bounds
                row_coord_clamped = cutlass.min(row_coord, mY_packed.shape[0] - 1)
                rPos[row_index] = mPos[row_coord_clamped].to(dtype=cute.Float32)
        tYrDZ_packed = creation_utils.allocate_tensor_like(
            tensor=tYrDY_packed,
            memspace="rmem",
            smem_allocator=allocator,
            dtype=mDZ_packed.element_type,
        )
        tYrY = cute.recast_tensor(tYrY_packed, dtype=dtype)
        tYrDY = cute.recast_tensor(tYrDY_packed, dtype=dtype)
        tYrDZ = cute.recast_tensor(tYrDZ_packed, dtype=dtype)
        for row_index in cutlass.range_constexpr(val_m):
            for col_index in cutlass.range_constexpr(vector_size):
                flat_index = row_index * vector_size + col_index
                _, col_coord = tYcY_packed[flat_index]
                freq_index = 2 * col_coord
                s, c = math_utils.rope_pos_freq(
                    pos=rPos[row_index],
                    freq_hi=mFreq[freq_index].to(dtype=cute.Float32),
                    freq_lo=mFreq[freq_index + 1].to(dtype=cute.Float32),
                )
                dy0 = tYrDY[2 * flat_index].to(dtype=cute.Float32)
                dy1 = tYrDY[2 * flat_index + 1].to(dtype=cute.Float32)
                dz0 = dy0 * c + dy1 * s
                dz1 = dy1 * c - dy0 * s
                y0 = tYrY[2 * flat_index].to(dtype=cute.Float32)
                y1 = tYrY[2 * flat_index + 1].to(dtype=cute.Float32)
                tYrDZ[2 * flat_index] = dz0.to(dtype=dtype)
                tYrDZ[2 * flat_index + 1] = dz1.to(dtype=dtype)
                # rope is orthogonal, so sum(y * dy) == sum(z * dz): the caller keeps only the rotated y
                rZdZ[row_index] = rZdZ[row_index] + y0 * dy0 + y1 * dy1
        _ = memory_utils.copy(
            src=tYrDZ_packed,
            dst=gDZ_packed,
            crd=tYcY_packed,
            shape=mDZ_packed.shape,
            config=config,
            thread_index=tidx,
            smem_allocator=allocator,
        )

    zdzs = []
    for row_index in cutlass.range_constexpr(val_m):
        zdz = cute.make_rmem_tensor((1,), cute.Float32)
        zdz[0] = rZdZ[row_index]
        zdzs.append(zdz.load())
    zdzs_reduced, _ = reduction_utils.reduce(
        zdzs,
        op="add",
        thread_shape=(thr_m, thr_n),
        smem_allocator=allocator,
        reduction_buffer=None,
    )
    # the norm's backward wants zdz / hidden, so scale rides the store
    for row_index in cutlass.range_constexpr(val_m):
        row_coord, _ = tYcY_packed[row_index * vector_size]
        row_in_bound = row_coord < mY_packed.shape[0]
        if ((tidx % thr_n) == 0) and row_in_bound:
            mZdZ[row_coord] = zdzs_reduced[row_index] * scale


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
    misc_utils.static_assert(mY_packed.element_type.width == 2 * mY.element_type.width)
    misc_utils.static_assert(mDY_packed.element_type.width == 2 * mDY.element_type.width)
    misc_utils.static_assert(mDZ_packed.element_type.width == 2 * mDZ.element_type.width)
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
