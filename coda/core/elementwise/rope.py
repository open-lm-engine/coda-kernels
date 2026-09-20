import operator
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


@cute.kernel
def qknorm_rope_bwd_kernel(
    mDX_packed: cute.Tensor,
    mDQ_packed: cute.Tensor,
    mDK_packed: cute.Tensor,
    mDGamma: cute.Tensor,
    mX_packed: cute.Tensor,
    mHeadMeanSq: cute.Tensor,
    mGamma: cute.Tensor,
    mPos: cute.Tensor,
    mFreq: cute.Tensor,
    head_dim: cutlass.Constexpr[int],
    num_heads_q: cutlass.Constexpr[int],
    num_heads_k: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
    dtype: type[cute.Numeric],
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    thr_m: cutlass.Constexpr[int],
    thr_n: cutlass.Constexpr[int],
    val_m: cutlass.Constexpr[int],
    vector_size: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    tile_N_packed = cutlass.const_expr(tiler_mn[1])
    num_threads = cutlass.const_expr(cute.size(tv_layout, mode=[0]))
    sDGamma = creation_utils.allocate_tensor_from_shape(
        shape=(thr_m, 2 * tile_N_packed),
        order="row",
        dtype=cute.Float32,
        memspace="smem",
        smem_allocator=allocator,
        byte_alignment=4,
    )

    idX_packed = cute.make_identity_tensor(mX_packed.shape)
    gDX_packed = cute.local_tile(mDX_packed, tiler_mn, (bidx, bidy))
    gX_packed = cute.local_tile(mX_packed, tiler_mn, (bidx, bidy))
    cX_packed = cute.local_tile(idX_packed, tiler_mn, (bidx, bidy))
    config = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mX_packed.element_type,
        num_bits_per_copy=mX_packed.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    copy_outputs_X = memory_utils.copy(
        src=gX_packed,
        dst="rmem",
        crd=cX_packed,
        shape=mX_packed.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )
    tXrX_packed = copy_outputs_X.dst_thread
    tXcX_packed = copy_outputs_X.crd_thread

    misc_utils.static_assert(mDQ_packed.element_type == mDK_packed.element_type)
    tXrDY_packed = creation_utils.allocate_tensor_like(
        tensor=tXrX_packed,
        memspace="rmem",
        dtype=mDQ_packed.element_type,
    )
    num_blocks_q = cutlass.const_expr(
        (head_dim * num_heads_q) //
        (2 * tile_N_packed)
    )
    if bidy < num_blocks_q:
        gDQ_packed = cute.local_tile(mDQ_packed, tiler_mn, (bidx, bidy))
        _ = memory_utils.copy(
            src=gDQ_packed,
            dst=tXrDY_packed,
            crd=cX_packed,
            shape=mX_packed.shape,
            config=config,
            thread_index=tidx,
            smem_allocator=allocator,
        )
    else:
        gDK_packed = cute.local_tile(mDK_packed, tiler_mn, (bidx, bidy - num_blocks_q))
        _ = memory_utils.copy(
            src=gDK_packed,
            dst=tXrDY_packed,
            crd=cX_packed,
            shape=mX_packed.shape,
            config=config,
            thread_index=tidx,
            smem_allocator=allocator,
        )

    tXrDX_packed = creation_utils.allocate_tensor_like(
        tensor=tXrX_packed,
        memspace="rmem",
        smem_allocator=allocator,
        dtype=mDX_packed.element_type,
    )
    tXrDY = cute.recast_tensor(tXrDY_packed, dtype=dtype)
    tXrDX = cute.recast_tensor(tXrDX_packed, dtype=dtype)
    tXrX = cute.recast_tensor(tXrX_packed, dtype=dtype)

    rDZ = creation_utils.allocate_tensor_from_shape(
        shape=(2 * vector_size,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    rDGamma = creation_utils.allocate_tensor_from_shape(
        shape=(2 * vector_size,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    cute.filter_zeros(rDGamma).fill(0.0)
    lanes_per_head = cutlass.const_expr(head_dim // (2 * vector_size))

    for row_index in cutlass.range_constexpr(val_m):
        row_coord, col_coord_begin = tXcX_packed[row_index * vector_size]
        # a row past M clamps its reads to the last row and is left out of drms and dgamma
        row_coord_clamped = cutlass.min(row_coord, mX_packed.shape[0] - 1)
        row_in_bound = row_coord < mX_packed.shape[0]

        # gamma is [gamma_q | gamma_k]
        head_idx = (2 * col_coord_begin) // head_dim
        gamma_offset = head_dim if head_idx >= num_heads_q else 0

        rms = cute.math.rsqrt(
            mHeadMeanSq[row_coord_clamped, head_idx] + eps,
            fastmath=True,
        )
        drms = cute.Float32.zero
        for col_index in cutlass.range_constexpr(vector_size):
            flat_index = row_index * vector_size + col_index
            _, col_coord = tXcX_packed[flat_index]
            gamma_index = (2 * col_coord) % head_dim
            freq_index = 2 * col_coord
            s, c = math_utils.rope_pos_freq(
                pos=mPos[row_coord_clamped].to(dtype=cute.Float32),
                freq_hi=mFreq[freq_index].to(dtype=cute.Float32),
                freq_lo=mFreq[freq_index + 1].to(dtype=cute.Float32),
            )
            dy0 = tXrDY[2 * flat_index].to(dtype=cute.Float32)
            dy1 = tXrDY[2 * flat_index + 1].to(dtype=cute.Float32)
            dz0 = dy0 * c + dy1 * s
            dz1 = dy1 * c - dy0 * s
            x0 = tXrX[2 * flat_index].to(dtype=cute.Float32)
            x1 = tXrX[2 * flat_index + 1].to(dtype=cute.Float32)
            g0 = mGamma[gamma_offset + gamma_index].to(dtype=cute.Float32)
            g1 = mGamma[gamma_offset + gamma_index + 1].to(dtype=cute.Float32)
            rDZ[2 * col_index] = dz0
            rDZ[2 * col_index + 1] = dz1

            if row_in_bound:
                drms = drms + dz0 * g0 * x0 + dz1 * g1 * x1

        if cutlass.const_expr(lanes_per_head > 1):
            drms = cute.arch.warp_reduction(
                drms,
                op=operator.add,
                threads_in_group=lanes_per_head,
            )

        # dssq2 = 2 * dL/dssq
        dssq2 = -drms * rms * rms * rms / head_dim
        for col_index in cutlass.range_constexpr(vector_size):
            flat_index = row_index * vector_size + col_index
            _, col_coord = tXcX_packed[flat_index]
            gamma_index = (2 * col_coord) % head_dim
            g0 = mGamma[gamma_offset + gamma_index].to(dtype=cute.Float32)
            g1 = mGamma[gamma_offset + gamma_index + 1].to(dtype=cute.Float32)
            dz0 = rDZ[2 * col_index]
            dz1 = rDZ[2 * col_index + 1]
            x0 = tXrX[2 * flat_index].to(dtype=cute.Float32)
            x1 = tXrX[2 * flat_index + 1].to(dtype=cute.Float32)
            tXrDX[2 * flat_index] = (rms * g0 * dz0 + x0 * dssq2).to(dtype=tXrDX.element_type)
            tXrDX[2 * flat_index + 1] = (rms * g1 * dz1 + x1 * dssq2).to(dtype=tXrDX.element_type)

            if row_in_bound:
                rDGamma[2 * col_index] = rDGamma[2 * col_index] + dz0 * x0 * rms
                rDGamma[2 * col_index + 1] = rDGamma[2 * col_index + 1] + dz1 * x1 * rms

    thr_row = tidx // thr_n
    col_coord_packed_offset = bidy * tile_N_packed
    for col_index in cutlass.range_constexpr(vector_size):
        _, col_coord_packed = tXcX_packed[col_index]
        col_coord_local = 2 * (col_coord_packed - col_coord_packed_offset)
        sDGamma[thr_row, col_coord_local] = rDGamma[2 * col_index]
        sDGamma[thr_row, col_coord_local + 1] = rDGamma[2 * col_index + 1]

    _ = memory_utils.copy(
        src=tXrDX_packed,
        dst=gDX_packed,
        crd=tXcX_packed,
        shape=mDX_packed.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )

    cute.arch.barrier()
    for i in cutlass.range_constexpr(misc_utils.ceil_div(2 * tile_N_packed, num_threads)):
        j = i * num_threads + tidx
        if j < 2 * tile_N_packed:
            dg = cute.Float32.zero
            for row in cutlass.range_constexpr(thr_m):
                dg = dg + sDGamma[row, j]
            mDGamma[bidx, bidy * 2 * tile_N_packed + j] = dg


@cute.jit
def _qknorm_rope_bwd(
    mDX: cute.Tensor,
    mDQ: cute.Tensor,
    mDK: cute.Tensor,
    mDGamma: cute.Tensor,
    mX: cute.Tensor,
    mHeadMeanSq: cute.Tensor,
    mGamma: cute.Tensor,
    mPos: cute.Tensor,
    mFreq: cute.Tensor,
    head_dim: cutlass.Constexpr[int],
    num_heads_q: cutlass.Constexpr[int],
    num_heads_k: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
    thr_m: cutlass.Constexpr[int],
    thr_n: cutlass.Constexpr[int],
    val_m: cutlass.Constexpr[int],
    stream: cuda.CUstream,
) -> int:
    mDX_packed = layout_utils.recast_tensor(mDX, dtype=cute.Int32)
    mDQ_packed = layout_utils.recast_tensor(mDQ, dtype=cute.Int32)
    mDK_packed = layout_utils.recast_tensor(mDK, dtype=cute.Int32)
    mX_packed = layout_utils.recast_tensor(mX, dtype=cute.Int32)
    vector_size = cutlass.const_expr(constants.NUM_BITS_PER_COPY // mX_packed.element_type.width)
    num_heads_qk = cutlass.const_expr(num_heads_q + num_heads_k)
    lanes_per_head = cutlass.const_expr(head_dim // (2 * vector_size))
    misc_utils.static_assert(len(mDX_packed.shape) == 2)
    misc_utils.static_assert(len(mDQ_packed.shape) == 2)
    misc_utils.static_assert(len(mDK_packed.shape) == 2)
    misc_utils.static_assert(len(mDGamma.shape) == 2)
    misc_utils.static_assert(len(mX_packed.shape) == 2)
    misc_utils.static_assert(len(mHeadMeanSq.shape) == 2)
    misc_utils.static_assert(len(mGamma.shape) == 1)
    misc_utils.static_assert(len(mPos.shape) == 1)
    misc_utils.static_assert(len(mFreq.shape) == 1)
    misc_utils.static_assert(mX.shape[1] == (head_dim * num_heads_qk))
    misc_utils.static_assert(mDQ.shape[1] == (head_dim * num_heads_q))
    misc_utils.static_assert(mDK.shape[1] == (head_dim * num_heads_k))
    misc_utils.static_assert(mX_packed.shape[1] == mDX_packed.shape[1])
    misc_utils.static_assert(mX_packed.shape[1] == (mDQ_packed.shape[1] + mDK_packed.shape[1]))
    misc_utils.static_assert(mX_packed.shape[1] % vector_size == 0)
    misc_utils.static_assert(mX_packed.shape[1] % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mDQ_packed.shape[1] % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mDK_packed.shape[1] % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mDGamma.shape[1] == mX.shape[1])
    misc_utils.static_assert(mGamma.shape[0] == (2 * head_dim))
    misc_utils.static_assert(mFreq.shape[0] == mX.shape[1])
    misc_utils.static_assert((head_dim % (2 * vector_size)) == 0)
    misc_utils.static_assert(lanes_per_head <= 32)
    misc_utils.static_assert((thr_n % lanes_per_head) == 0)
    misc_utils.static_assert(misc_utils.is_power_of_2(lanes_per_head))
    tiler_mn, tv_layout = layout_utils.make_layout_tv_from_shape(
        thread_shape=(thr_m, thr_n),
        thread_order="row",
        value_shape=(val_m, vector_size),
        value_order="row",
    )

    # ((TileM, TileN), (RestM, RestN))
    gX_packed = cute.zipped_divide(mX_packed, tiler_mn)
    num_blocks = gX_packed.shape[1]
    num_threads = cute.size(tv_layout, mode=[0])
    misc_utils.static_assert(len(num_blocks) == 2)
    kernel = qknorm_rope_bwd_kernel(
        mDX_packed=mDX_packed,
        mDQ_packed=mDQ_packed,
        mDK_packed=mDK_packed,
        mDGamma=mDGamma,
        mX_packed=mX_packed,
        mHeadMeanSq=mHeadMeanSq,
        mGamma=mGamma,
        mPos=mPos,
        mFreq=mFreq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        dtype=mX.element_type,
        tiler_mn=tiler_mn,
        tv_layout=tv_layout,
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        vector_size=vector_size,
    )
    kernel.launch(
        grid=[*num_blocks, 1],
        block=[num_threads, 1, 1],
        cluster=None,
        smem=None,
        stream=stream,
    )
    return kernel.smem_usage()


@jit_cache
def _compile_qknorm_rope_bwd(
    size: int,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    dtype: type[cute.Numeric],
    pos_dtype: type[cute.Numeric],
    freq_dtype: type[cute.Numeric],
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> Callable:
    m = cute.sym_int()
    num_heads_qk = cutlass.const_expr(num_heads_q + num_heads_k)
    size_q = cutlass.const_expr(head_dim * num_heads_q)
    size_k = cutlass.const_expr(head_dim * num_heads_k)
    vector_size = cutlass.const_expr(constants.NUM_BITS_PER_COPY // dtype.width)
    misc_utils.static_assert(size == (head_dim * num_heads_qk))
    misc_utils.static_assert((vector_size % 2) == 0)
    mDX = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mDQ = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size_q),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mDK = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size_k),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mDGamma = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(cute.sym_int(), size),
        stride=(cute.sym_int64(divisibility=1), 1),
        assumed_align=4,
    )
    mX = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mHeadMeanSq = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(m, size // head_dim),
        stride=(cute.sym_int64(divisibility=1), 1),
        assumed_align=4,
    )
    mGamma = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(2 * head_dim,),
        stride=(1,),
        assumed_align=dtype.width // 8,
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
        _qknorm_rope_bwd,
        mDX=mDX,
        mDQ=mDQ,
        mDK=mDK,
        mDGamma=mDGamma,
        mX=mX,
        mHeadMeanSq=mHeadMeanSq,
        mGamma=mGamma,
        mPos=mPos,
        mFreq=mFreq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        stream=stream,
        options="--enable-tvm-ffi",
    )


def qknorm_rope_bwd_(
    dx: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dgamma: torch.Tensor,
    x: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    fn = _compile_qknorm_rope_bwd(
        size=x.shape[1],
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        dtype=torch2cute_dtype_map[x.dtype],
        pos_dtype=torch2cute_dtype_map[pos.dtype],
        freq_dtype=torch2cute_dtype_map[freq.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    fn(dx, dq, dk, dgamma, x, head_mean_sq, gamma, pos, freq)
