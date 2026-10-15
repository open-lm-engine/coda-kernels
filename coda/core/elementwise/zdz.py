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


@cute.kernel
def qknorm_rope_bwd_zdz_kernel(
    mDY_packed: cute.Tensor,
    mDV_packed: cute.Tensor | None,
    mDZ_packed: cute.Tensor,
    mZdZ: cute.Tensor,
    mDGamma: cute.Tensor,
    mPreact_packed: cute.Tensor,
    mV_packed: cute.Tensor,
    mRstd: cute.Tensor,
    mHeadMeanSq: cute.Tensor,
    mGamma: cute.Tensor,
    mPos: cute.Tensor,
    mFreq: cute.Tensor,
    head_dim: cutlass.Constexpr[int],
    num_heads_q: cutlass.Constexpr[int],
    num_heads_k: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
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

    tile_N_packed = cutlass.const_expr(tiler_mn[1])
    num_threads = cutlass.const_expr(cute.size(tv_layout, mode=[0]))
    # a lane covers 2 * vector_size columns (two per packed word),
    # so one head spans lanes_per_head lanes
    lanes_per_head = cutlass.const_expr(head_dim // (2 * vector_size))
    # the CTA's lanes form num_head_groups head-wide runs, each holding one partial dgamma row, summed at the end
    num_head_groups = cutlass.const_expr(num_threads // lanes_per_head)
    num_tiles_q = cutlass.const_expr((head_dim * num_heads_q) // (2 * tile_N_packed))
    num_tiles_k = cutlass.const_expr((head_dim * num_heads_k) // (2 * tile_N_packed))
    num_tiles_qk = cutlass.const_expr(num_tiles_q + num_tiles_k)
    misc_utils.static_assert((head_dim % (2 * vector_size)) == 0)
    misc_utils.static_assert((num_threads % lanes_per_head) == 0)
    misc_utils.static_assert(((head_dim * num_heads_q) % (2 * tile_N_packed)) == 0)
    misc_utils.static_assert(((head_dim * num_heads_k) % (2 * tile_N_packed)) == 0)
    sDGamma = creation_utils.allocate_tensor_from_shape(
        shape=(num_head_groups, 2 * head_dim),
        order="row",
        dtype=cute.Float32,
        memspace="smem",
        smem_allocator=allocator,
        byte_alignment=4,
    )

    idDY_packed = cute.make_identity_tensor(mDY_packed.shape)
    config = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mDY_packed.element_type,
        num_bits_per_copy=mDY_packed.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )

    rDN = creation_utils.allocate_tensor_from_shape(
        shape=(2 * vector_size,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    # the row's running sum of z * dz
    rZdZ = creation_utils.allocate_tensor_from_shape(
        shape=(val_m,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    # a lane sits at the same offset within every head it visits, so its gamma and dgamma are tile-invariant
    rDGammaQ = creation_utils.allocate_tensor_from_shape(
        shape=(2 * vector_size,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    rDGammaK = creation_utils.allocate_tensor_from_shape(
        shape=(2 * vector_size,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    # preact is the raw accumulator: z = rstd * preact, with rstd the input norm's row scale
    rRstd = creation_utils.allocate_tensor_from_shape(
        shape=(val_m,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    rGammaQ = creation_utils.allocate_tensor_from_shape(
        shape=(2 * vector_size,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    rGammaK = creation_utils.allocate_tensor_from_shape(
        shape=(2 * vector_size,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    rPos = creation_utils.allocate_tensor_from_shape(
        shape=(val_m,),
        order="row",
        dtype=cute.Float32,
        memspace="rmem",
    )
    rZdZ.fill(value=0.0)
    rDGammaQ.fill(value=0.0)
    rDGammaK.fill(value=0.0)

    # tiles never straddle the [Q | K | V] regions, so each tile's region is static
    misc_utils.static_assert(mDY_packed.shape[1] % tiler_mn[1] == 0)
    for tile_index in cutlass.range_constexpr(misc_utils.ceil_div(mDY_packed.shape[1], tiler_mn[1])):
        gDY_packed = cute.local_tile(mDY_packed, tiler_mn, (bidx, tile_index))
        gDZ_packed = cute.local_tile(mDZ_packed, tiler_mn, (bidx, tile_index))
        cDY_packed = cute.local_tile(idDY_packed, tiler_mn, (bidx, tile_index))
        copy_outputs_DY = memory_utils.copy(
            src=gDY_packed,
            dst="rmem",
            crd=cDY_packed,
            shape=mDY_packed.shape,
            config=config,
            thread_index=tidx,
            smem_allocator=allocator,
        )
        tDYrDY_packed = copy_outputs_DY.dst_thread
        tDYcDY_packed = copy_outputs_DY.crd_thread
        # rstd, gamma and pos don't change across tiles: load them once
        if cutlass.const_expr(tile_index == 0):
            for row_index in cutlass.range_constexpr(val_m):
                row_coord, _ = tDYcDY_packed[row_index * vector_size]
                # a row past M is never stored, but its rstd and pos reads must stay in bounds
                row_coord_clamped = cutlass.min(row_coord, mDY_packed.shape[0] - 1)
                rRstd[row_index] = mRstd[row_coord_clamped].to(dtype=cute.Float32)
                rPos[row_index] = mPos[row_coord_clamped].to(dtype=cute.Float32)
            # gamma is [gamma_q | gamma_k]
            for col_index in cutlass.range_constexpr(vector_size):
                _, col_coord_packed = tDYcDY_packed[col_index]
                gamma_index = (2 * col_coord_packed) % head_dim
                rGammaQ[2 * col_index] = mGamma[gamma_index].to(dtype=cute.Float32)
                rGammaQ[2 * col_index + 1] = mGamma[gamma_index + 1].to(dtype=cute.Float32)
                rGammaK[2 * col_index] = mGamma[head_dim + gamma_index].to(dtype=cute.Float32)
                rGammaK[2 * col_index + 1] = mGamma[head_dim + gamma_index + 1].to(dtype=cute.Float32)
        # copies move packed words; the math indexes single elements through the views below
        tDYrDZ_packed = creation_utils.allocate_tensor_like(
            tensor=tDYrDY_packed,
            memspace="rmem",
            smem_allocator=allocator,
            dtype=mDZ_packed.element_type,
        )
        tDYrDY = cute.recast_tensor(tDYrDY_packed, dtype=dtype)
        tDYrDZ = cute.recast_tensor(tDYrDZ_packed, dtype=dtype)

        if cutlass.const_expr(tile_index < num_tiles_qk):
            if cutlass.const_expr(tile_index < num_tiles_q):
                rGamma = rGammaQ
                rDGamma = rDGammaQ
            else:
                rGamma = rGammaK
                rDGamma = rDGammaK
            gPreact_packed = cute.local_tile(mPreact_packed, tiler_mn, (bidx, tile_index))
            copy_outputs_Preact = memory_utils.copy(
                src=gPreact_packed,
                dst="rmem",
                crd=cDY_packed,
                shape=mDY_packed.shape,
                config=config,
                thread_index=tidx,
                smem_allocator=allocator,
            )
            tDYrPreact_packed = copy_outputs_Preact.dst_thread
            tDYrPreact = cute.recast_tensor(tDYrPreact_packed, dtype=dtype)
            for row_index in cutlass.range_constexpr(val_m):
                row_coord, col_coord_packed_begin = tDYcDY_packed[row_index * vector_size]
                # the last tile can run past M. rows beyond M never load or store their tiles,
                # read head_mean_sq from row M - 1 to stay in bounds,
                # and are skipped by dgamma, which sums over rows
                row_coord_clamped = cutlass.min(row_coord, mDY_packed.shape[0] - 1)
                row_in_bound = row_coord < mDY_packed.shape[0]
                head_index = (2 * col_coord_packed_begin) // head_dim
                rms = cute.math.rsqrt(
                    mHeadMeanSq[row_coord_clamped, head_index] + eps,
                    fastmath=True,
                )
                rms_cubed = rms * rms * rms
                drms = cute.Float32.zero
                for col_index in cutlass.range_constexpr(vector_size):
                    flat_index = row_index * vector_size + col_index
                    _, col_coord_packed = tDYcDY_packed[flat_index]
                    freq_index = 2 * col_coord_packed
                    s, c = math_utils.rope_pos_freq(
                        pos=rPos[row_index],
                        freq_hi=mFreq[freq_index].to(dtype=cute.Float32),
                        freq_lo=mFreq[freq_index + 1].to(dtype=cute.Float32),
                    )
                    dy0 = tDYrDY[2 * flat_index].to(dtype=cute.Float32)
                    dy1 = tDYrDY[2 * flat_index + 1].to(dtype=cute.Float32)
                    # dn = R^T dy is the gradient of the qk-norm output n = gamma * z * rms
                    dn0 = dy0 * c + dy1 * s
                    dn1 = dy1 * c - dy0 * s
                    z0 = tDYrPreact[2 * flat_index].to(dtype=cute.Float32) * rRstd[row_index]
                    z1 = tDYrPreact[2 * flat_index + 1].to(dtype=cute.Float32) * rRstd[row_index]
                    g0 = rGamma[2 * col_index]
                    g1 = rGamma[2 * col_index + 1]
                    rDN[2 * col_index] = dn0
                    rDN[2 * col_index + 1] = dn1
                    drms = drms + dn0 * g0 * z0 + dn1 * g1 * z1

                # zdz needs this head's sum(z * dz). the norm makes it exactly eps * rms^3 * drms.
                # a direct sum of z * dz would be swamped by error: dz is rounded, and z is rebuilt
                # from the bf16 preact, so mean(z^2) misses the forward's head_mean_sq by far more than eps.
                # drms is still split across the head's lanes here, so each lane adds its own part
                rZdZ[row_index] = rZdZ[row_index] + eps * rms_cubed * drms
                if cutlass.const_expr(lanes_per_head > 1):
                    drms = cute.arch.warp_reduction(
                        drms,
                        op=operator.add,
                        threads_in_group=lanes_per_head,
                    )

                # dssq2 = 2 * dL/dssq
                dssq2 = -drms * rms_cubed / head_dim
                for col_index in cutlass.range_constexpr(vector_size):
                    flat_index = row_index * vector_size + col_index
                    g0 = rGamma[2 * col_index]
                    g1 = rGamma[2 * col_index + 1]
                    dn0 = rDN[2 * col_index]
                    dn1 = rDN[2 * col_index + 1]
                    z0 = tDYrPreact[2 * flat_index].to(dtype=cute.Float32) * rRstd[row_index]
                    z1 = tDYrPreact[2 * flat_index + 1].to(dtype=cute.Float32) * rRstd[row_index]
                    tDYrDZ[2 * flat_index] = (rms * g0 * dn0 + z0 * dssq2).to(dtype=tDYrDZ.element_type)
                    tDYrDZ[2 * flat_index + 1] = (rms * g1 * dn1 + z1 * dssq2).to(dtype=tDYrDZ.element_type)

                    if row_in_bound:
                        rDGamma[2 * col_index] = rDGamma[2 * col_index] + dn0 * z0 * rms
                        rDGamma[2 * col_index + 1] = rDGamma[2 * col_index + 1] + dn1 * z1 * rms

        else:
            gV_packed = cute.local_tile(mV_packed, tiler_mn, (bidx, tile_index - num_tiles_qk))
            copy_outputs_V = memory_utils.copy(
                src=gV_packed,
                dst="rmem",
                crd=cDY_packed,
                shape=mDY_packed.shape,
                config=config,
                thread_index=tidx,
                smem_allocator=allocator,
            )
            tDYrV_packed = copy_outputs_V.dst_thread
            tDYrV = cute.recast_tensor(tDYrV_packed, dtype=dtype)
            if cutlass.const_expr(mDV_packed is not None):
                gDV_packed = cute.local_tile(mDV_packed, tiler_mn, (bidx, tile_index - num_tiles_qk))
                copy_outputs_DV = memory_utils.copy(
                    src=gDV_packed,
                    dst="rmem",
                    crd=cDY_packed,
                    shape=mDY_packed.shape,
                    config=config,
                    thread_index=tidx,
                    smem_allocator=allocator,
                )
                tDYrDV_packed = copy_outputs_DV.dst_thread
                tDYrDV = cute.recast_tensor(tDYrDV_packed, dtype=dtype)
            # v skips the qk-norm and rope, so its gradient passes through unchanged
            for row_index in cutlass.range_constexpr(val_m):
                for col_index in cutlass.range_constexpr(vector_size):
                    flat_index = row_index * vector_size + col_index
                    dz0 = tDYrDY[2 * flat_index].to(dtype=cute.Float32)
                    dz1 = tDYrDY[2 * flat_index + 1].to(dtype=cute.Float32)
                    # v also feeds the next block's XSA, and dv is that gradient
                    if cutlass.const_expr(mDV_packed is not None):
                        dz0 = dz0 + tDYrDV[2 * flat_index].to(dtype=cute.Float32)
                        dz1 = dz1 + tDYrDV[2 * flat_index + 1].to(dtype=cute.Float32)
                    v0 = tDYrV[2 * flat_index].to(dtype=cute.Float32)
                    v1 = tDYrV[2 * flat_index + 1].to(dtype=cute.Float32)
                    tDYrDZ[2 * flat_index] = dz0.to(dtype=tDYrDZ.element_type)
                    tDYrDZ[2 * flat_index + 1] = dz1.to(dtype=tDYrDZ.element_type)
                    # z is v on the v columns, so this adds v's share of sum(z * dz)
                    rZdZ[row_index] = rZdZ[row_index] + v0 * dz0 + v1 * dz1

        _ = memory_utils.copy(
            src=tDYrDZ_packed,
            dst=gDZ_packed,
            crd=tDYcDY_packed,
            shape=mDZ_packed.shape,
            config=config,
            thread_index=tidx,
            smem_allocator=allocator,
        )

    # each run of lanes_per_head lanes covers one head, so a gamma column has one partial per run
    head_group = tidx // lanes_per_head
    for col_index in cutlass.range_constexpr(vector_size):
        _, col_coord_packed = tDYcDY_packed[col_index]
        gamma_index = (2 * col_coord_packed) % head_dim
        sDGamma[head_group, gamma_index] = rDGammaQ[2 * col_index]
        sDGamma[head_group, gamma_index + 1] = rDGammaQ[2 * col_index + 1]
        sDGamma[head_group, head_dim + gamma_index] = rDGammaK[2 * col_index]
        sDGamma[head_group, head_dim + gamma_index + 1] = rDGammaK[2 * col_index + 1]

    cute.arch.barrier()
    for i in cutlass.range_constexpr(misc_utils.ceil_div(2 * head_dim, num_threads)):
        j = i * num_threads + tidx
        if j < 2 * head_dim:
            dg = cute.Float32.zero
            for group in cutlass.range_constexpr(num_head_groups):
                dg = dg + sDGamma[group, j]
            mDGamma[bidx, j] = dg

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
        row_coord, _ = tDYcDY_packed[row_index * vector_size]
        row_in_bound = row_coord < mDY_packed.shape[0]
        if ((tidx % thr_n) == 0) and row_in_bound:
            mZdZ[row_coord] = zdzs_reduced[row_index] * scale


@cute.jit
def _qknorm_rope_bwd_zdz(
    mDY: cute.Tensor,
    mDV: cute.Tensor | None,
    mDZ: cute.Tensor,
    mZdZ: cute.Tensor,
    mDGamma: cute.Tensor,
    mPreact: cute.Tensor,
    mV: cute.Tensor,
    mRstd: cute.Tensor,
    mHeadMeanSq: cute.Tensor,
    mGamma: cute.Tensor,
    mPos: cute.Tensor,
    mFreq: cute.Tensor,
    head_dim: cutlass.Constexpr[int],
    num_heads_q: cutlass.Constexpr[int],
    num_heads_k: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    thr_m: cutlass.Constexpr[int],
    thr_n: cutlass.Constexpr[int],
    val_m: cutlass.Constexpr[int],
    stream: cuda.CUstream,
) -> int:
    mDY_packed = layout_utils.recast_tensor(mDY, dtype=cute.Int32)
    if cutlass.const_expr(mDV is not None):
        mDV_packed = layout_utils.recast_tensor(mDV, dtype=cute.Int32)
    else:
        mDV_packed = None
    mDZ_packed = layout_utils.recast_tensor(mDZ, dtype=cute.Int32)
    mPreact_packed = layout_utils.recast_tensor(mPreact, dtype=cute.Int32)
    mV_packed = layout_utils.recast_tensor(mV, dtype=cute.Int32)
    vector_size = cutlass.const_expr(constants.NUM_BITS_PER_COPY // mDY_packed.element_type.width)
    num_heads_qk = cutlass.const_expr(num_heads_q + num_heads_k)
    size_q = cutlass.const_expr(head_dim * num_heads_q)
    size_k = cutlass.const_expr(head_dim * num_heads_k)
    lanes_per_head = cutlass.const_expr(head_dim // (2 * vector_size))
    misc_utils.static_assert(mDY_packed.element_type.width == 2 * mDY.element_type.width)
    if cutlass.const_expr(mDV is not None):
        misc_utils.static_assert(mDV_packed.element_type.width == 2 * mDV.element_type.width)
    misc_utils.static_assert(mDZ_packed.element_type.width == 2 * mDZ.element_type.width)
    misc_utils.static_assert(mPreact_packed.element_type.width == 2 * mPreact.element_type.width)
    misc_utils.static_assert(mV_packed.element_type.width == 2 * mV.element_type.width)
    misc_utils.static_assert(len(mDY_packed.shape) == 2)
    if cutlass.const_expr(mDV is not None):
        misc_utils.static_assert(len(mDV_packed.shape) == 2)
    misc_utils.static_assert(len(mDZ_packed.shape) == 2)
    misc_utils.static_assert(len(mZdZ.shape) == 1)
    misc_utils.static_assert(len(mDGamma.shape) == 2)
    misc_utils.static_assert(len(mPreact_packed.shape) == 2)
    misc_utils.static_assert(len(mV_packed.shape) == 2)
    misc_utils.static_assert(len(mRstd.shape) == 1)
    misc_utils.static_assert(len(mHeadMeanSq.shape) == 2)
    misc_utils.static_assert(len(mGamma.shape) == 1)
    misc_utils.static_assert(len(mPos.shape) == 1)
    misc_utils.static_assert(len(mFreq.shape) == 1)
    misc_utils.static_assert(mHeadMeanSq.shape[1] == num_heads_qk)
    misc_utils.static_assert(mPreact.shape[1] == (head_dim * num_heads_qk))
    misc_utils.static_assert(mDY_packed.shape[1] == (mPreact_packed.shape[1] + mV_packed.shape[1]))
    misc_utils.static_assert(mDY_packed.shape[1] == mDZ_packed.shape[1])
    if cutlass.const_expr(mDV is not None):
        misc_utils.static_assert(mDV_packed.shape[1] == mV_packed.shape[1])
    misc_utils.static_assert(mDY_packed.shape[1] % (thr_n * vector_size) == 0)
    if cutlass.const_expr(mDV is not None):
        misc_utils.static_assert(mDV_packed.shape[1] % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mDZ_packed.shape[1] % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mPreact_packed.shape[1] % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mV_packed.shape[1] % (thr_n * vector_size) == 0)
    # q and k use different gammas, so no tile may straddle the q | k boundary
    misc_utils.static_assert((size_q // 2) % (thr_n * vector_size) == 0)
    misc_utils.static_assert((size_k // 2) % (thr_n * vector_size) == 0)
    misc_utils.static_assert(mDGamma.shape[1] == (2 * head_dim))
    misc_utils.static_assert(mGamma.shape[0] == (2 * head_dim))
    misc_utils.static_assert(mFreq.shape[0] == mPreact.shape[1])
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

    num_blocks = cute.ceil_div(mDY_packed.shape[0], tiler_mn[0])
    num_threads = cute.size(tv_layout, mode=[0])
    kernel = qknorm_rope_bwd_zdz_kernel(
        mDY_packed=mDY_packed,
        mDV_packed=mDV_packed,
        mDZ_packed=mDZ_packed,
        mZdZ=mZdZ,
        mDGamma=mDGamma,
        mPreact_packed=mPreact_packed,
        mV_packed=mV_packed,
        mRstd=mRstd,
        mHeadMeanSq=mHeadMeanSq,
        mGamma=mGamma,
        mPos=mPos,
        mFreq=mFreq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        scale=scale,
        dtype=mDY.element_type,
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
def _compile_qknorm_rope_bwd_zdz(
    size: int,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    scale: float,
    has_dv: bool,
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
    size_qk = cutlass.const_expr(size_q + size_k)
    size_v = cutlass.const_expr(size - size_qk)
    vector_size = cutlass.const_expr(constants.NUM_BITS_PER_COPY // dtype.width)
    # size is the full [Q | K | V] width: whole v heads follow q and k
    misc_utils.static_assert(size_v > 0)
    misc_utils.static_assert((size_v % head_dim) == 0)
    misc_utils.static_assert((vector_size % 2) == 0)
    mDY = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    if cutlass.const_expr(has_dv):
        mDV = cute.runtime.make_fake_tensor(
            dtype=dtype,
            shape=(m, size_v),
            stride=(cute.sym_int64(divisibility=vector_size), 1),
            assumed_align=16,
        )
    else:
        mDV = None
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
    mDGamma = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(cute.sym_int(), 2 * head_dim),
        stride=(cute.sym_int64(divisibility=1), 1),
        assumed_align=4,
    )
    mPreact = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size_qk),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mV = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size_v),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mRstd = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(m,),
        stride=(1,),
        assumed_align=4,
    )
    mHeadMeanSq = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(m, num_heads_qk),
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
        shape=(size_qk,),
        stride=(1,),
        assumed_align=freq_dtype.width // 8,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _qknorm_rope_bwd_zdz,
        mDY=mDY,
        mDV=mDV,
        mDZ=mDZ,
        mZdZ=mZdZ,
        mDGamma=mDGamma,
        mPreact=mPreact,
        mV=mV,
        mRstd=mRstd,
        mHeadMeanSq=mHeadMeanSq,
        mGamma=mGamma,
        mPos=mPos,
        mFreq=mFreq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        scale=scale,
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        stream=stream,
        options="--enable-tvm-ffi",
    )


def qknorm_rope_bwd_zdz_(
    dy: torch.Tensor,
    dv: torch.Tensor | None,
    dz: torch.Tensor,
    zdz: torch.Tensor,
    dgamma: torch.Tensor,
    preact: torch.Tensor,
    v: torch.Tensor,
    rstd: torch.Tensor,
    head_mean_sq: torch.Tensor,
    gamma: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    scale: float,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    fn = _compile_qknorm_rope_bwd_zdz(
        size=dy.shape[1],
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        scale=scale,
        has_dv=dv is not None,
        dtype=torch2cute_dtype_map[dy.dtype],
        pos_dtype=torch2cute_dtype_map[pos.dtype],
        freq_dtype=torch2cute_dtype_map[freq.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    fn(
        dy,
        dv,
        dz,
        zdz,
        dgamma,
        preact,
        v,
        rstd,
        head_mean_sq,
        gamma,
        pos,
        freq,
    )
