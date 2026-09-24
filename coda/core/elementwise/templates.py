import torch
import cutlass
import dataclasses
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from typing import Callable

from quack.cache import jit_cache
from quack.autotuner import autotune, AutotuneConfig
from quack.cute_dsl_utils import torch2cute_dtype_map

from coda.core.ops import misc_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils
from coda.core.ops.constants import AUTOTUNE_CACHE_RESULTS


@dataclasses.dataclass(frozen=True)
class ElementwiseConfig(object):
    thr_m: int
    thr_n: int
    val_m: int


@cute.kernel
def elementwise_apply_kernel(
    fn: cutlass.Constexpr[Callable],
    mX: cute.Tensor,
    mY: cute.Tensor,
    mZ: cute.Tensor,
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    vector_size: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    idX = cute.make_identity_tensor(mX.shape)
    gX = cute.local_tile(mX, tiler_mn, (bidx, bidy))
    gY = cute.local_tile(mY, tiler_mn, (bidx, bidy))
    gZ = cute.local_tile(mZ, tiler_mn, (bidx, bidy))
    cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))
    config_X = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mX.element_type,
        num_bits_per_copy=mX.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    config_Y = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mY.element_type,
        num_bits_per_copy=mY.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    config_Z = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mZ.element_type,
        num_bits_per_copy=mZ.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    copy_outputs_X = memory_utils.copy(
        src=gX,
        dst="rmem",
        crd=cX,
        shape=mX.shape,
        config=config_X,
        thread_index=tidx,
        smem_allocator=allocator,
    )
    copy_outputs_Y = memory_utils.copy(
        src=gY,
        dst="rmem",
        crd=cX,
        shape=mY.shape,
        config=config_Y,
        thread_index=tidx,
        smem_allocator=allocator,
    )
    tXrX = copy_outputs_X.dst_thread
    tYrY = copy_outputs_Y.dst_thread
    tZrZ = creation_utils.allocate_tensor_like(
        tensor=tXrX,
        memspace="rmem",
        smem_allocator=allocator,
        dtype=mZ.element_type,
    )

    # apply custom function
    fn(tXrX, tYrY, tZrZ)

    # Copy the results back
    _ = memory_utils.copy(
        src=tZrZ,
        dst=gZ,
        crd=copy_outputs_X.crd_thread,
        shape=mZ.shape,
        config=config_Z,
        thread_index=tidx,
        smem_allocator=allocator,
    )


@cute.jit
def elementwise_apply(
    fn: cutlass.Constexpr[Callable],
    mX: cute.Tensor,
    mY: cute.Tensor,
    mZ: cute.Tensor,
    thr_m: cutlass.Constexpr[int],
    thr_n: cutlass.Constexpr[int],
    val_m: cutlass.Constexpr[int],
    stream: cuda.CUstream,
) -> int:
    vector_size = cutlass.const_expr(
        128 //
        cutlass.max(
            mX.element_type.width,
            mY.element_type.width,
            mZ.element_type.width,
        )
    )
    misc_utils.static_assert(len(mX.shape) == 2)
    misc_utils.static_assert(len(mY.shape) == 2)
    misc_utils.static_assert(len(mZ.shape) == 2)
    misc_utils.static_assert(mX.shape[1] == mY.shape[1])
    misc_utils.static_assert(mX.shape[1] == mZ.shape[1])
    misc_utils.static_assert(mX.shape[1] % vector_size == 0)
    tiler_mn, tv_layout = layout_utils.make_layout_tv_from_shape(
        thread_shape=(thr_m, thr_n),
        thread_order="row",
        value_shape=(val_m, vector_size),
        value_order="row",
    )

    # ((TileM, TileN), (RestM, RestN))
    gX = cute.zipped_divide(mX, tiler_mn)
    num_blocks = gX.shape[1]
    num_threads = cute.size(tv_layout, mode=[0])
    misc_utils.static_assert(len(num_blocks) == 2)
    kernel = elementwise_apply_kernel(
        fn,
        mX,
        mY,
        mZ,
        tiler_mn,
        tv_layout,
        vector_size,
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
def _compile_elementwise(
    op: Callable,
    size: int,
    x_dtype: type[cute.Numeric],
    y_dtype: type[cute.Numeric],
    z_dtype: type[cute.Numeric],
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> Callable:
    m = cute.sym_int()
    vector_size = cutlass.const_expr(
        128 //
        cutlass.max(
            x_dtype.width,
            y_dtype.width,
            z_dtype.width,
        )
    )
    mX = cute.runtime.make_fake_tensor(
        dtype=x_dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mY = cute.runtime.make_fake_tensor(
        dtype=y_dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mZ = cute.runtime.make_fake_tensor(
        dtype=z_dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        elementwise_apply,
        op,
        mX,
        mY,
        mZ,
        thr_m,
        thr_n,
        val_m,
        stream,
        options="--enable-tvm-ffi",
    )


def _elementwise_op(
    op: Callable,
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    assert X.shape == Y.shape == Z.shape
    size = X.shape[1]
    compiled_fn = _compile_elementwise(
        op=op,
        size=size,
        x_dtype=torch2cute_dtype_map[X.dtype],
        y_dtype=torch2cute_dtype_map[Y.dtype],
        z_dtype=torch2cute_dtype_map[Z.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    compiled_fn(X, Y, Z)


@autotune(
    configs=[
        AutotuneConfig(
            config=ElementwiseConfig(
                thr_m=thr_m,
                thr_n=thr_n,
                val_m=val_m,
            )
        )
        for thr_m, thr_n, val_m in (
            # 128 threads/block
            (4, 32, 4),
            (4, 32, 2),
            (2, 64, 2),
            # 256 threads/block
            (8, 32, 2),
            (4, 64, 2),
            (8, 32, 4),
            # 512 threads/block
            (8, 64, 2),
            (4, 128, 2),
            (16, 32, 2),
            # 1024 threads/block
            (8, 128, 1),
            (8, 128, 2),
        )
    ],
    key=["op"],
    # `Z` may be `X` or `Y` itself
    restore_value=("Z",),
    cache_results=AUTOTUNE_CACHE_RESULTS,
)
def _elementwise_op_tuned(
    op: Callable,
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor,
    config: ElementwiseConfig | None,
) -> None:
    if config is None:
        config = ElementwiseConfig(thr_m=4, thr_n=32, val_m=4)

    _elementwise_op(
        op=op,
        X=X,
        Y=Y,
        Z=Z,
        thr_m=config.thr_m,
        thr_n=config.thr_n,
        val_m=config.val_m,
    )
