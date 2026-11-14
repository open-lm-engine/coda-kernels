import torch
import cutlass
import cutlass.cute as cute
from typing import cast, NamedTuple

from coda.core.ops import constants
from coda.core.ops import misc_utils
from coda.core.ops import dtype_utils
from coda.core.ops import layout_utils
from coda.core.ops import creation_utils


class MemoryCopyConfig(NamedTuple):
    op: str
    dtype: type[cute.Numeric]
    num_bits_per_copy: int
    tiler_mn: tuple[int, int]
    layout_tv: tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]


class MemoryCopyStruct(NamedTuple):
    copy_atom: cute.CopyAtom | None
    tiled_copy: cute.TiledCopy | None
    thread_copy: cute.ThrCopy | None
    src_block: cute.Tensor | None
    dst_block: cute.Tensor | None
    crd_block: cute.Tensor | None
    src_thread: cute.Tensor
    dst_thread: cute.Tensor
    crd_thread: cute.Tensor
    allocation: cute.Tensor | None
    predication: cute.Tensor | None


def prepare_copy(
    config: MemoryCopyConfig,
    thread_index: int | cute.Int32,
) -> tuple[cute.CopyAtom, cute.TiledCopy, cute.ThrCopy]:
    # Generate copy operation
    if config.op == "cp.async":
        copy_op = cute.nvgpu.cpasync.CopyG2SOp()
    elif config.op == "universal":
        copy_op = cute.nvgpu.CopyUniversalOp()
    else:
        raise NotImplementedError(f"Unsupported copy operation: {config.op}")

    # Generate copy atom
    copy_atom = cute.make_copy_atom(
        op=copy_op,
        copy_internal_type=config.dtype,
        num_bits_per_copy=config.num_bits_per_copy,
    )

    # Generate tiled copy
    tiled_copy = cute.make_tiled_copy(
        atom=copy_atom,
        layout_tv=config.layout_tv,
        tiler_mn=config.tiler_mn,
    )

    # Generate thread copy
    thread_copy = tiled_copy.get_slice(
        thr_idx=thread_index,
    )
    return copy_atom, tiled_copy, thread_copy


@cute.jit
def prepare_predicate_1D(
    src_thread: cute.Tensor,
    dst_thread: cute.Tensor,
    crd_thread: cute.Tensor,
    shape: cute.Shape,
) -> cute.Tensor:
    misc_utils.static_assert(src_thread.shape == dst_thread.shape == crd_thread.shape)
    misc_utils.static_assert(len(shape) == 1)
    misc_utils.static_assert(len(dst_thread.shape) == 2)
    misc_utils.static_assert(len(dst_thread.shape[0]) == 2)
    misc_utils.static_assert(len(dst_thread.shape[1]) == 1)

    # per CuTeDSL tutorial, predication is checked at the granularity of
    # a copy atom, so the predicate tensor does not need separate booleans
    # for individual elements within a copy atom (e.g., crd_thread.shape[0][0].)
    pred_thread = creation_utils.allocate_tensor_from_shape(
        shape=(
            crd_thread.shape[0][1],
            cute.size(crd_thread, mode=[1]),
        ),
        order="col",  # not sure if row or col
        memspace="rmem",
        smem_allocator=None,
        dtype=cute.Boolean,
    )
    for rest_v in cutlass.range_constexpr(pred_thread.shape[0]):
        for i in cutlass.range_constexpr(pred_thread.shape[1]):
            pred_thread[rest_v, i] = cute.elem_less(
                crd_thread[(0, rest_v), i],
                shape[0],
            )

    return pred_thread


@cute.jit
def prepare_predicate_2D(
    src_thread: cute.Tensor,
    dst_thread: cute.Tensor,
    crd_thread: cute.Tensor,
    shape: cute.Shape,
    dim: int | None,
) -> cute.Tensor:
    misc_utils.static_assert(src_thread.shape == dst_thread.shape == crd_thread.shape)
    misc_utils.static_assert(len(shape) == 2)
    misc_utils.static_assert(len(dst_thread.shape) == 3)
    misc_utils.static_assert(len(dst_thread.shape[0]) == 2)

    if cutlass.const_expr(dim == -1):
        # predication on the 2nd dimension
        pred_layout = cute.make_layout(
            (
                cute.size(crd_thread, mode=[0, 1]),
                cute.size(crd_thread, mode=[1]),
                cute.size(crd_thread, mode=[2]),
            ),
            stride=(
                cute.size(crd_thread, mode=[2]),
                0,
                1,
            ),
        )
        pred_thread = creation_utils.allocate_tensor_from_layout(
            pred_layout,
            memspace="rmem",
            smem_allocator=None,
            dtype=cute.Boolean,
        )
        for rest_v in cutlass.range_constexpr(pred_thread.shape[0]):
            for rest_dim in cutlass.range_constexpr(pred_thread.shape[2]):
                pred_thread[rest_v, 0, rest_dim] = cute.elem_less(
                    crd_thread[(0, rest_v), 0, rest_dim][1],
                    shape[1],
                )

    elif cutlass.const_expr(dim is None):
        pred_layout = cute.make_layout(
            (
                cute.size(crd_thread, mode=[0, 1]),
                cute.size(crd_thread, mode=[1]),
                cute.size(crd_thread, mode=[2]),
            ),
            stride=(
                cute.size(crd_thread, mode=[1]) * cute.size(crd_thread, mode=[2]),
                cute.size(crd_thread, mode=[2]),
                1,
            ),
        )
        pred_thread = creation_utils.allocate_tensor_from_layout(
            pred_layout,
            memspace="rmem",
            smem_allocator=None,
            dtype=cute.Boolean,
        )
        for rest_v in cutlass.range_constexpr(pred_thread.shape[0]):
            for rest_m in cutlass.range_constexpr(pred_thread.shape[1]):
                for rest_dim in cutlass.range_constexpr(pred_thread.shape[2]):
                    # `elem_less` compares the whole coordinate tuple
                    pred_thread[rest_v, rest_m, rest_dim] = cute.elem_less(
                        crd_thread[(0, rest_v), rest_m, rest_dim],
                        shape,
                    )

    else:
        raise NotImplementedError

    return pred_thread


@cute.jit
def copy(
    src: cute.Tensor,
    dst: cute.Tensor | str,
    crd: cute.Tensor,
    shape: cute.Shape,
    config: MemoryCopyConfig,
    thread_index: int | cute.Int32,
    smem_allocator: cutlass.utils.SmemAllocator,
    predication: cute.Tensor | None = None,
    filter_zeros: cutlass.Constexpr = False,
) -> MemoryCopyStruct:

    # we assume that `src, dst` are thread tensor if
    # register-backed but otherwise block tensor
    copy_atom, tiled_copy, thread_copy = prepare_copy(
        config=config,
        thread_index=thread_index,
    )

    if cutlass.const_expr(src.memspace.name != "rmem"):
        # register tensor is always thread view
        src_block = src
        crd_block = crd
        src_thread = thread_copy.partition_S(src)
        crd_thread = thread_copy.partition_S(crd)
    else:
        src_block = None
        crd_block = None
        src_thread = src
        crd_thread = crd

    if cutlass.const_expr(isinstance(dst, str)):
        dst = cast(str, dst)
        if cutlass.const_expr(dst != "rmem"):
            if cutlass.const_expr(src.memspace.name != "rmem"):
                # that is, `src` is a block Tensor
                dst = creation_utils.allocate_tensor_like(
                    tensor=src,
                    memspace=dst,
                    smem_allocator=smem_allocator,
                )
            else:
                raise NotImplementedError(f"`dst` needs to be a Tensor if `src` is an rmem Tensor")
        else:
            # again, register tensor is always thread view
            dst = creation_utils.allocate_tensor_like(
                tensor=src_thread,
                memspace=dst,
                smem_allocator=smem_allocator,
            )
        allocation = dst
    else:
        allocation = None

    dst = cast(cute.Tensor, dst)
    if cutlass.const_expr(dst.memspace.name != "rmem"):
        dst_block = dst
        dst_thread = thread_copy.partition_D(dst)
    else:
        dst_block = None
        dst_thread = dst

    assert cutlass.const_expr(cute.size(src_thread) == cute.size(dst_thread))
    assert cutlass.const_expr(src_thread.element_type == dst_thread.element_type)
    if cutlass.const_expr(src_block is not None and dst_block is not None):
        src_block = cast(cute.Tensor, src_block)
        dst_block = cast(cute.Tensor, dst_block)
        assert cutlass.const_expr(cute.size(src_block) == cute.size(dst_block))
        assert cutlass.const_expr(src_block.element_type == dst_block.element_type)

    if cutlass.const_expr(len(shape) == 1):
        if cutlass.const_expr(predication is not None):
            misc_utils.static_assert(predication.element_type == cute.Boolean)
            misc_utils.static_assert(
                predication.shape ==
                (
                    cute.size(crd_thread, mode=[0, 1]),
                    cute.size(crd_thread, mode=[1]),
                )
            )
            pred_thread = predication
        else:
            pred_thread = prepare_predicate_1D(
                src_thread=src_thread,
                dst_thread=dst_thread,
                crd_thread=crd_thread,
                shape=shape,
            )

    elif cutlass.const_expr(len(shape) == 2):
        if cutlass.const_expr(predication is not None):
            misc_utils.static_assert(predication.element_type == cute.Boolean)
            misc_utils.static_assert(
                predication.shape ==
                (
                    cute.size(crd_thread, mode=[0, 1]),
                    cute.size(crd_thread, mode=[1]),
                    cute.size(crd_thread, mode=[2]),
                )
            )
            pred_thread = predication
        else:
            # skipped elements are left unwritten, which only rmem and gmem tolerate: shared
            # staging is read by threads that cannot tell, and would need a fill first
            if cutlass.const_expr(dst.memspace in (cute.AddressSpace.rmem, cute.AddressSpace.gmem)):
                dim = None
            else:
                dim = -1
            pred_thread = prepare_predicate_2D(
                src_thread=src_thread,
                dst_thread=dst_thread,
                crd_thread=crd_thread,
                shape=shape,
                dim=dim,
            )

    else:
        misc_utils.static_assert(predication is None)
        # not implemented yet
        pred_thread = None

    if cutlass.const_expr(filter_zeros):
        cute.copy(
            atom=copy_atom,
            src=cute.filter_zeros(src_thread),
            dst=cute.filter_zeros(dst_thread),
            pred=pred_thread,
        )
    else:
        cute.copy(
            atom=copy_atom,
            src=src_thread,
            dst=dst_thread,
            pred=pred_thread,
        )

    return MemoryCopyStruct(
        copy_atom=copy_atom,
        tiled_copy=tiled_copy,
        thread_copy=thread_copy,
        src_block=src_block,
        dst_block=dst_block,
        crd_block=crd_block,
        src_thread=src_thread,
        dst_thread=dst_thread,
        crd_thread=crd_thread,
        allocation=allocation,
        predication=pred_thread,
    )


def simple_copy(
    src: cute.Tensor,
    dst: cute.Tensor | str,
    crd: cute.Tensor,
    smem_allocator: cutlass.utils.SmemAllocator,
    filter_zeros: cutlass.Constexpr = False,
) -> MemoryCopyStruct:
    # we assume that `src, dst` are always thread tensor
    if cutlass.const_expr(isinstance(dst, str)):
        dst = cast(str, dst)
        dst = creation_utils.allocate_tensor_like(
            tensor=src,
            memspace=dst,
            smem_allocator=smem_allocator,
        )
        allocation = dst
    else:
        allocation = None

    dst = cast(cute.Tensor, dst)
    assert cutlass.const_expr(cute.size(src) == cute.size(dst))
    if cutlass.const_expr(filter_zeros):
        cute.autovec_copy(
            src=cute.filter_zeros(src),
            dst=cute.filter_zeros(dst),
        )
    else:
        cute.autovec_copy(
            src=src,
            dst=dst,
        )

    return MemoryCopyStruct(
        copy_atom=None,
        tiled_copy=None,
        thread_copy=None,
        src_block=None,
        dst_block=None,
        crd_block=None,
        src_thread=src,
        dst_thread=dst,
        crd_thread=crd,
        allocation=allocation,
        predication=None,
    )


def direct_copy(
    op: cute.nvgpu.CopyG2ROp | cute.nvgpu.CopyR2GOp,
    src: cute.Tensor,
    dst: cute.Tensor,
    cache_policy: str | None,
) -> None:
    # copy all of `src` to `dst` in a single vectorized load or store
    dtype = misc_utils.get_dtype(src)
    copy_atom = cute.make_copy_atom(
        op=op,
        copy_internal_type=dtype,
        num_bits_per_copy=cute.size(src) * dtype.width,
    )
    if cutlass.const_expr(cache_policy is None):
        cute.copy(
            atom=copy_atom,
            src=src,
            dst=dst,
        )
    elif cutlass.const_expr(cache_policy == "evict_first"):
        cute.copy(
            atom=copy_atom,
            src=src,
            dst=dst,
            cache_policy=cute.Int64(constants.L2_EVICT_FIRST),
        )
    else:
        raise NotImplementedError(f"Unsupported cache policy: {cache_policy}")


def dereference_load(
    tensor: cute.Tensor,
    coord: cute.Coord | None = None,
    dtype: type[cute.Numeric] | None = None,
) -> cute.TensorSSA | cute.Numeric:
    misc_utils.static_assert(isinstance(tensor, cute.Tensor))
    if cutlass.const_expr(coord is None):
        value = tensor.load()
    else:
        value = tensor[coord]

    if cutlass.const_expr(dtype is not None):
        value = cast(cute.TensorSSA | cute.Numeric, value)
        dtype = cast(type[cute.Numeric], dtype)
        value = value.to(dtype=dtype)

    return value


def dereference_store(
    tensor: cute.Tensor,
    value: cute.TensorSSA | cute.Numeric | cutlass.cutlass_dsl.cutlass_arith.ArithValue,
    coord: cute.Coord | None = None,
    dtype: type[cute.Numeric] | None = None,
) -> None:
    misc_utils.static_assert(isinstance(tensor, cute.Tensor))
    if cutlass.const_expr(dtype is not None):
        dtype = cast(type[cute.Numeric], dtype)
        value = value.to(dtype=dtype)

    if cutlass.const_expr(coord is None):
        misc_utils.static_assert(isinstance(value, cute.TensorSSA))
        tensor.store(value)
    else:
        misc_utils.static_assert(isinstance(value, cute.TensorSSA | cute.Numeric | cutlass.cutlass_dsl.cutlass_arith.ArithValue))
        tensor[coord] = value


def make_tma_atoms_and_tensors(
    op: str,
    tensor: cute.Tensor,
    smem_layout_staged: cute.ComposedLayout,
    smem_tile: tuple[int, int],
    num_multicast: int = 1,
) -> tuple[cute.CopyAtom, cute.Tensor]:
    """Create TMA atoms and tensors for input tensors.

    :param tensor: Input tensor (A or B)
    :type tensor: cute.Tensor
    :param smem_layout_staged: Shared memory layout for the tensor
    :type smem_layout_staged: cute.ComposedLayout
    :param smem_tile: Shared memory tile shape
    :type smem_tile: Tuple[int, int]
    :param num_multicast: Multicast dimension
    :type num_multicast: int

    :return: TMA atom and tensor
    :rtype: Tuple[cute.CopyAtom, cute.Tensor]
    """
    if cutlass.const_expr(op == "g2s"):
        if cutlass.const_expr(num_multicast == 1):
            copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        else:
            copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
    elif cutlass.const_expr(op == "s2g"):
        copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
    elif cutlass.const_expr(op == "s2g-add"):
        copy_op = cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)
    else:
        raise NotImplementedError(f"Unsupported copy operation: {op}")

    smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
    tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        copy_op,
        tensor,
        smem_layout,
        smem_tile,
        num_multicast=num_multicast,
    )
    return tma_atom, tma_tensor


def g2s_copy_1d(
    src: cute.Tensor,
    dst: cute.Tensor,
    crd: cute.Tensor,
    shape: cute.Shape,
    num_threads: int,
    thread_index: cute.Int32,
) -> None:
    dtype = misc_utils.get_dtype(src)
    vector_size = cutlass.const_expr(
        cutlass.max(32, dtype.width) //
        dtype.width
    )
    num_bits_per_copy = (
        vector_size *
        dtype.width
    )
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(vector_size)
    tiler_mn, layout_tv = cute.make_layout_tv(
        thr_layout=thr_layout,
        val_layout=val_layout,
    )
    config = MemoryCopyConfig(
        op="cp.async",
        dtype=dtype,
        num_bits_per_copy=num_bits_per_copy,
        tiler_mn=tiler_mn,
        layout_tv=layout_tv,
    )

    copy(
        src=src,
        dst=dst,
        crd=crd,
        shape=shape,
        config=config,
        thread_index=thread_index,
        smem_allocator=None,
    )


def g2s_copy_2d_row_reduction(
    src: cute.Tensor,
    dst: cute.Tensor,
    crd: cute.Tensor,
    shape: cute.Shape,
    num_threads: int,
    thread_index: cute.Int32,
) -> None:
    dtype = misc_utils.get_dtype(src)
    num_rows, num_cols = src.shape
    num_threads_per_row = layout_utils.get_num_threads_per_row(
        size=num_cols,
    )
    num_threads_per_col = (
        num_threads //
        num_threads_per_row
    )
    num_threads = (
        num_threads_per_col,
        num_threads_per_row,
    )
    vector_size = cutlass.const_expr(
        num_cols //
        num_threads_per_row
    )
    vector_size = cutlass.const_expr(
        misc_utils.greatest_power_of_2_dividing(vector_size)
    )
    num_bits_per_copy = (
        vector_size *
        dtype.width
    )
    tiler_mn, layout_tv = layout_utils.make_2D_row_reduction_layout(
        size=num_cols,
        dtype=dtype,
        num_threads=num_threads,
        cluster_size=None,
        num_bits_per_copy=num_bits_per_copy,
    )
    misc_utils.static_assert(num_rows % tiler_mn[0] == 0)
    misc_utils.static_assert(num_cols % tiler_mn[1] == 0)
    config = MemoryCopyConfig(
        op="cp.async",
        dtype=dtype,
        num_bits_per_copy=num_bits_per_copy,
        tiler_mn=tiler_mn,
        layout_tv=layout_tv,
    )

    copy(
        src=src,
        dst=dst,
        crd=crd,
        shape=shape,
        config=config,
        thread_index=thread_index,
        smem_allocator=None,
    )


def s2r_copy_1d(
    src: cute.Tensor,
    dtype: type[cute.Numeric] | None = None,
) -> cute.Tensor:
    copy_outputs = simple_copy(
        src=src,
        dst="rmem",
        crd=None,
        smem_allocator=None,
        filter_zeros=True,
    )
    dst = copy_outputs.dst_thread
    if cutlass.const_expr(dtype is not None):
        dst = dtype_utils.convert(dst, dtype=dtype)
    return dst


def g2r_copy_1d(
    src: cute.Tensor,
    dst: cute.Tensor,
    cache_policy: str | None = None,
) -> None:
    # load `src` from global memory directly into the registers `dst`,
    # without going through shared memory
    # cache_policy="evict_first": the L2 cache evicts this data first
    direct_copy(
        op=cute.nvgpu.CopyG2ROp(),
        src=src,
        dst=dst,
        cache_policy=cache_policy,
    )


def r2g_copy_1d(
    src: cute.Tensor,
    dst: cute.Tensor,
    cache_policy: str | None = None,
) -> None:
    # store the registers `src` directly to `dst` in global memory,
    # without going through shared memory
    # cache_policy="evict_first": the L2 cache evicts this data first
    direct_copy(
        op=cute.nvgpu.CopyR2GOp(),
        src=src,
        dst=dst,
        cache_policy=cache_policy,
    )
