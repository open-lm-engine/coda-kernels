import cutlass
import cutlass.cute as cute
from typing import cast

from coda.core.ops.misc_utils import (
    product,
    get_dtype,
    static_assert,
)


def make_ordered_layout(
    shape: tuple | cute.Shape,
    order: tuple | str,
) -> cute.Layout:
    if cutlass.const_expr(isinstance(order, str)):
        if order == "row":
            order = tuple(range(len(shape)))[::-1]
        elif order == "col":
            order = tuple(range(len(shape)))
        else:
            raise ValueError
    if cutlass.const_expr(isinstance(order, list)):
        order = tuple(order)
    assert cutlass.const_expr(isinstance(order, tuple))
    return cute.make_ordered_layout(
        shape=shape,
        order=order,
    )


def make_layout_tv_from_shape(
    thread_shape: tuple[int, int],
    thread_order: tuple[int, int] | str,
    value_shape: tuple[int, int],
    value_order: tuple[int, int] | str,
) -> tuple[tuple[int, int], cute.Layout]:
    num_threads = thread_shape[0] * thread_shape[1]
    static_assert(len(thread_shape) == 2)
    static_assert(len(value_shape) == 2)
    static_assert(num_threads % 32 == 0)
    static_assert(num_threads <= 1024)
    thread_layout = make_ordered_layout(shape=thread_shape, order=thread_order)
    value_layout = make_ordered_layout(shape=value_shape, order=value_order)
    tiler_mn, layout_tv = cute.make_layout_tv(
        thr_layout=thread_layout,
        val_layout=value_layout,
    )
    tiler_mn = cast(tuple[int, int], tiler_mn)
    return tiler_mn, layout_tv


def get_num_threads(size: int) -> int:
    if cutlass.const_expr(size <= 16384):
        n = 128
    else:
        n = 256
    return n


def get_num_threads_per_row(size: int) -> int:

    if cutlass.const_expr(size <= 64):
        n = 8
    elif cutlass.const_expr(size <= 128):
        n = 16
    elif cutlass.const_expr(size <= 3072):
        n = 32
    elif cutlass.const_expr(size <= 6144):
        n = 64
    elif cutlass.const_expr(size <= 16384):
        n = 128
    else:
        n = 256
    return n


def get_cluster_size_per_row(size: int, dtype: cute.Numeric) -> int:

    if cutlass.const_expr(dtype.width == 16):
        # 16-bit types (fp16, bf16)
        if cutlass.const_expr(size <= 16 * 1024):
            n = 1
        elif cutlass.const_expr(size <= 32 * 1024):
            n = 2
        elif cutlass.const_expr(size <= 64 * 1024):
            n = 4
        elif cutlass.const_expr(size <= 128 * 1024):
            n = 8
        else:
            n = 16

    else:
        # 32-bit types (fp32)
        if cutlass.const_expr(size <= 32 * 1024):
            n = 1
        elif cutlass.const_expr(size <= 64 * 1024):
            n = 2
        elif cutlass.const_expr(size <= 128 * 1024):
            n = 4
        elif cutlass.const_expr(size <= 256 * 1024):
            n = 8
        else:
            n = 16

    return n


def make_2D_row_reduction_layout(
    size: int,
    dtype: type[cute.Numeric],
    num_threads: tuple[int, int] | None,
    cluster_size: int | None,
    num_bits_per_copy: int,
) -> tuple[tuple[int, int], cute.Layout]:

    if cutlass.const_expr(num_threads is None):
        _num_threads_per_row = get_num_threads_per_row(size)
        _num_threads_per_col = get_num_threads(size) // _num_threads_per_row
        num_threads = (_num_threads_per_col, _num_threads_per_row)

    if cutlass.const_expr(cluster_size is None):
        cluster_size = get_cluster_size_per_row(size=size, dtype=dtype)

    vector_size = num_bits_per_copy // dtype.width
    num_vectors = size // vector_size
    static_assert(size % vector_size == 0)
    static_assert(num_bits_per_copy % dtype.width == 0)
    static_assert(product(num_threads) % cute.arch.WARP_SIZE == 0)

    num_threads_per_col, num_threads_per_row = num_threads
    num_blocks_per_row = cute.ceil_div(num_vectors, num_threads_per_row * cluster_size)

    # Each tile has `[num_threads_per_col, size]` elements
    num_elements_per_col = num_threads_per_col
    num_elements_per_row = num_threads_per_row * vector_size * num_blocks_per_row
    num_elements_per_col_block = num_threads_per_col
    num_elements_per_row_block = num_threads_per_row * vector_size
    num_elements_per_block = num_elements_per_col_block * num_elements_per_row_block
    tiler_mn = cast(
        tuple[int, int],
        (num_elements_per_col, num_elements_per_row),
    )
    layout_tv_shape = (
        (num_threads_per_row, num_threads_per_col),
        (vector_size, num_blocks_per_row),
    )
    layout_tv_stride = (
        (num_threads_per_col * vector_size, 1),
        (num_elements_per_col_block, num_elements_per_block),
    )
    layout_tv = cute.make_layout(
        shape=layout_tv_shape,
        stride=layout_tv_stride,
    )
    return tiler_mn, layout_tv


def make_2D_elementwise_layout(
    dtype: type[cute.Numeric],
    num_bits_per_copy: int,
) -> tuple[tuple[int, int], cute.Layout]:
    vector_size = num_bits_per_copy // dtype.width
    thr_layout = make_ordered_layout((4, 32), order="row")
    val_layout = make_ordered_layout((4, vector_size), order="row")
    return cute.make_layout_tv(thr_layout, val_layout)


def unsqueeze(
    tensor: cute.Tensor,
    dim: cutlass.Constexpr,
    size: cutlass.Constexpr,
) -> cute.Tensor:
    if cutlass.const_expr(dim == 0):
        layout_extra = cute.make_layout((size,), stride=(0,))
        layout_unsqueezed = cute.prepend(tensor.layout, layout_extra)

    elif cutlass.const_expr(dim == -1):
        layout_extra = cute.make_layout((size,), stride=(0,))
        layout_unsqueezed = cute.append(tensor.layout, layout_extra)

    else:
        raise NotImplementedError

    return cute.make_tensor(
        iterator=tensor.iterator,
        layout=layout_unsqueezed,
    )


def assumed_align_stride(
    tensor: cute.Tensor,
    assumed_align: int = 4,
) -> cute.Tensor:
    # Assume all strides are divisible by 32 bits except the last stride
    assumed_align_vector_size = (
        assumed_align * 8 //
        tensor.element_type.width
    )
    new_stride = tuple(
        cute.assume(stride, divby=assumed_align_vector_size)
        if not cute.is_static(stride) else stride
        for stride in tensor.stride
    )
    new_layout = cute.make_layout(
        shape=tensor.shape,
        stride=new_stride,
    )
    return cute.make_tensor(
        iterator=tensor.iterator,
        layout=new_layout,
    )


def recast_tensor(tensor: cute.Tensor, dtype: type[cute.Numeric]) -> cute.Tensor:
    # `cute.recast_tensor` rounds lengths and strides that do not convert exactly, and drops the divisibility of
    # dynamic strides, which vector copies need to prove their alignment
    # https://github.com/Dao-AILab/quack/blob/v0.6.5/quack/gemm_base.py#L117
    # a boolean has a width of 1 bit but is stored in 8, so the widths below would be wrong
    static_assert(tensor.element_type is not cute.Boolean)
    static_assert(dtype is not cute.Boolean)
    src_width = tensor.element_type.width
    dst_width = dtype.width

    # refuse what would round: only the unit-stride dimension may convert its length instead of its stride
    for length, stride in zip(tensor.shape, tensor.stride):
        # a length-1 dimension never uses its stride
        if cutlass.const_expr(cute.is_static(length) and length == 1 and cute.is_static(stride)):
            continue
        length_num_bits = cute.get_divisibility(length) * src_width
        stride_num_bits = cute.get_divisibility(stride) * src_width
        if cutlass.const_expr(stride_num_bits % dst_width != 0):
            # a stride that does not convert exactly must be the unit stride
            static_assert(cute.is_static(stride) and stride == 1)
            # and its length must convert exactly instead
            static_assert(length_num_bits % dst_width == 0)

    tensor_recast = cute.recast_tensor(tensor, dtype=dtype)
    new_stride = []
    for stride, stride_recast in zip(tensor.stride, tensor_recast.stride):
        if cutlass.const_expr(cute.is_static(stride)):
            static_assert(cute.is_static(stride_recast))
            new_stride.append(stride_recast)
        else:
            divisibility = cute.get_divisibility(stride) * src_width // dst_width
            new_stride.append(cute.assume(stride_recast, divby=divisibility))

    new_layout = cute.make_layout(
        shape=tensor_recast.shape,
        stride=tuple(new_stride),
    )
    new_tensor = cute.make_tensor(
        iterator=tensor_recast.iterator,
        layout=new_layout,
    )
    # the pointer type is signless, so a rebuilt tensor cannot be e.g. Uint32
    static_assert(new_tensor.element_type is dtype)
    return new_tensor


def select_nonzero_stride_modes(
    tensor: cute.Tensor,
    layout_ref: cute.Layout,
) -> cute.Tensor:
    new_tensor = convert_layout_zero_stride(
        tensor=tensor,
        layout_ref=layout_ref,
    )
    return new_tensor[None, 0]


def convert_layout_zero_stride(
    tensor: cute.Tensor,
    layout_ref: cute.Layout,
) -> cute.Tensor:
    # Adopted from quack:
    # Reorganizes a tensor layout by separating its modes (dimensions) based on
    # whether they have zero or non-zero strides in a reference layout. This is
    # useful for separating broadcast dimensions from regular dimensions in tensor
    # operations, making it easier to reason about which dimensions are being
    # iterated over vs. which are being broadcast.
    static_assert(isinstance(tensor, cute.Tensor))
    layout = tensor.layout
    layout_flat = cute.flatten(layout)
    layout_flat_ref = cute.flatten(layout_ref)

    # Group the modes with non-zero stride in the ref_layout
    # together, and the modes with zero stride together
    layout_rank = cute.rank(layout_flat)
    nonzero_modes = [
        i for i
        in range(layout_rank)
        if layout_flat_ref[i].stride != 0
    ]
    zero_modes = [
        i for i
        in range(layout_rank)
        if layout_flat_ref[i].stride == 0
    ]

    # There's an edge case when all modes are zero stride
    new_shape = (
        tuple(layout_flat[i].shape for i in nonzero_modes) if len(nonzero_modes) > 0 else (1,),
        tuple(layout_flat[i].shape for i in zero_modes),
    )
    new_stride = (
        tuple(layout_flat[i].stride for i in nonzero_modes) if len(nonzero_modes) > 0 else (0,),
        tuple(layout_flat[i].stride for i in zero_modes),
    )
    new_layout = cute.make_layout(
        shape=new_shape,
        stride=new_stride,
    )
    return cute.make_tensor(
        iterator=tensor.iterator,
        layout=new_layout,
    )


@cute.jit
def permute_gated_Cregs_b16(tensor: cute.Tensor) -> None:
    # https://github.com/Dao-AILab/quack/blob/main/quack/layout_utils.py
    static_assert(get_dtype(tensor).width == 16)
    static_assert(cute.size(tensor.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b16 permutation")
    tensor_i32 = cute.recast_tensor(tensor, cute.Int32)

    quad_idx = cute.arch.lane_idx() % 4
    lane_03 = quad_idx == 0 or quad_idx == 3
    selector_upper = cute.Int32(0x5410) if lane_03 else cute.Int32(0x1054)
    selector_lower = cute.Int32(0x7632) if lane_03 else cute.Int32(0x3276)
    # upper_map = [0, 3, 1, 2]
    # lower_map = [1, 2, 0, 3]
    # upper_idx = upper_map[quad_idx]
    # indexing isn't supported so we have to do arithmetic
    upper_idx = quad_idx // 2 if quad_idx % 2 == 0 else 3 - quad_idx // 2
    lower_idx = upper_idx ^ 1

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range_constexpr(cute.size(tensor_i32.shape) // 2):
        upper, lower = tensor_i32[i * 2 + 0], tensor_i32[i * 2 + 1]
        upper0 = upper if lane_03 else lower
        lower0 = lower if lane_03 else upper
        upper0 = cute.arch.shuffle_sync(upper0, offset=upper_idx, mask_and_clamp=mask_and_clamp)
        lower0 = cute.arch.shuffle_sync(lower0, offset=lower_idx, mask_and_clamp=mask_and_clamp)
        tensor_i32[i * 2 + 0] = cute.arch.prmt(upper0, lower0, selector_upper)
        tensor_i32[i * 2 + 1] = cute.arch.prmt(upper0, lower0, selector_lower)
