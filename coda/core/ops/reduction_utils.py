import math
import warnings
import operator
import cutlass
import cutlass.cute as cute
import torch.utils._pytree as pytree
from typing import NamedTuple, cast
from collections.abc import Callable, Sequence

from coda.core.ops.misc_utils import Numeric, static_assert, get_dtype, numel
from coda.core.ops.dtype_utils import f32x2_to_i64, i64_to_f32x2
from coda.core.ops.creation_utils import allocate_tensor_from_shape


@cute.jit
def warp_reduce(
    xs: Sequence[Numeric] | cute.TensorSSA,
    op: Callable[[Sequence[Numeric], Sequence[Numeric]], Sequence[Numeric]] | str,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> Sequence[Numeric] | cute.TensorSSA:

    if cutlass.const_expr(isinstance(xs, cute.TensorSSA)):
        ys = cute.make_rmem_tensor(xs.shape, xs.dtype)
        ys.store(xs)
        for k in cutlass.range_constexpr(cute.size(xs.shape)):
            (ys[k],) = warp_reduce((ys[k],), op=op, width=width)
        return ys.load()

    else:
        if cutlass.const_expr(isinstance(op, str)):
            dtype = get_dtype(xs[0])
            static_assert(pytree.tree_all(lambda x: get_dtype(x) == dtype, xs))
            op = _create_simple_block_reduction_op(name=op, element_type=dtype).combine_fn

        static_assert(pytree.tree_all(lambda x: isinstance(x, Numeric), xs))
        return pytree.tree_map(
            lambda x: cute.arch.warp_reduction(
                x,
                op=op,
                threads_in_group=width,
            ),
            xs,
        )


@cute.jit
def warp_to_block_reduction_exchange(
    xs: Sequence[Numeric],
    reduction_buffer: cute.Tensor,
    reduction_buffer_shape: tuple[int, int, int],
    init_value_fn: Callable[[Sequence[Numeric]], Sequence[Numeric]],
) -> Sequence[Numeric]:
    static_assert(pytree.tree_all(lambda x: isinstance(x, Numeric), xs))
    static_assert(reduction_buffer_shape[0] == len(xs))

    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    row_idx = warp_idx // reduction_buffer_shape[2]
    col_idx = warp_idx % reduction_buffer_shape[2]
    num_elements = cutlass.const_expr(len(xs))

    if lane_idx == 0:
        for element_index in cutlass.range_constexpr(num_elements):
            reduction_buffer[element_index, row_idx, col_idx] = xs[element_index]
    cute.arch.barrier()

    # block reduction on smem, this strange-looking block of code is to get around
    # certain corner cases when manipulating python structure inside branching
    ys = list(init_value_fn(xs))
    for element_index in cutlass.range_constexpr(num_elements):
        y = ys[element_index]
        if lane_idx < reduction_buffer_shape[2]:
            y = reduction_buffer[element_index, row_idx, lane_idx]
        ys[element_index] = y

    return ys


class BlockReductionOp(NamedTuple):
    """Defines a reduction operation for hierarchical value combination.
    
    This class encapsulates the logic for performing reductions across different
    computational levels in a GPU block:
    - Thread-level: Within individual threads using loops or SSA operations
    - Warp-level: Across threads within the same warp
    - Block-level: Across warps within the same block

    Attributes:
        combine_fn: Binary function that combines two PyTree structures of values.
                   Used for warp-level and block-level reductions.
        reduce_ssa: Optional specialized function for thread-level reduction of
                   TensorSSA values. Supports only ADD, MUL, MAX, MIN operations
                   on TensorSSA objects with optional postprocessing. When None,
                   thread-level reduction falls back to iterative combine_fn calls.
        reduce_wrp: Optional specialized function for warp-level reduction of
                   thread-reduced values (Sequence[Numeric]). Provides custom
                   optimized implementations for specific reduction operations
                   (e.g., using warp shuffle intrinsics). When None, warp-level
                   reduction falls back to generic warp_reduce using combine_fn.
        init_value: Initial value for the reduction operation.
    """
    combine_fn: Callable[[Sequence[Numeric], Sequence[Numeric]], Sequence[Numeric]]
    reduce_ssa: Callable[[Sequence[cute.TensorSSA]], Sequence[Numeric]] | None
    reduce_wrp: Callable[[Sequence[Numeric]], Sequence[Numeric]] | None
    init_value: Callable[[Sequence[cute.TensorSSA | Numeric]], Sequence[Numeric]] | Sequence[Numeric] | Numeric

    def get_init_value(
        self,
        xs: Sequence[cute.TensorSSA | Numeric],
    ) -> Sequence[Numeric]:
        if cutlass.const_expr(callable(self.init_value)):
            init_value_fn = cast(Callable[[Sequence[cute.TensorSSA | Numeric]], Sequence[Numeric]], self.init_value)
            ys = init_value_fn(xs)
        elif cutlass.const_expr(isinstance(self.init_value, tuple | list)):
            ys = cast(Sequence[Numeric], self.init_value)
        elif cutlass.const_expr(isinstance(self.init_value, int | float | Numeric)):
            ys = pytree.tree_map(lambda _: self.init_value, xs)
        else:
            raise NotImplementedError
        static_assert(len(xs) == len(ys))
        return ys

    @cute.jit
    def thread_reduction(
        self,
        xs: Sequence[cute.TensorSSA],
    ) -> Sequence[Numeric]:
        if cutlass.const_expr(self.reduce_ssa is not None):
            ys = self.reduce_ssa(xs)
        else:
            static_assert(pytree.tree_all(lambda x: numel(x) == numel(xs[0]), xs))
            ys = self.get_init_value(xs)
            for index in cutlass.range_constexpr(numel(xs[0])):
                hs = pytree.tree_map(lambda x: x[index], xs)
                ys = self.combine_fn(ys, hs)
        return ys

    @cute.jit
    def warp_reduction(
        self,
        xs: Sequence[Numeric],
        width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
    ) -> Sequence[Numeric]:
        static_assert(pytree.tree_all(lambda x: isinstance(x, Numeric), xs))
        if cutlass.const_expr(self.reduce_wrp is not None):
            # reduce_wrp takes no width: it always reduces the whole warp
            static_assert(width == cute.arch.WARP_SIZE)
            xs = self.reduce_wrp(xs)
        else:
            xs = warp_reduce(xs=xs, width=width, op=self.combine_fn)
        return xs

    @cute.jit
    def block_reduction(
        self,
        xs: Sequence[Numeric],
        reduction_buffer: cute.Tensor,
        reduction_buffer_shape: tuple[int, int, int],
    ) -> Sequence[Numeric]:
        ys = warp_to_block_reduction_exchange(
            xs=xs,
            reduction_buffer=reduction_buffer,
            reduction_buffer_shape=reduction_buffer_shape,
            init_value_fn=self.get_init_value,
        )
        ys = self.warp_reduction(ys)
        return type(xs)(ys)

    def combine_fn_singleton(
        self,
        x0: Numeric,
        x1: Numeric,
    ) -> Numeric:
        (y,) = self.combine_fn((x0,), (x1,))
        return y

    def get_init_value_singleton(
        self,
        x: cute.TensorSSA | Numeric,
    ) -> Numeric:
        (y,) = self.get_init_value(xs=(x,))
        return y

    @cute.jit
    def thread_reduction_singleton(
        self,
        x: cute.TensorSSA,
    ) -> Numeric:
        (y,) = self.thread_reduction(xs=(x,))
        return y

    @cute.jit
    def warp_reduction_singleton(
        self,
        x: Numeric,
        width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
    ) -> Numeric:
        (x,) = self.warp_reduction(
            xs=(x,),
            width=width,
        )
        return x

    @cute.jit
    def block_reduction_singleton(
        self,
        x: Numeric,
        reduction_buffer: cute.Tensor,
        reduction_buffer_shape: tuple[int, int, int],
    ) -> Numeric:
        (y,) = self.block_reduction(
            xs=(x,),
            reduction_buffer=reduction_buffer,
            reduction_buffer_shape=reduction_buffer_shape,
        )
        return y


def prepare_simple_block_reduction_op(
    name: str,
    element_type: type[cute.Numeric],
) -> tuple[cute.ReductionOp, Callable, cute.Numeric]:
    if name == "add":
        ssa_op = cute.ReductionOp.ADD
        wrp_op = operator.add
        init_value = element_type(0.)
    elif name == "mul":
        ssa_op = cute.ReductionOp.MUL
        wrp_op = operator.mul
        init_value = element_type(1.)
    elif name == "max":
        ssa_op = cute.ReductionOp.MAX
        if cutlass.const_expr(element_type == cute.Float32):
            wrp_op = cute.arch.fmax
        else:
            wrp_op = max
        init_value = element_type(float("-inf"))
    elif name == "min":
        ssa_op = cute.ReductionOp.MIN
        wrp_op = min
        init_value = element_type(float("inf"))
    else:
        raise NotImplementedError

    return ssa_op, wrp_op, init_value


def _create_simple_block_reduction_op(
    name: str,
    element_type: type[cute.Numeric],
) -> BlockReductionOp:
    """A collection of block reduction ops that reduces each tensor using simple ops"""
    ssa_op, wrp_op, init_value = prepare_simple_block_reduction_op(
        name=name,
        element_type=element_type,
    )

    def _combine_fn(tensor_x: cute.TensorSSA | Numeric, tensor_y: cute.TensorSSA | Numeric) -> Numeric:
        return wrp_op(tensor_x, tensor_y)

    def _reduce_ssa(tensor: cute.TensorSSA) -> Numeric:
        return tensor.reduce(
            op=ssa_op,
            init_val=init_value,
            reduction_profile=0,
        )

    return BlockReductionOp(
        combine_fn=lambda tree_x, tree_y: pytree.tree_map(_combine_fn, tree_x, tree_y),
        reduce_ssa=lambda tree: pytree.tree_map(_reduce_ssa, tree),
        reduce_wrp=None,
        init_value=init_value,
    )


_REDUCTION_OP_REGISTRY: dict[str, BlockReductionOp] = {}


def register_reduction_op(name: str, op: BlockReductionOp, overwrite: bool = False) -> None:
    """Register a BlockReductionOp with a string name.
    
    Args:
        name: String identifier for the reduction operation
        op: BlockReductionOp instance to register
        
    Raises:
        ValueError: If name is already registered
    """
    if name in _REDUCTION_OP_REGISTRY:
        if not overwrite:
            raise ValueError(f"Reduction operation '{name}' is already registered")
        else:
            warnings.warn(f"Overwriting reduction operation '{name}'")
    _REDUCTION_OP_REGISTRY[name] = op


def get_registered_reduction_op(name: str, element_type: type[cute.Numeric]) -> BlockReductionOp:
    """Get a registered reduction operation by name.
    
    Args:
        name: String identifier for the reduction operation
        element_type: Element type for built-in operations (fallback only)
        
    Returns:
        BlockReductionOp instance
        
    Raises:
        KeyError: If name is not registered
    """
    if name in _REDUCTION_OP_REGISTRY:
        return _REDUCTION_OP_REGISTRY[name]

    if name in {"add", "mul", "max", "min"}:
        return _create_simple_block_reduction_op(
            name=name,
            element_type=element_type,
        )

    raise KeyError(f"Reduction operation '{name}' is not registered. Available: {list(_REDUCTION_OP_REGISTRY.keys())}")


def reduce(
    xs: Sequence[cute.TensorSSA],
    op: BlockReductionOp | str,
    thread_shape: tuple[int, int],
    smem_allocator: cutlass.utils.SmemAllocator,
    reduction_buffer: cute.Tensor | None,
) -> tuple[Sequence[Numeric], cute.Tensor | None]:

    static_assert(isinstance(xs, tuple | list))
    static_assert(pytree.tree_all(lambda x: isinstance(x, cute.TensorSSA), xs))
    static_assert(pytree.tree_all(lambda x: x.element_type == xs[0].element_type, xs))

    if cutlass.const_expr(isinstance(op, BlockReductionOp)):
        resolved_op = op
    elif cutlass.const_expr(isinstance(op, str)):
        resolved_op = get_registered_reduction_op(
            name=op,
            element_type=xs[0].element_type,
        )
    else:
        available_ops = list(_REDUCTION_OP_REGISTRY.keys()) + ["add", "mul", "max", "min"]
        raise TypeError(
            f"Invalid reduction operation: {op!r} (type: {type(op).__name__}). "
            f"Expected either:\n"
            f"  - BlockReductionOp instance, or\n"
            f"  - str with one of the available operations: {sorted(set(available_ops))}\n"
            f"Use register_reduction_op() to register custom operations."
        )

    # thread reduction
    outputs = resolved_op.thread_reduction(xs)

    # warp reduction
    outputs = resolved_op.warp_reduction(
        outputs,
        width=min(thread_shape[1], cute.arch.WARP_SIZE),
    )

    # block reduction
    num_elements = len(outputs)
    num_warps = cute.size(thread_shape) // cute.arch.WARP_SIZE
    num_warps_per_row = max(thread_shape[1] // cute.arch.WARP_SIZE, 1)
    num_warps_per_col = num_warps // num_warps_per_row
    reduction_buffer_shape = (num_elements, num_warps_per_col, num_warps_per_row)
    # a warp never splits a row: a row spans whole warps, or a warp holds whole rows,
    # and the block is whole warps, so the divisions above are exact
    static_assert(
        (thread_shape[1] % cute.arch.WARP_SIZE == 0) or
        (cute.arch.WARP_SIZE % thread_shape[1] == 0)
    )
    static_assert(cute.size(thread_shape) % cute.arch.WARP_SIZE == 0)
    static_assert(num_warps_per_row <= cute.arch.WARP_SIZE)
    static_assert(pytree.tree_all(lambda x: x.element_type == xs[0].element_type, xs))

    reduction_buffer_created = None
    if cutlass.const_expr(num_warps_per_row > 1):
        if cutlass.const_expr(reduction_buffer is not None):
            reduction_buffer = cast(cute.Tensor, reduction_buffer)
            static_assert(reduction_buffer.shape == reduction_buffer_shape)
            static_assert(reduction_buffer.element_type == xs[0].element_type)
            static_assert(reduction_buffer.memspace == cute.AddressSpace.smem)
            # a reused buffer may still be read by warps finishing the previous reduction
            cute.arch.barrier()
        else:
            reduction_buffer_created = allocate_tensor_from_shape(
                shape=reduction_buffer_shape,
                order="row",
                dtype=xs[0].element_type,
                memspace="smem",
                smem_allocator=smem_allocator,
                byte_alignment=4,
            )
            reduction_buffer = reduction_buffer_created

        outputs = resolved_op.block_reduction(
            outputs,
            reduction_buffer=reduction_buffer,
            reduction_buffer_shape=reduction_buffer_shape,
        )

    return outputs, reduction_buffer_created


def online_softmax_reduce(
    x: cute.TensorSSA,
    thread_shape: tuple[int, int],
    smem_allocator: cutlass.utils.SmemAllocator,
    reduction_buffer: cute.Tensor | None,
) -> tuple[cute.TensorSSA, Numeric, Numeric]:
    max_op = get_registered_reduction_op(
        name="max",
        element_type=x.element_type,
    )
    add_op = get_registered_reduction_op(
        name="add",
        element_type=x.element_type,
    )

    (max_x,) = max_op.thread_reduction((x,))
    (max_x,) = max_op.warp_reduction(
        (max_x,),
        width=min(thread_shape[1], cute.arch.WARP_SIZE),
    )

    exp_x = cute.math.exp(x - max_x, fastmath=True)
    (sum_exp_x,) = add_op.thread_reduction((exp_x,))
    (sum_exp_x,) = add_op.warp_reduction(
        (sum_exp_x,),
        width=min(thread_shape[1], cute.arch.WARP_SIZE),
    )

    # block reduction
    # this function assumes that all values owned by a warp are in the same row.
    # 1. each warp is in one row
    # 2. each thread is in one row as well
    num_elements = 1
    num_warps = cute.size(thread_shape) // cute.arch.WARP_SIZE
    num_warps_per_row = max(thread_shape[1] // cute.arch.WARP_SIZE, 1)
    num_warps_per_col = num_warps // num_warps_per_row
    reduction_buffer_shape = (num_elements, num_warps_per_col, num_warps_per_row)
    static_assert(num_warps_per_row <= cute.arch.WARP_SIZE)

    reduction_buffer_created = None
    if cutlass.const_expr(num_warps_per_row > 1):
        if cutlass.const_expr(reduction_buffer is not None):
            reduction_buffer = cast(cute.Tensor, reduction_buffer)
            static_assert(reduction_buffer.shape == reduction_buffer_shape)
            static_assert(reduction_buffer.element_type == cute.Int64)
            static_assert(reduction_buffer.memspace == cute.AddressSpace.smem)
        else:
            reduction_buffer_created = allocate_tensor_from_shape(
                shape=reduction_buffer_shape,
                order="row",
                dtype=cute.Int64,
                memspace="smem",
                smem_allocator=smem_allocator,
                byte_alignment=4,
            )
            reduction_buffer = reduction_buffer_created

        def _exchange_init_fn(_x: tuple[Numeric]) -> tuple[Numeric]:
            _x0 = f32x2_to_i64(
                cute.Float32(float("-inf")),
                cute.Float32(0.),
            )
            return (_x0,)

        # we reduce a tuple of Float32 as Int64
        static_assert(get_dtype(x) == cute.Float32)
        static_assert(get_dtype(max_x) == cute.Float32)
        static_assert(get_dtype(sum_exp_x) == cute.Float32)

        max_x_and_sum_exp_x = f32x2_to_i64(max_x, sum_exp_x)
        (max_x_wrp_and_sum_exp_x,) = warp_to_block_reduction_exchange(
            (max_x_and_sum_exp_x,),
            reduction_buffer=reduction_buffer,
            reduction_buffer_shape=reduction_buffer_shape,
            init_value_fn=_exchange_init_fn,
        )
        max_x_wrp, sum_exp_x = i64_to_f32x2(max_x_wrp_and_sum_exp_x)
        max_x_blk = max_op.warp_reduction(max_x_wrp)
        sum_exp_x = sum_exp_x * cute.math.exp(max_x_wrp - max_x_blk, fastmath=True)
        sum_exp_x = add_op.warp_reduction(sum_exp_x)
        exp_x = exp_x * cute.math.exp(max_x - max_x_blk, fastmath=True)
        max_x = max_x_blk

    return exp_x, max_x, sum_exp_x


def online_softmax_combine_singleton(
    m0: cute.Numeric,
    m1: cute.Numeric,
    s0: cute.Numeric,
    s1: cute.Numeric,
) -> tuple[cute.Numeric, cute.Numeric]:
    static_assert(get_dtype(m0) is cute.Float32)
    static_assert(get_dtype(m1) is cute.Float32)
    static_assert(get_dtype(s0) is cute.Float32)
    static_assert(get_dtype(s1) is cute.Float32)
    max_op = get_registered_reduction_op(
        name="max",
        element_type=cute.Float32,
    )
    add_op = get_registered_reduction_op(
        name="add",
        element_type=cute.Float32,
    )
    m = max_op.combine_fn_singleton(m0, m1)
    s0_new = s0 * cute.math.exp(m0 - m, fastmath=True)
    s1_new = s1 * cute.math.exp(m1 - m, fastmath=True)
    s = add_op.combine_fn_singleton(s0_new, s1_new)
    return m, s


def online_softmax_combine_warp(
    m: cute.Numeric,
    s: cute.Numeric,
    width: cutlass.Constexpr[int],
) -> tuple[cute.Numeric, cute.Numeric]:
    static_assert(get_dtype(m) is cute.Float32)
    static_assert(get_dtype(s) is cute.Float32)
    max_op = get_registered_reduction_op(
        name="max",
        element_type=cute.Float32,
    )
    add_op = get_registered_reduction_op(
        name="add",
        element_type=cute.Float32,
    )

    m_new = max_op.warp_reduction_singleton(
        m,
        width=width,
    )
    s_new = s * cute.math.exp(m - m_new, fastmath=True)
    s_new = add_op.warp_reduction_singleton(
        s_new,
        width=width,
    )
    return m_new, s_new
