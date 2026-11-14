import operator
import cutlass
import cutlass.cute as cute
from typing import NamedTuple

from quack import layout_utils as quack_layout_utils
from quack.compile_utils import div_for_dtype, make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.epilogue.ops import EpiOp
from quack.gemm_runtime.host import FakeArgCtx
from quack.gemm_sm90 import GemmSm90

from coda.core.ops import constants
from coda.core.ops import misc_utils
from coda.core.ops import memory_utils
from coda.core.gemm.epilogues import ConstInt

# For HOPPER, we assume the warp layout is (8, 4): 8 lanes along M, 4 along N
# https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#wgmma-64n16-d
HOPPER_LANES_IN_N = 4


class Tile(EpiOp):
    # an (M, N) N-major operand the hook loads or stores
    # in global memory directly: no smem, no TMA
    def host_fake_arg(
        self,
        key: tuple[type[cute.Numeric], int],
        fctx: FakeArgCtx,
    ) -> cute.Tensor:
        # a plain 2-D GEMM only: no batch
        # dimension, no swapped operands
        assert not fctx.batched
        assert not fctx.swapped
        dtype, _ = key
        return make_fake_tensor(
            dtype=dtype,
            shape=(fctx.m, fctx.n),
            divisibility=div_for_dtype(dtype),
            leading_dim=1,
        )


@cute.jit
def _tile_chunks(
    tensor: cute.Tensor,
    tile_shape_mn: cutlass.Constexpr[tuple[int, int]],
    tile_coord_mn: cute.Coord,
) -> cute.Tensor:
    vector_size = cutlass.const_expr(constants.NUM_BITS_PER_COPY // tensor.dtype.width)
    tile = cute.local_tile(tensor, tile_shape_mn, tile_coord_mn)
    # cut each row into chunks of vector_size columns
    # (1, VectorSize, TileM, TileN / VectorSize)
    tile_chunks = cute.flat_divide(tile, (1, vector_size))
    # index a row's chunks by (lane, k): the k-th chunk that lane holds
    # (1, VectorSize, TileM, (LanesInN, ChunksPerLane))
    lane_chunks = cute.logical_divide(tile_chunks, (None, None, None, HOPPER_LANES_IN_N))
    # drop the leading 1 from the shape
    # (VectorSize, TileM, (LanesInN, ChunksPerLane))
    return lane_chunks[0, None, None, None]


@cute.jit
def _quad_transpose(
    rG: cute.Tensor,
    lane_in_n: cute.Int32,
) -> None:
    # Hopper's tensor cores spread a row over 4 lanes: each lane holds 2 of
    # every 8 columns (a chunk). Hence the lanes swap values until each
    # holds whole chunks. For every 4 chunks of the row:
    #
    #             before                       after
    #   lane 0    columns 0-1 of each chunk    all 8 columns of chunk 0
    #   lane 1    columns 2-3 of each chunk    all 8 columns of chunk 1
    #   lane 2    columns 4-5 of each chunk    all 8 columns of chunk 2
    #   lane 3    columns 6-7 of each chunk    all 8 columns of chunk 3
    misc_utils.static_assert(rG.shape[0] == (2, 4))
    for k in cutlass.range_constexpr(cute.size(rG, mode=[1])):
        # two rounds of swaps: first with the lane 2 away, then with the neighbour
        for offset in (2, 1):
            keep_upper = (lane_in_n & offset) != 0
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr((j & offset) == 0):
                    for e in cutlass.range_constexpr(2):
                        lower = rG[(e, j), k]
                        upper = rG[(e, j + offset), k]
                        sent = cutlass.select_(keep_upper, lower, upper)
                        received = cute.arch.shuffle_sync_bfly(sent, offset=offset)
                        rG[(e, j), k] = cutlass.select_(keep_upper, received, lower)
                        rG[(e, j + offset), k] = cutlass.select_(keep_upper, upper, received)


class GemmXsaBwdSm90(GemmSm90):
    _epi_ops = (
        Tile("x"),
        Tile("v"),
        Tile("dx"),
        Tile("dv"),
        ConstInt("head_dim"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        x: cute.Tensor
        v: cute.Tensor
        dx: cute.Tensor
        dv: cute.Tensor
        head_dim: cutlass.Constexpr[int]

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # the split below is for two thread groups running the multiply
        # and plain loads of A; quack uses other splits otherwise
        assert not self.gather_A
        assert self.mma_warp_groups == 2
        # the threads that run the matrix multiply also hold the hook's
        # working rows of x, v, dx and dv, so give them more registers
        # than quack's default split does (40 for the loading threads,
        # 232 for the rest). 24 and 240 is the split quack itself uses
        # for register-heavy kernels; here it is 0.5 to 1% faster
        self.num_regs_load = 24
        self.num_regs_mma = 240

    def epi_to_underlying_arguments(self, args: EpilogueArguments, *, loc=None, ip=None) -> EpilogueArguments:
        return args

    @cute.jit
    def epi_visit_acc(
        self,
        params: EpilogueArguments,
        acc: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tile_coord_mnkl: cute.Coord,
        tidx: cute.Int32,
    ) -> None:
        tile_M, tile_N, _ = self.cta_tile_shape_mnk
        head_dim = params.head_dim
        # x, v, dx and dv move 16 bytes at a time: 8 bf16 values
        vector_size = cutlass.const_expr(constants.NUM_BITS_PER_COPY // params.x.dtype.width)
        # the tensor cores leave each row spread over HOPPER_LANES_IN_N
        # neighbouring threads (lanes): of every 8 columns (a chunk), each
        # lane holds 2. The swap below trades them for whole chunks, so a
        # lane ends up with one chunk in every HOPPER_LANES_IN_N of the row
        chunks_per_lane = cutlass.const_expr(tile_N // vector_size // HOPPER_LANES_IN_N)
        head_chunks_per_lane = cutlass.const_expr(head_dim // vector_size // HOPPER_LANES_IN_N)
        heads_per_tile = cutlass.const_expr(tile_N // head_dim)
        # this thread's lane along N
        lane_in_n = tidx % HOPPER_LANES_IN_N
        tile_shape_mn = (tile_M, tile_N)
        tile_coord_mn = tile_coord_mnkl[:2]
        # skip a tile that starts past M. A (2, 1) cluster takes tiles in
        # pairs along M, so an odd number of tiles gets one extra
        num_rows = params.x.shape[0]
        tile_row_start = tile_coord_mnkl[0] * tile_M
        tile_in_bound = tile_row_start < num_rows

        misc_utils.static_assert(vector_size == 8)
        misc_utils.static_assert(params.v.dtype == params.x.dtype)
        misc_utils.static_assert(params.dx.dtype == params.x.dtype)
        misc_utils.static_assert(params.dv.dtype == params.x.dtype)
        # each head's dot products are formed within one tile,
        # so a tile must hold whole heads
        misc_utils.static_assert(tile_N % head_dim == 0)
        # a head must split into whole chunks, the same number on every lane
        misc_utils.static_assert(head_dim % (vector_size * HOPPER_LANES_IN_N) == 0)
        # the swap below rewrites the accumulator itself,
        # so there must be no output D to store from it
        misc_utils.static_assert(self.d_dtype is None)

        # this thread holds a few rows of G = dy W and some columns of each.
        # tCcC[i, j] is the (row, column) in the tile of the j-th value in
        # its i-th row, and rG_mn[i, j] is the value itself
        idC = cute.make_identity_tensor(tile_shape_mn)
        tCcC = quack_layout_utils.reshape_acc_to_mn(tiled_mma.get_slice(tidx).partition_C(idC))
        rG_mn = quack_layout_utils.reshape_acc_to_mn(acc)

        gX = _tile_chunks(params.x, tile_shape_mn=tile_shape_mn, tile_coord_mn=tile_coord_mn)
        gV = _tile_chunks(params.v, tile_shape_mn=tile_shape_mn, tile_coord_mn=tile_coord_mn)
        gDX = _tile_chunks(params.dx, tile_shape_mn=tile_shape_mn, tile_coord_mn=tile_coord_mn)
        gDV = _tile_chunks(params.dv, tile_shape_mn=tile_shape_mn, tile_coord_mn=tile_coord_mn)

        if tile_in_bound:
            for row_index in cutlass.range_constexpr(cute.size(rG_mn, mode=[0])):
                row_coord, _ = tCcC[row_index, 0]
                # rows past M are not loaded or stored. They are still computed,
                # as the warp shuffles and reductions need every lane
                row_in_bound = (tile_row_start + row_coord) < num_rows
                # this row of G, split into groups of vector_size values
                # ((2, LanesInN), ChunksPerLane)
                rG = cute.flat_divide(rG_mn[row_index, None], (vector_size,))
                # Hopper's tensor cores spread a row of G over 4 lanes: each lane
                # holds 2 of every 8 columns. x and v are loaded 8 neighbouring
                # columns at a time, hence the lanes swap G so that each holds
                # the same columns of G, x and v
                _quad_transpose(rG, lane_in_n=lane_in_n)

                rX = cute.make_rmem_tensor(layout_or_shape=(vector_size, chunks_per_lane), dtype=params.x.dtype)
                rV = cute.make_rmem_tensor(layout_or_shape=(vector_size, chunks_per_lane), dtype=params.v.dtype)
                rDX = cute.make_rmem_tensor(layout_or_shape=(vector_size, chunks_per_lane), dtype=params.dx.dtype)
                rDV = cute.make_rmem_tensor(layout_or_shape=(vector_size, chunks_per_lane), dtype=params.dv.dtype)

                if row_in_bound:
                    for k in cutlass.range_constexpr(chunks_per_lane):
                        # x, v, dx and dv pass through L2 once: evict them
                        # first, so the dy panels stay resident
                        memory_utils.g2r_copy_1d(
                            src=gX[None, row_coord, (lane_in_n, k)],
                            dst=rX[None, k],
                            cache_policy="evict_first",
                        )
                        memory_utils.g2r_copy_1d(
                            src=gV[None, row_coord, (lane_in_n, k)],
                            dst=rV[None, k],
                            cache_policy="evict_first",
                        )
                else:
                    # a row past M computes from zeros and stores nothing
                    rX.fill(0.0)
                    rV.fill(0.0)

                for head in cutlass.range_constexpr(heads_per_tile):
                    g_dot_v = cute.Float32.zero
                    x_dot_v = cute.Float32.zero
                    v_dot_v = cute.Float32.zero
                    for head_chunk in cutlass.range_constexpr(head_chunks_per_lane):
                        k = head * head_chunks_per_lane + head_chunk
                        for col_index in cutlass.range_constexpr(vector_size):
                            v = rV[col_index, k].to(dtype=cute.Float32)
                            g_dot_v = g_dot_v + rG[col_index, k].to(dtype=cute.Float32) * v
                            x_dot_v = x_dot_v + rX[col_index, k].to(dtype=cute.Float32) * v
                            v_dot_v = v_dot_v + v * v

                    g_dot_v = cute.arch.warp_reduction(g_dot_v, op=operator.add, threads_in_group=HOPPER_LANES_IN_N)
                    x_dot_v = cute.arch.warp_reduction(x_dot_v, op=operator.add, threads_in_group=HOPPER_LANES_IN_N)
                    v_dot_v = cute.arch.warp_reduction(v_dot_v, op=operator.add, threads_in_group=HOPPER_LANES_IN_N)

                    # no epsilon, as in the forward
                    inv_v_dot_v = cute.arch.rcp_approx(v_dot_v)
                    s = x_dot_v * inv_v_dot_v
                    a = g_dot_v * inv_v_dot_v
                    two_a_s = cute.Float32(2.0) * a * s

                    for head_chunk in cutlass.range_constexpr(head_chunks_per_lane):
                        k = head * head_chunks_per_lane + head_chunk
                        for col_index in cutlass.range_constexpr(vector_size):
                            g = rG[col_index, k].to(dtype=cute.Float32)
                            x = rX[col_index, k].to(dtype=cute.Float32)
                            v = rV[col_index, k].to(dtype=cute.Float32)
                            rDV[col_index, k] = (two_a_s * v - (s * g + a * x)).to(dtype=rDV.dtype)
                            rDX[col_index, k] = (g - a * v).to(dtype=rDX.dtype)
                        if row_in_bound:
                            memory_utils.r2g_copy_1d(
                                src=rDX[None, k],
                                dst=gDX[None, row_coord, (lane_in_n, k)],
                                cache_policy="evict_first",
                            )
                            memory_utils.r2g_copy_1d(
                                src=rDV[None, k],
                                dst=gDV[None, row_coord, (lane_in_n, k)],
                                cache_policy="evict_first",
                            )
