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

        misc_utils.static_assert(vector_size == 8)
        misc_utils.static_assert(params.v.dtype == params.x.dtype)
        misc_utils.static_assert(params.dx.dtype == params.x.dtype)
        misc_utils.static_assert(params.dv.dtype == params.x.dtype)
