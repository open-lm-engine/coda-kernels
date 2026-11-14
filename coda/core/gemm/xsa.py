import operator
import cutlass
import cutlass.cute as cute
from typing import NamedTuple

from quack.compile_utils import div_for_dtype, make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.epilogue.ops import EpiOp
from quack.gemm_runtime.host import FakeArgCtx
from quack.gemm_sm90 import GemmSm90

from coda.core.ops import constants
from coda.core.ops import misc_utils
from coda.core.gemm.epilogues import ConstInt


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
