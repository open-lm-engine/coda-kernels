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
