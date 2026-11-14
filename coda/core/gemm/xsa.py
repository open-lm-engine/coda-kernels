import operator
import cutlass
import cutlass.cute as cute
from typing import NamedTuple

from quack.cute_dsl_utils import mlir_namedtuple
from quack.gemm_sm90 import GemmSm90

from coda.core.ops import constants
from coda.core.ops import misc_utils
from coda.core.gemm.epilogues import ConstInt


class GemmXsaBwdSm90(GemmSm90):

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        x: cute.Tensor
        v: cute.Tensor
        dx: cute.Tensor
        dv: cute.Tensor
        head_dim: cutlass.Constexpr[int]
