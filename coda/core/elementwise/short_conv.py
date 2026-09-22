import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from typing import Callable

from quack.activation import silu, dsilu
from quack.cache import jit_cache
from quack.cute_dsl_utils import torch2cute_dtype_map

from coda.core.ops import constants
from coda.core.ops import misc_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils


def short_conv_fwd_(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    activation: str | None,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    width = weight.shape[1]
    assert x.shape[0] == y.shape[0] + width - 1
    fn = _compile_short_conv_fwd(
        size=x.shape[1],
        width=width,
        activation=activation,
        dtype=torch2cute_dtype_map[x.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    fn(x, y, weight)


def short_conv_bwd_(
    x: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    dweight: torch.Tensor,
    activation: str | None,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    width = weight.shape[1]
    fn = _compile_short_conv_bwd(
        size=x.shape[1],
        width=width,
        activation=activation,
        dtype=torch2cute_dtype_map[x.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    fn(x, dx, dy, weight, dweight)
