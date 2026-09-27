import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from typing import Callable

from quack.activation import silu, dsilu
from quack.cache import jit_cache
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.tile_scheduler import RasterOrder

from coda.core.ops import misc_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils


def short_conv_fwd_(
    x: torch.Tensor,
    y: torch.Tensor,
    state: torch.Tensor | None,
    weight: torch.Tensor,
    activation: str | None,
    thr_m: int,
    thr_n: int,
    val_m: int,
    num_bits_per_copy: int,
    raster_order: RasterOrder,
) -> None:
    raise NotImplementedError


def short_conv_bwd_(
    dx: torch.Tensor,
    dy: torch.Tensor,
    dstate: torch.Tensor | None,
    dweight: torch.Tensor,
    x: torch.Tensor,
    state: torch.Tensor | None,
    weight: torch.Tensor,
    activation: str | None,
    thr_m: int,
    thr_n: int,
    val_m: int,
    num_bits_per_copy: int,
    raster_order: RasterOrder,
) -> None:
    raise NotImplementedError


def short_conv_dweight_(
    dweight: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    raise NotImplementedError
