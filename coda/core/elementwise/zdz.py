import operator
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from typing import Callable

from quack.cache import jit_cache
from quack.cute_dsl_utils import torch2cute_dtype_map

from coda.core.ops import math_utils
from coda.core.ops import misc_utils
from coda.core.ops import constants
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils


def rope_bwd_zdz_(
    y: torch.Tensor,
    dy: torch.Tensor,
    dz: torch.Tensor,
    zdz: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    scale: float,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    fn = _compile_rope_bwd_zdz(
        size=y.shape[1],
        scale=scale,
        dtype=torch2cute_dtype_map[y.dtype],
        pos_dtype=torch2cute_dtype_map[pos.dtype],
        freq_dtype=torch2cute_dtype_map[freq.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    fn(
        y,
        dy,
        dz,
        zdz,
        pos,
        freq,
    )
