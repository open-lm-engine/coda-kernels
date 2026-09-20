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


@jit_cache
def _compile_rope_bwd_zdz(
    size: int,
    scale: float,
    dtype: type[cute.Numeric],
    pos_dtype: type[cute.Numeric],
    freq_dtype: type[cute.Numeric],
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> Callable:
    m = cute.sym_int()
    vector_size = cutlass.const_expr(constants.NUM_BITS_PER_COPY // dtype.width)
    mY = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mDY = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mDZ = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mZdZ = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(m,),
        stride=(1,),
        assumed_align=4,
    )
    mPos = cute.runtime.make_fake_tensor(
        dtype=pos_dtype,
        shape=(m,),
        stride=(1,),
        assumed_align=pos_dtype.width // 8,
    )
    mFreq = cute.runtime.make_fake_tensor(
        dtype=freq_dtype,
        shape=(size,),
        stride=(1,),
        assumed_align=freq_dtype.width // 8,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _rope_bwd_zdz,
        mY=mY,
        mDY=mDY,
        mDZ=mDZ,
        mZdZ=mZdZ,
        mPos=mPos,
        mFreq=mFreq,
        scale=scale,
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        stream=stream,
        options="--enable-tvm-ffi",
    )


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
