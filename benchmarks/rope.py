import math
import torch
from einops import rearrange, repeat
from coda.core.elementwise import functional as elementwise_functional
from coda.core.gemm import functional as gemm_functional

from benchmarks.backends import torch_ops
from benchmarks.backends.torch_ops import PairLayout
from benchmarks.utils import data
from benchmarks.utils import operands
from benchmarks.utils.cases import Case, Implementation, make_oracle_fn
from benchmarks.utils.data import Sampler, Shape


OPS = {
    "gemm_rope": gemm_rope,
    "gemm_rmsnorm_rope": gemm_rmsnorm_rope,
    "rope_bwd_zdz": rope_bwd_zdz,
    "gemm_qknorm_rope": gemm_qknorm_rope,
    "gemm_rmsnorm_qknorm_rope": gemm_rmsnorm_qknorm_rope,
    "qknorm_rope_bwd": qknorm_rope_bwd,
    "gemm": gemm,
    "qknorm_rope_bwd_zdz": qknorm_rope_bwd_zdz,
}
