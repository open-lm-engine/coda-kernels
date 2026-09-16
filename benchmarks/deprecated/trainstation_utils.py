import os
import sys

TRAINSTATION_ROOT = "/workspace/main/trainstation"
assert os.path.isdir(TRAINSTATION_ROOT)
if TRAINSTATION_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(TRAINSTATION_ROOT))
from src.ops.gemm_interface import (
    gemm_rms,
    gemm_lse,
    gemm_rstd_norm_fwd,
    gemm_rms_bwd,
    gemm_dgated_zdz,
)
from src.ops.fused_blocks import (
    fused_transformer_block_func,
    gemm_residual_rmsnorm_gemm_fwd,
)
from src.ops.rope import rope_bwd_zdz
