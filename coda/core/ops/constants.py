import os

# width of one vectorized global access; vector_size = NUM_BITS_PER_COPY // dtype.width
NUM_BITS_PER_COPY = 128

# bytes in one L2 cache line; one L2 slice owns each whole line
NUM_BYTES_PER_CACHE_LINE = 128

# fused backwards accumulate into the cotangent they are handed instead of cloning it
ALLOW_INPLACE_GRAD_OUTPUT = os.environ.get("CODA_ALLOW_INPLACE_GRAD_OUTPUT", "1") == "1"

# off by default: the autotune key carries no GPU identity
AUTOTUNE_CACHE_RESULTS = os.environ.get("CODA_AUTOTUNE_CACHE", "0") == "1"

# off by default: a generic GEMM, with no epilogue, is not CODA's, so it runs torch's own
USE_CODA_GEMM = os.environ.get("CODA_USE_CODA_GEMM", "0") == "1"

# off by default: qk-norm forwards write no head_mean_sq, and their backwards recompute it from preact
RETURN_HEAD_MEAN_SQ = os.environ.get("CODA_RETURN_HEAD_MEAN_SQ", "0") == "1"

# .L2::cache_hint policy words, CUTLASS CacheHintSm90 encodings
L2_EVICT_FIRST = 0x12F0000000000000
