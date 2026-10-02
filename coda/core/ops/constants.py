import os

# width of one vectorized global access; vector_size = NUM_BITS_PER_COPY // dtype.width
NUM_BITS_PER_COPY = 128

# fused backwards accumulate into the cotangent they are handed instead of cloning it
ALLOW_INPLACE_GRAD_OUTPUT = os.environ.get("CODA_ALLOW_INPLACE_GRAD_OUTPUT", "1") == "1"

# off by default: the autotune key carries no GPU identity
AUTOTUNE_CACHE_RESULTS = os.environ.get("CODA_AUTOTUNE_CACHE", "0") == "1"
