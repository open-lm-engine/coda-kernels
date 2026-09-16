#!/usr/bin/env bash
set -ex

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"

NUM=30
if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: OUTPUT_DIR is not set"
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python -m kernels.benchmarks.block \
    --num "${NUM}" \
    --bench-rapier \
    --output "${OUTPUT_DIR}/block-rapier.pth"

python -m kernels.benchmarks.gemm \
    --num "${NUM}" \
    --output "${OUTPUT_DIR}/gemm.pth"

python -m kernels.benchmarks.block \
    --num "${NUM}" \
    --output "${OUTPUT_DIR}/block-other.pth"
