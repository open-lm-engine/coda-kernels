# CODA: GPU Kernels as GEMM-plus-Epilogue Programs

<p align="center">
  <img src="figs/icon.jpg" width="350" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.19269"><img src="https://img.shields.io/badge/arXiv-2605.19269-b31b1b.svg" alt="arXiv"></a>
</p>

**CODA** is a GPU kernel abstraction that expresses Transformer operators as GEMM-plus-epilogue programs, fusing normalization, activations, residual updates, and reductions into the GEMM output tile before it is written to global memory, combining framework-level productivity with hardware-level efficiency. CODA is built on [CUTLASS CuTeDSL](https://github.com/NVIDIA/cutlass) and targets NVIDIA Hopper (H100) GPUs.

<p align="center">
  <img src="figs/reparameterization.png" width="700" />
</p>


## Updates
- September 28, 2026. Released `v0.3.0`.
- July 19, 2026. Released `v0.2`.
- June 23, 2026. We are restructuring CODA. For legacy version, please check `v0.1` tag.

## Installation

> **Note:** the `coda-kernels` package on PyPI is deprecated and no longer updated. Install from GitHub instead.

```bash
pip install git+https://github.com/open-lm-engine/coda-kernels.git@v0.3.0
```

Or from source:

```bash
git clone https://github.com/open-lm-engine/coda-kernels.git
cd coda-kernels
pip install -e .
```

CODA depends on a [fork of quack](https://github.com/HanGuo97/quack) pinned to a commit. If `quack-kernels` 0.6.5 is already installed from PyPI, pip keeps it because the versions match: run `pip uninstall quack-kernels` first, or install with `uv pip install`, which replaces it.


## Functional API

- [`linear_swiglu`](coda/kernels/functional/swiglu.py)
- [`linear_sigmoid`](coda/kernels/functional/sigmoid.py)
- [`linear_cross_entropy`](coda/kernels/functional/cross_entropy.py)
- [`linear_cross_entropy_forward`](coda/kernels/functional/cross_entropy.py)
- [`linear_qknorm_rope`](coda/kernels/functional/qknorm_rope.py)
