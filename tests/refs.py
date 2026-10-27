import math
import torch
from einops import rearrange, reduce, repeat


# ---- primitives


def swiglu(preact: torch.Tensor) -> torch.Tensor:
    # gate and up in adjacent columns (2i, 2i + 1), the pairing CODA's gated epilogue emits. In bf16, eager rounds
    # silu(gate) before the product, as a model's eager swiglu does, and a compiled one does not
    pairs = rearrange(preact, "m (n two) -> m n two", two=2)
    gate = pairs[..., 0]
    up = pairs[..., 1]
    silu = torch.nn.functional.silu(gate)
    return silu * up


def dswiglu(preact: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # swiglu's vjp in closed form, interleaved like preact, and the postact it recomputes on the way: closed so that
    # it compiles under fullgraph as torch's arm, and pinned to fp64 autograd in test_refs
    pairs = rearrange(preact, "m (n two) -> m n two", two=2)
    gate = pairs[..., 0]
    up = pairs[..., 1]
    sigmoid = torch.nn.functional.sigmoid(gate)
    silu = gate * sigmoid
    grad_silu = grad * up
    one_minus_sigmoid = 1 - sigmoid
    silu_one_minus_sigmoid = silu * one_minus_sigmoid
    dsilu = sigmoid + silu_one_minus_sigmoid
    grad_gate = grad_silu * dsilu
    grad_up = grad * silu
    grad_pairs = torch.stack((grad_gate, grad_up), dim=-1)
    grad_preact = rearrange(grad_pairs, "m n two -> m (n two)")
    postact = silu * up
    return grad_preact, postact
