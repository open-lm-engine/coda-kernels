import torch

from benchmarks.backends import torch_ops


def linear_sigmoid(
    x: torch.Tensor,
    weight: torch.Tensor,
    use_quack: bool = False,
) -> torch.Tensor:
    # sigmoid(x @ weight^T), weight (N, K) as in
    # nn.Linear; its autograd is the backward's reference
    return torch_ops.gemm_sigmoid(
        A=x,
        B=weight.mT,
        use_quack=use_quack,
    )


def linear_swiglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    layout: torch_ops.PairLayout,
    use_quack: bool = False,
) -> torch.Tensor:
    # weight (2N, K), gate and up rows interleaved or
    # split; its autograd is the backward's reference
    _, postact = torch_ops.gemm_swiglu(
        A=x,
        B=weight.mT,
        layout=layout,
        use_quack=use_quack,
    )
    return postact


def linear_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    positions: torch.Tensor | None,
    frequencies: torch.Tensor | None,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    eps: float,
    layout: torch_ops.PairLayout,
    use_quack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return q, k, v


def linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    use_quack: bool = False,
) -> torch.Tensor:
    # weight (V, K) as in nn.Linear; the loss over the logits widened
    # to fp32, and its autograd is the backward's reference
    logits = torch_ops.gemm(
        A=x,
        B=weight.mT,
        use_quack=use_quack,
    )
