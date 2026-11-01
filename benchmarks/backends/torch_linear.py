import torch

from benchmarks.backends import torch_ops


def linear_sigmoid(
    x: torch.Tensor,
    weight: torch.Tensor,
    use_quack: bool = False,
) -> torch.Tensor:
    # sigmoid(x @ weight^T), weight (N, K)
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
    # swiglu(x @ weight^T), weight (2N, K) with gate and up rows
    # interleaved, or all gates then all ups in the split layout
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
    # weight rows: q's, then k's, then v's. q and k get qk-norm and rope,
    # v only the GEMM. In the split layout, each head's q and k rows and
    # gamma come permuted so element j pairs with j + head_dim / 2
    size_q = head_dim * num_heads_q
    size_k = head_dim * num_heads_k
    size_qk = size_q + size_k
    size_v = weight.shape[0] - size_qk
    weight_qk, weight_v = torch.split(weight, (size_qk, size_v))
    rotated_qk, _, _ = torch_ops.gemm_qknorm_rope(
        A=x,
        B=weight_qk.mT,
        weight=gamma,
        positions=positions,
        frequencies=frequencies,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        eps=eps,
        layout=layout,
        cos=cos,
        sin=sin,
        use_quack=use_quack,
    )
    q, k = torch.split(rotated_qk, (size_q, size_k), dim=-1)
    v = torch_ops.gemm(
        A=x,
        B=weight_v.mT,
        use_quack=use_quack,
    )
    return q, k, v


def linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    logits_dtype: torch.dtype,
    ignore_index: int = -100,
    reduction: str = "mean",
    use_quack: bool = False,
) -> torch.Tensor:
    # cross entropy of x @ weight^T in logits_dtype, weight (vocab, dim)
    logits = torch_ops.gemm(
        A=x,
        B=weight.mT,
        use_quack=use_quack,
    )
    return torch_ops.cross_entropy_loss(
        logits=logits,
        target=target,
        ignore_index=ignore_index,
        reduction=reduction,
        logits_dtype=logits_dtype,
    )
