import torch
from typing import NamedTuple
from collections import defaultdict


class Precision(NamedTuple):
    # e.g., {"coda": {"out": float}, "torch/eager": {"out": float}}
    errors: dict[str, dict[str, float]]
    # the error of the answer stored in CODA's dtype, e.g., {"out": float}
    errors_minimum: dict[str, float]


def relative_l2_error(
    value: torch.Tensor,
    answer: torch.Tensor,
) -> float:
    # the error's size over the answer's
    assert answer.dtype == torch.float64
    difference = value.to(dtype=torch.float64) - answer
    error = torch.linalg.vector_norm(difference)
    size = torch.linalg.vector_norm(answer)
    return (error / size).item()
