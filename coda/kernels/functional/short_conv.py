import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.elementwise.functional import short_conv_bwd, short_conv_fwd


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation: str | None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    return ShortConv.apply(
        x,
        weight,
        activation,
        initial_state,
    )
