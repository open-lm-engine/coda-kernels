import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.elementwise.functional import short_conv_bwd, short_conv_fwd


class ShortConv(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        activation: str | None,
        initial_state: torch.Tensor | None,
    ) -> torch.Tensor:
        y = short_conv_fwd(
            x=x,
            weight=weight,
            activation=activation,
            initial_state=initial_state,
        )
        ctx.activation = activation
        ctx.save_for_backward(
            x,
            weight,
            initial_state,
        )
        return y

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
        x, weight, initial_state = ctx.saved_tensors
        dx, dweight, dinitial_state = short_conv_bwd(
            dy=dy,
            x=x,
            weight=weight,
            activation=ctx.activation,
            initial_state=initial_state,
        )
        return dx, dweight, None, dinitial_state


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
