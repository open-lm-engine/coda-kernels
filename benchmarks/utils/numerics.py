import torch
from typing import NamedTuple
from collections import defaultdict

# each dtype's significand bits after the leading one
SIGNIFICAND_BITS = {
    torch.bfloat16: 7,
    torch.float32: 23,
}
# each dtype's smallest normal exponent
SMALLEST_EXPONENT = {
    torch.bfloat16: -126,
    torch.float32: -126,
}
