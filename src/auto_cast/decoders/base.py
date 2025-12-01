from typing import Any

from torch import Tensor, nn


class Decoder(nn.Module):
    """Base Decoder."""

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
