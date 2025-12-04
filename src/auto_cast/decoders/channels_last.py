from einops import rearrange

from auto_cast.decoders.base import Decoder
from auto_cast.types import Tensor


class ChannelsLast(Decoder):
    """Base Decoder."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the ChannelsLast decoder."""
        return rearrange(x, "b c t w h -> b t w h c")
