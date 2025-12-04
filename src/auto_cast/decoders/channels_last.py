from einops import rearrange

from auto_cast.decoders.base import Decoder
from auto_cast.types import Tensor, TensorBCTSPlus, TensorBTSPlusC


class ChannelsLast(Decoder):
    """Decoder that splits channels and time and reorders to channels-last format.

    The decoder splits (channel*time) back to (time, channel) and moves the channels
    to the last dimension assuming one or more spatial dimensions that are the last
    dimensions of the ecoded tensor.
    """

    def __init__(self, output_channels: int, time_steps: int = 1) -> None:
        """Initialize the ChannelsLast decoder.

        Parameters
        ----------
        output_channels: int
            Number of output channels (C).
        time_steps: int
            Number of time steps (T) that were merged with channels in encoding.
        """
        super().__init__()
        self.output_channels = output_channels
        self.time_steps = time_steps

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the ChannelsLast decoder.

        Expects input shape (B, C*T, spatial...) and outputs (B, T, spatial..., C).
        """
        x = rearrange(
            x, "b (c t) ... -> b c t ...", c=self.output_channels, t=self.time_steps
        )
        return rearrange(x, "b c t ... -> b t ... c")

    def decode(self, z: TensorBCTSPlus) -> TensorBTSPlusC:
        return self.forward(z)
