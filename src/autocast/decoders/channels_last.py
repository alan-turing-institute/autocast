from einops import rearrange

from autocast.decoders.base import Decoder
from autocast.types import TensorBCTS, TensorBTSC


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

    def decode(self, z: TensorBCTS) -> TensorBTSC:
        """Decode the latent tensor by rearranging channels and time.

        Expects input shape (B, C*T, spatial...) and outputs (B, T, spatial..., C).
        """
        c = self.output_channels
        t = self.time_steps
        z = rearrange(z, "b (c t) ... -> b c t ...", c=c, t=t)
        return rearrange(z, "b c t ... -> b t ... c")
