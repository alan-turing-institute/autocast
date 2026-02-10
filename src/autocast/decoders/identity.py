from autocast.decoders.base import Decoder
from autocast.types.types import TensorBNC, TensorBTSC


class IdentityDecoder(Decoder):
    """Identity decoder that returns the input as output."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.latent_channels = in_channels

    def decode(self, z: TensorBNC) -> TensorBTSC:
        if self.latent_channels is None:
            self.latent_channels = z.shape[-1]
        return z
