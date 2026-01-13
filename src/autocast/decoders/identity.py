from autocast.decoders.base import Decoder
from autocast.types.types import TensorBNC, TensorBTSC


class IdentityDecoder(Decoder):
    """Identity decoder that returns the input as output."""

    def __init__(self) -> None:
        super().__init__()

    def decode(self, z: TensorBNC) -> TensorBTSC:
        return z
