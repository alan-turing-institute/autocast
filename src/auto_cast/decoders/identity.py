from auto_cast.decoders.base import Decoder
from auto_cast.types.types import TensorBNC, TensorBTSC


class IdentityDecoder(Decoder):
    """Identity decoder that returns the input as output."""

    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: TensorBNC) -> TensorBTSC:
        return x

    def decode(self, z: TensorBNC) -> TensorBTSC:
        return self.forward(z)
