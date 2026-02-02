from autocast.encoders.base import Encoder
from autocast.types.batch import Batch
from autocast.types.types import TensorBNC


class IdentityEncoder(Encoder):
    """Identity encoder that passes through input unchanged."""

    channel_dim: int = -1

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.latent_dim = in_channels

    def encode(self, batch: Batch) -> TensorBNC:
        return batch.input_fields


class IdentityEncoderWithCond(IdentityEncoder):
    """Permute and concatenate Encoder."""
