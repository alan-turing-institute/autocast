from autocast.encoders.base import EncoderWithCond
from autocast.types.batch import Batch
from autocast.types.types import TensorBNC


class IdentityEncoder(EncoderWithCond):
    """Identity encoder that passes through input unchanged."""

    channel_axis: int = -1

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.latent_channels = in_channels

    def encode(self, batch: Batch) -> TensorBNC:
        return batch.input_fields
