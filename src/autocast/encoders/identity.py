from autocast.encoders.base import EncoderWithCond
from autocast.types.batch import (
    Batch,
    EncodedBatch,
    EncodedSphericalBatch,
    SphericalBatch,
)
from autocast.types.types import TensorBNC


class IdentityEncoder(EncoderWithCond):
    """Identity encoder that passes through input unchanged.

    The encoded fields stay on the grid of the batch, so a SphericalBatch gives an
    EncodedSphericalBatch on the same grid.
    """

    channel_axis: int = -1

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.latent_channels = in_channels

    def encode(self, batch: Batch) -> TensorBNC:
        return batch.input_fields

    def encode_batch(
        self, batch: Batch, encoded_info: dict | None = None
    ) -> EncodedBatch:
        """Encode a full Batch, keeping the grid of a SphericalBatch.

        Args:
            batch: Input batch to be encoded.
            encoded_info: Optional dictionary of additional encoded information.

        Returns:
            An EncodedSphericalBatch with the grid of a SphericalBatch, otherwise
            the EncodedBatch of `EncoderWithCond.encode_batch`.
        """
        encoded = super().encode_batch(batch, encoded_info)
        if not isinstance(batch, SphericalBatch):
            return encoded
        return EncodedSphericalBatch(
            encoded_inputs=encoded.encoded_inputs,
            encoded_output_fields=encoded.encoded_output_fields,
            global_cond=encoded.global_cond,
            encoded_info=encoded.encoded_info,
            grid_type=batch.grid_type,
        )
