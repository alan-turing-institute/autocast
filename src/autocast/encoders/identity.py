from autocast.encoders.base import Encoder
from autocast.types.batch import Batch
from autocast.types.types import Tensor, TensorBNC


class IdentityEncoder(Encoder):
    """Permute and concatenate Encoder."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Batch) -> Tensor:
        return batch.input_fields

    def encode(self, batch: Batch) -> TensorBNC:
        return self.forward(batch)
