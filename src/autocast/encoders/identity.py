from autocast.encoders.base import Encoder
from autocast.types.batch import Batch
from autocast.types.types import Tensor, TensorBNC


class IdentityEncoder(Encoder):
    """Identity Encoder that passes through input fields unchanged."""

    def __init__(
        self,
        with_constant_fields: bool = False,
        with_constant_scalars: bool = False,
    ) -> None:
        super().__init__()
        self.with_constant_fields = with_constant_fields
        self.with_constant_scalars = with_constant_scalars

    def forward(self, batch: Batch) -> Tensor:
        return batch.input_fields

    def encode(self, batch: Batch) -> TensorBNC:
        return self.forward(batch)
