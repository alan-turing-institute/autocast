import torch
from einops import rearrange

from autocast.encoders.base import Encoder
from autocast.types import Batch, Tensor, TensorBNC


class PermuteConcat(Encoder):
    """Permute and concatenate Encoder."""

    def __init__(self, with_constants: bool = False) -> None:
        super().__init__()
        self.with_constants = with_constants

    def forward(self, batch: Batch) -> Tensor:
        # Destructure batch, time, space, channels
        b, t, w, h, _ = batch.input_fields.shape  # TODO: generalize beyond 2D spatial
        x = batch.input_fields
        x = rearrange(x, "b t w h c -> b c t w h")

        if self.with_constants and batch.constant_fields is not None:
            constants_fields = batch.constant_fields  # (b, w, h, c_const)
            constants_fields = rearrange(constants_fields, "b w h c -> b c 1 w h")
            constants_fields = constants_fields.expand(b, -1, t, w, h)
            x = torch.cat([x, constants_fields], dim=1)

        if self.with_constants and batch.constant_scalars is not None:
            scalars = batch.constant_scalars
            scalars = rearrange(scalars, "b c -> b c 1 1 1")
            scalars = scalars.expand(b, -1, t, w, h)
            x = torch.cat([x, scalars], dim=1)

        return rearrange(x, "b c t w h -> b (c t) w h")

    def encode(self, batch: Batch) -> TensorBNC:
        return self.forward(batch)
