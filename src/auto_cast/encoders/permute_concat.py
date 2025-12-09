import torch
from einops import rearrange

from auto_cast.encoders.base import Encoder
from auto_cast.types import Batch, Tensor, TensorBNC


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
            constants = batch.constant_fields
            constants = rearrange(constants, "b w h c -> b c 1 w h")
            x = torch.cat([x, constants], dim=1)
        if self.with_constants and batch.constant_scalars is not None:
            scalars = batch.constant_scalars
            scalars = rearrange(scalars, "b c -> b c 1 1 1")
            scalars = scalars.expand(b, -1, t, w, h)
            x = torch.cat([x, scalars], dim=1)
        return rearrange(x, "b c t w h -> b (c t) w h")

    def encode(self, batch: Batch) -> TensorBNC:
        return self.forward(batch)
