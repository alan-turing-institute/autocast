import torch
from einops import rearrange

from autocast.encoders.base import EncoderWithCond
from autocast.types import Batch, TensorBNC


class PermuteConcat(EncoderWithCond):
    """Permute and concatenate Encoder.

    Concatenates channels and time dimensions. The actual latent_dim is
    in_channels * n_timesteps, but we store the base in_channels.
    """

    channel_dim: int = 1
    outputs_time_channel_concat: bool = True

    def __init__(self, in_channels: int, with_constants: bool = False) -> None:
        super().__init__()
        self.with_constants = with_constants
        self.latent_dim = in_channels  # Base channels before time concatenation

    def encode(self, batch: Batch) -> TensorBNC:
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
