import torch
from einops import rearrange

from autocast.encoders.base import EncoderWithCond
from autocast.types import Batch, TensorBNC


class PermuteConcat(EncoderWithCond):
    """Permute and concatenate Encoder.

    Concatenates channels and time dimensions into a single channel dimension.
    Output shape: (B, C*T, H, W) where C is in_channels and T is n_steps_input.

    `with_constants` folds in per-trajectory constants (static fields and
    scalars, broadcast over time). `with_forcing` folds in time-varying
    external drivers, aligned step-for-step with the input window. Both add to
    the channel count, so `in_channels` must already account for them --
    `setup_autoencoder_components` does this when resolving `auto`.
    """

    channel_axis: int = 1
    outputs_time_channel_concat: bool = True

    def __init__(
        self,
        in_channels: int,
        n_steps_input: int,
        with_constants: bool = False,
        with_forcing: bool = False,
    ) -> None:
        super().__init__()
        self.with_constants = with_constants
        self.with_forcing = with_forcing
        self.n_steps_input = n_steps_input
        self.latent_channels = in_channels * n_steps_input

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

        if self.with_forcing and batch.forcing_fields is not None:
            # `forcing_fields` spans the input *and* output window; take the
            # steps that line up with `input_fields` so it concatenates along
            # channels without changing the time axis.
            forcing = batch.forcing_fields[:, : self.n_steps_input]
            if forcing.shape[1] != t:
                msg = (
                    f"forcing_fields covers {batch.forcing_fields.shape[1]} steps, "
                    f"which cannot supply the {t} input steps this encoder "
                    "expects. It must span at least n_steps_input."
                )
                raise ValueError(msg)
            forcing = rearrange(forcing, "b t w h c -> b c t w h")
            x = torch.cat([x, forcing], dim=1)

        if self.with_constants and batch.constant_scalars is not None:
            scalars = batch.constant_scalars
            scalars = rearrange(scalars, "b c -> b c 1 1 1")
            scalars = scalars.expand(b, -1, t, w, h)
            x = torch.cat([x, scalars], dim=1)

        return rearrange(x, "b c t w h -> b (c t) w h")
