from collections.abc import Callable

import torch
from einops import rearrange

from autocast.decoders.base import Decoder
from autocast.encoders.base import Encoder
from autocast.external.lola.lola_autoencoder import get_autoencoder
from autocast.types.batch import Batch


class ChannelsFirstEncoder(Encoder):
    """Channels-First Encoder Wrapper for Lola AutoEncoder."""

    wrapped_encode_func: Callable

    def preprocess(self, batch: Batch) -> Batch:
        x = batch.input_fields
        x = rearrange(x, "B T ... C -> B C T ...")
        return Batch(
            input_fields=x,
            output_fields=batch.output_fields,
            constant_scalars=batch.constant_scalars,
            constant_fields=batch.constant_fields,
        )

    def encode(self, batch: Batch) -> torch.Tensor:
        batch = self.preprocess(batch)
        outputs = []
        for idx in range(batch.input_fields.shape[2]):  # loop over time dimension
            x = batch.input_fields[:, :, idx, ...].contiguous()
            x = self.wrapped_encode_func(x)
            outputs.append(x)

        # Stack outputs along time dimension
        stacked = torch.stack(outputs, dim=1)  # (B, T, C, spatial...)
        return rearrange(stacked, "B T C ... -> B T ... C")


class WrappedDecoder(Decoder):
    """Wrapper around Lola Encoder to match expected interface."""

    wrapped_autoencoder: torch.nn.Module
    wrapped_decode_func: Callable

    def __init__(self, **kwargs):
        super().__init__()
        self.wrapped_autoencoder = get_autoencoder(**kwargs)
        runpath = kwargs.get("runpath")
        if runpath is not None:
            state = torch.load(
                runpath / "state.pth",
                weights_only=True,
                map_location=kwargs.get("device"),
            )
            self.wrapped_autoencoder.load_state_dict(state)
        self.wrapped_decode_func = self.wrapped_autoencoder.decode
