from collections.abc import Callable
from pathlib import Path

import torch
from einops import rearrange
from tqdm.auto import trange

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

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__()
        self.batch_size = kwargs.pop("batch_size", 16)
        runpath = kwargs.pop("runpath", None)
        self.wrapped_autoencoder = get_autoencoder(**kwargs)
        if runpath is not None:
            print(f"Loading AutoEncoder weights from {runpath}")
            state = torch.load(
                Path(runpath) / "state.pth",
                weights_only=True,
                map_location=device,
            )
            self.wrapped_autoencoder.load_state_dict(state)
        self.wrapped_decode_func = self.wrapped_autoencoder.decode

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b, t, *_ = z.shape
        z = rearrange(z, "B T ... C -> (B T) C ...")
        outputs = []
        for i in trange(0, z.shape[0], self.batch_size):
            z_batch = z[i : i + self.batch_size]
            decoded_batch = self.wrapped_decode_func(z_batch)
            outputs.append(decoded_batch)
        decoded = torch.cat(outputs, dim=0)
        stacked = rearrange(decoded, "(B T) C ... -> B T ... C", B=b, T=t)
        return stacked
