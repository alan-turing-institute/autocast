from collections.abc import Callable
from pathlib import Path

import torch
from einops import rearrange

from autocast.encoders.base import EncoderWithCond
from autocast.external.lola.lola_autoencoder import get_autoencoder
from autocast.types.batch import Batch


class ChannelsFirstEncoder(EncoderWithCond):
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
        b, _, t, *_ = batch.input_fields.shape
        x = rearrange(batch.input_fields, "B C T ... -> (B T) C ...")
        x = self.wrapped_encode_func(x)
        return rearrange(x, "(B T) C ... -> B T ... C", B=b, T=t)


class WrappedEncoder(ChannelsFirstEncoder):
    """Wrapper around Lola Encoder to match expected interface."""

    wrapped_autoencoder: torch.nn.Module
    wrapped_encode_func: Callable

    def __init__(self, **kwargs):
        super().__init__()
        self.mean = kwargs.pop("mean", None)
        self.std = kwargs.pop("std", None)
        device = kwargs.pop("device", None)
        runpath = kwargs.pop("runpath", None)
        self.wrapped_autoencoder = get_autoencoder(**kwargs)
        if runpath is not None:
            runpath = Path(runpath)
            state = torch.load(
                runpath / "state.pth",
                weights_only=True,
                map_location=device,
            )
            self.wrapped_autoencoder.load_state_dict(state)
            print(f"Loaded autoencoder weights from {runpath / 'state.pth'}")
        self.wrapped_encode_func = self.wrapped_autoencoder.encode

    def preprocess(self, batch: Batch) -> Batch:
        batch = Batch(
            input_fields=(batch.input_fields - self.mean) / self.std,
            output_fields=batch.output_fields,
            constant_scalars=batch.constant_scalars,
            constant_fields=batch.constant_fields,
        )

        return ChannelsFirstEncoder.preprocess(self, batch)
