from collections.abc import Callable
from pathlib import Path

import torch
from einops import rearrange

from autocast.decoders.base import Decoder
from autocast.external.lola.lola_autoencoder import get_autoencoder
from autocast.types import TensorBTSC
from autocast.types.batch import Batch


class WrappedDecoder(Decoder):
    """Wrapper around Lola Encoder to match expected interface."""

    wrapped_autoencoder: torch.nn.Module
    wrapped_decode_func: Callable

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__()
        self.batch_size = kwargs.pop("batch_size", 16)
        self.mean = kwargs.pop("mean")
        self.std = kwargs.pop("std")
        runpath = kwargs.pop("runpath", None)
        self.wrapped_autoencoder = get_autoencoder(**kwargs)
        if runpath is not None:
            runpath = Path(runpath)
            state = torch.load(
                Path(runpath) / "state.pth",
                weights_only=True,
                map_location=device,
            )
            self.wrapped_autoencoder.load_state_dict(state)
            print(f"Loaded autoencoder weights from {runpath / 'state.pth'}")
        self.wrapped_decode_func = self.wrapped_autoencoder.decode

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b, t, *_ = z.shape
        z = rearrange(z, "B T ... C -> (B T) C ...")
        decoded = self.wrapped_decode_func(z)
        stacked = rearrange(decoded, "(B T) C ... -> B T ... C", B=b, T=t)
        stacked = self.postprocess(stacked)
        return stacked

    def postprocess(self, decoded: torch.Tensor) -> torch.Tensor:
        output = super().postprocess(decoded)
        return output * self.std + self.mean
