from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
from einops import rearrange

from autocast.decoders.base import Decoder
from autocast.external.lola.field_transform import restore_fields
from autocast.external.lola.lola_autoencoder import get_autoencoder


class WrappedDecoder(Decoder):
    """Wrapper around Lola Encoder to match expected interface."""

    wrapped_autoencoder: torch.nn.Module
    wrapped_decode_func: Callable
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__()
        self.batch_size = kwargs.pop("batch_size", 16)
        self.chunk_size = kwargs.pop("chunk_size", None)
        self.log1p_indices = kwargs.pop("log1p_indices", ())
        noisy = kwargs.pop("noisy", True)
        self.latent_channels = kwargs["lat_channels"]
        self.register_buffer("mean", torch.as_tensor(kwargs.pop("mean")))
        self.register_buffer("std", torch.as_tensor(kwargs.pop("std")))
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
        decode = self.wrapped_autoencoder.decode
        self.wrapped_decode_func = decode if noisy else partial(decode, noisy=False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b, t, *_ = z.shape
        z = rearrange(z, "B T ... C -> (B T) C ...")
        decoded = self._chunked_apply(self.wrapped_decode_func, z)
        stacked = rearrange(decoded, "(B T) C ... -> B T ... C", B=b, T=t)
        stacked = self.postprocess(stacked)
        return stacked

    def postprocess(self, decoded: torch.Tensor) -> torch.Tensor:
        output = super().postprocess(decoded)
        return restore_fields(output, self.mean, self.std, self.log1p_indices)
