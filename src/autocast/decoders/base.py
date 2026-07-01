from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import nn

from autocast.types import Tensor, TensorBNC, TensorBTSC


class Decoder(nn.Module, ABC):
    """Base Decoder."""

    latent_channels: int

    # Optional batch-chunking knob for the heavy forward pass. See
    # ``autocast.encoders.base.GenericEncoder.chunk_size`` for the mirror.
    chunk_size: int | None = None

    def _chunked_apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """Apply ``fn`` to ``x`` in chunks along the leading batch dim.

        Numerically identical to ``fn(x)`` when ``chunk_size`` is unset or the
        input already fits. Concrete decoders call this around the flattened-
        batch compute to cap activation memory on large-resolution rollouts.
        """
        chunk_size = self.chunk_size
        if chunk_size is None or chunk_size <= 0 or x.shape[0] <= chunk_size:
            return fn(x)
        return torch.cat(
            [fn(x[i : i + chunk_size]) for i in range(0, x.shape[0], chunk_size)],
            dim=0,
        )

    def postprocess(self, decoded: Tensor) -> TensorBTSC:
        """Optionally transform the decoded tensor before returning.

        Subclasses can override to implement post-decoding steps. Default is
        identity.
        """
        return decoded

    @abstractmethod
    def decode(self, z: TensorBNC) -> TensorBTSC:
        """Decode the latent tensor back to the original space.

        Args:
            z: Latent tensor to be decoded.

        Returns:
            Decoded tensor in the original space.
        """

    def forward(self, z: TensorBNC) -> TensorBTSC:
        return self.decode(z)
