from abc import ABC
from typing import Any

from torch import nn

from auto_cast.types import Batch, Tensor


class Encoder(nn.Module, ABC):
    """Base encoder."""

    encoder_model: nn.Module
    latent_dim: int

    def preprocess(self, batch: Batch) -> Batch:
        """Optionally transform a batch before encoding.

        Subclasses can override to implement pre-encoding steps that still
        return a fully-populated `Batch` instance. Default is identity.
        """
        return batch

    def encode(self, batch: Batch) -> Tensor:
        """Encode the input tensor into the latent space.

        Parameters
        ----------
        x: Batch
            Input batch to be encoded.

        Returns
        -------
        Tensor
            Encoded tensor in the latent space.
        """
        msg = "The encode method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, batch: Batch) -> Tensor:
        return self.encode(batch)
