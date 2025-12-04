from abc import ABC
from typing import Any

from torch import nn

from auto_cast.types import Tensor


class Encoder(nn.Module, ABC):
    """Base encoder."""

    def encode(self, x: Tensor) -> Tensor:
        """Encode the input tensor into the latent space.

        Parameters
        ----------
        x: Tensor
            Input tensor to be encoded.

        Returns
        -------
        Tensor
            Encoded tensor in the latent space.
        """
        msg = "The encode method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
