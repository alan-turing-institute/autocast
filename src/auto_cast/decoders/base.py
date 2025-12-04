from abc import ABC
from typing import Any

from torch import nn

from auto_cast.types import Tensor


class Decoder(nn.Module, ABC):
    """Base Decoder."""

    def decode(self, z: Tensor) -> Tensor:
        """Decode the latent tensor back to the original space.

        Parameters
        ----------
        z: Tensor
            Latent tensor to be decoded.

        Returns
        -------
            Tensor: Decoded tensor in the original space.
        """
        msg = "The decode method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
