from abc import ABC, abstractmethod

from torch import nn

from autocast.types import Tensor, TensorBNC, TensorBTSC


class Decoder(nn.Module, ABC):
    """Base Decoder."""

    def postprocess(self, decoded: Tensor) -> TensorBTSC:
        """Optionally transform the decoded tensor before returning.

        Subclasses can override to implement post-decoding steps. Default is
        identity.
        """
        return decoded

    @abstractmethod
    def decode(self, z: TensorBNC) -> TensorBTSC:
        """Decode the latent tensor back to the original space.

        Parameters
        ----------
        z: TensorBNC
            Latent tensor to be decoded.

        Returns
        -------
            Tensor: Decoded tensor in the original space.
        """

    def forward(self, z: TensorBNC) -> TensorBTSC:
        return self.decode(z)
