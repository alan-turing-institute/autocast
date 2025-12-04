from abc import ABC

from torch import nn

from auto_cast.types import Tensor, TensorBTSC


class Decoder(nn.Module, ABC):
    """Base Decoder."""

    decoder_model: nn.Module
    latent_dim: int

    def __init__(self, latent_dim: int, output_channels: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels

    def postprocess(self, decoded: Tensor) -> TensorBTSC:
        """Optionally transform the decoded tensor before returning.

        Subclasses can override to implement post-decoding steps. Default is
        identity.
        """
        return decoded

    def decode(self, z: TensorBTSC) -> Tensor:
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

    def __call__(self, z: TensorBTSC) -> TensorBTSC:
        return self.decode(z)
