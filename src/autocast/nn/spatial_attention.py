from einops import rearrange
from torch import Tensor, nn

from autocast.types import TensorBCS


class SpatialAttentionWrapper(nn.Module):
    """Wrapper to handle spatial dimensions for MultiheadSelfAttention.

    Converts (B, C, W, H, ...) -> (B, W*H*..., C) -> attention -> (B, C, W, H, ...).
    """

    def __init__(self, attention_module: nn.Module, spatial: int) -> None:
        """Initialize the spatial attention wrapper.

        Parameters
        ----------
        attention_module: nn.Module
            The attention module to wrap.
        spatial: int
            Number of spatial dimensions.

        """
        super().__init__()
        self.attention = attention_module
        self.spatial = spatial

    def forward(self, x: TensorBCS) -> TensorBCS:
        """Forward pass handling spatial dimension transformation.

        Parameters
        ----------
        x: Tensor
            Input tensor with shape (B, C, spatial_dims...).

        Returns
        -------
        Tensor
            Output tensor with shape (B, C, spatial_dims...).

        """
        batch_size, channels = x.shape[:2]
        spatial_shape = x.shape[2:]

        # Flatten spatial dims and transpose: (B, C, H*W*...) -> (B, H*W*..., C)
        x_flat = rearrange(x, "b c ... -> b (...) c")

        # Apply attention
        y = self.attention(x_flat)

        # Transpose back and reshape: (B, W*H*..., C) -> (B, C, W, H, ...)
        return rearrange(y, "b ... c -> b c ...").reshape(
            batch_size, channels, *spatial_shape
        )


class Residual(nn.Sequential):
    """Residual wrapper that adds input to output.

    This wrapper can be used to add a residual connection around any sequence
    of layers.
    """

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Parameters
        ----------
        input: Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Input + output from sequential layers.

        """
        return input + super().forward(input)
