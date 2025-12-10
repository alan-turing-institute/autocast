from einops import rearrange
from torch import nn

from autocast.types import Tensor

from .residual import ResBlock, Residual
from .spatial_attention import SpatialAttentionWrapper

__all__ = ["ResBlock", "Residual", "SpatialAttentionWrapper"]


class Rearrange(nn.Module):
    """Permute the input tensor dimensions."""

    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = pattern

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, self.pattern)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern})"
