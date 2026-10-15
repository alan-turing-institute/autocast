"""Temporary imports for Isambard timing checkpoints; remove after migration."""

from autocast.nn.spatiotemporal_vit import (
    SpatiotemporalViTBackbone,
)
from autocast.nn.spatiotemporal_vit import (
    _SpatiotemporalBlock as _SpacetimeBlock,
)
from autocast.nn.spatiotemporal_vit import (
    _SpatiotemporalViT as _SpacetimeViT,
)

__all__ = [
    "SpacetimeViTBackbone",
    "SpatiotemporalViTBackbone",
    "_SpacetimeBlock",
    "_SpacetimeViT",
]

SpacetimeViTBackbone = SpatiotemporalViTBackbone
