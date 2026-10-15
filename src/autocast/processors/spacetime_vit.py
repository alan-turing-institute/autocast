"""Temporary imports for Isambard timing checkpoints; remove after migration."""

from autocast.processors.spatiotemporal_vit import (
    SpatiotemporalViTMaskedWindowProcessor,
    SpatiotemporalViTProcessor,
)

SpacetimeViTProcessor = SpatiotemporalViTProcessor
SpacetimeViTMaskedWindowProcessor = SpatiotemporalViTMaskedWindowProcessor
