"""Compatibility of ViT names in configs and serialized objects."""

import pickle

import pytest
from hydra.utils import get_class

from autocast.nn.base import TemporalBackboneBase
from autocast.nn.spatiotemporal_vit import SpatiotemporalViTBackbone
from autocast.nn.vit import SpatialViTBackbone


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("nn.vit.TemporalViTBackbone", "nn.vit.SpatialViTBackbone"),
        (
            "nn.spacetime_vit.SpacetimeViTBackbone",
            "nn.spatiotemporal_vit.SpatiotemporalViTBackbone",
        ),
        (
            "nn.spacetime_vit.SpatiotemporalViTBackbone",
            "nn.spatiotemporal_vit.SpatiotemporalViTBackbone",
        ),
        (
            "nn.spacetime_vit._SpacetimeBlock",
            "nn.spatiotemporal_vit._SpatiotemporalBlock",
        ),
        (
            "nn.spacetime_vit._SpacetimeViT",
            "nn.spatiotemporal_vit._SpatiotemporalViT",
        ),
        (
            "processors.spacetime_vit.SpacetimeViTProcessor",
            "processors.spatiotemporal_vit.SpatiotemporalViTProcessor",
        ),
        (
            "processors.spacetime_vit.SpacetimeViTMaskedWindowProcessor",
            "processors.spatiotemporal_vit.SpatiotemporalViTMaskedWindowProcessor",
        ),
    ],
)
def test_legacy_vit_names_resolve_to_same_class(legacy, canonical):
    canonical_class = get_class(f"autocast.{canonical}")
    assert get_class(f"autocast.{legacy}") is canonical_class
    # Protocol 0 GLOBAL encodes the original module/class lookup in a pickle.
    module, legacy = f"autocast.{legacy}".rsplit(".", 1)
    legacy_reference = f"c{module}\n{legacy}\n.".encode()
    assert pickle.loads(legacy_reference) is canonical_class


def test_vit_backbones_keep_existing_shared_base():
    assert TemporalBackboneBase.__name__ == "TemporalBackboneBase"
    assert issubclass(SpatialViTBackbone, TemporalBackboneBase)
    assert issubclass(SpatiotemporalViTBackbone, TemporalBackboneBase)
