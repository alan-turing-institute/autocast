"""Compatibility of ViT names in configs and serialized objects."""

import pickle

import pytest
from hydra.utils import get_class

from autocast.nn.base import TemporalBackboneBase


@pytest.mark.parametrize(
    ("module", "legacy", "canonical"),
    [
        ("vit", "TemporalViTBackbone", "SpatialViTBackbone"),
        ("spacetime_vit", "SpacetimeViTBackbone", "SpatiotemporalViTBackbone"),
    ],
)
def test_legacy_backbone_names_resolve_to_same_class(module, legacy, canonical):
    module = f"autocast.nn.{module}"
    canonical_class = get_class(f"{module}.{canonical}")
    assert get_class(f"{module}.{legacy}") is canonical_class
    assert issubclass(canonical_class, TemporalBackboneBase)
    assert TemporalBackboneBase.__name__ == "TemporalBackboneBase"
    # Protocol 0 GLOBAL encodes the original module/class lookup in a pickle.
    legacy_reference = f"c{module}\n{legacy}\n.".encode()
    assert pickle.loads(legacy_reference) is canonical_class
