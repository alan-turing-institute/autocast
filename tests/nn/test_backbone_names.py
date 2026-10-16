"""Compatibility of ViT names in configs and serialized objects."""

import pickle

from hydra.utils import get_class

from autocast.nn.base import TemporalBackboneBase
from autocast.nn.spatiotemporal_vit import SpatiotemporalViTBackbone
from autocast.nn.vit import SpatialViTBackbone


def test_legacy_spatial_vit_name_resolves_to_same_class():
    assert get_class("autocast.nn.vit.TemporalViTBackbone") is SpatialViTBackbone
    # Protocol 0 GLOBAL encodes the original module/class lookup in a pickle.
    legacy_reference = b"cautocast.nn.vit\nTemporalViTBackbone\n."
    assert pickle.loads(legacy_reference) is SpatialViTBackbone


def test_vit_backbones_keep_existing_shared_base():
    assert TemporalBackboneBase.__name__ == "TemporalBackboneBase"
    assert issubclass(SpatialViTBackbone, TemporalBackboneBase)
    assert issubclass(SpatiotemporalViTBackbone, TemporalBackboneBase)
