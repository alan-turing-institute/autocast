"""Adding configured references must not alter unrelated processor defaults."""

import pytest
from omegaconf import OmegaConf

from autocast.scripts.setup import _apply_processor_channel_defaults


@pytest.mark.parametrize("masked_window", [False, True])
@pytest.mark.parametrize("persistence_reference", [False, True])
def test_existing_backbone_and_normalizer_defaults_are_unchanged(
    masked_window, persistence_reference
):
    config = OmegaConf.create(
        {
            "n_steps_input": 7,
            "n_steps_output": 9,
            "backbone": {
                "n_steps_input": "auto",
                "n_steps_output": "auto",
                "spatial_resolution": "auto",
            },
            "residual_normalizer": {"n_steps_output": "auto"},
        }
    )
    if masked_window:
        config._target_ = "autocast.processors.FlowMatchingMaskedWindowProcessor"
    if persistence_reference:
        config.reference = {
            "_target_": "autocast.processors.residual_reference.LastFrameReference",
            "n_steps_output": "auto",
        }

    _apply_processor_channel_defaults(
        config,
        in_channels=1,
        out_channels=1,
        n_channels_out=1,
        n_steps_input=2,
        n_steps_output=3,
        spatial_resolution=(2, 3),
    )

    assert config.n_steps_input == 7
    assert config.n_steps_output == 9
    assert config.backbone.n_steps_input == (5 if masked_window else 2)
    assert config.backbone.n_steps_output == (5 if masked_window else 3)
    assert config.backbone.spatial_resolution == "auto"
    assert config.residual_normalizer.n_steps_output == 3
    if persistence_reference:
        assert config.reference["n_steps_output"] == 3
