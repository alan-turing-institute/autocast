from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from the_well.data.normalization import ZScoreNormalization

from autocast.data.datamodule import TheWellDataModule
from autocast.data.dataset import TheWell
from autocast.scripts.setup import setup_epd_model


@pytest.mark.parametrize("use_normalization", [True, False])
def test_well_model_setup_preserves_output_units(
    monkeypatch, make_toy_batch, use_normalization
):
    """Carry the loader's normalizer through the wrapper into predictions."""
    normalizer = ZScoreNormalization(
        stats={
            "mean": {"a": 10.0, "b": 20.0},
            "std": {"a": 2.0, "b": 3.0},
            "mean_delta": {"a": 0.0, "b": 0.0},
            "std_delta": {"a": 1.0, "b": 1.0},
        },
        core_field_names=["a", "b"],
        core_constant_field_names=[],
    )
    # Only replace file I/O; exercise the real wrapper and model setup.
    loader = Mock(norm=normalizer if use_normalization else None, metadata=None)
    monkeypatch.setattr("autocast.data.dataset.WellDataset", Mock(return_value=loader))
    dataset = TheWell(path="unused", use_normalization=use_normalization)
    batch = make_toy_batch(batch_size=1, t_in=1, w=2, h=2, c=2)
    config = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "_target_": "autocast.encoders.identity.IdentityEncoder",
                    "in_channels": 2,
                },
                "decoder": {
                    "_target_": "autocast.decoders.identity.IdentityDecoder",
                    "in_channels": 2,
                },
                "processor": {"_target_": "conftest.CondCaptureProcessor"},
            },
            "optimizer": {"optimizer": "adamw", "learning_rate": 1e-4},
        }
    )
    model = setup_epd_model(
        config,
        {"example_batch": batch, "n_steps_input": 1, "n_steps_output": 1},
        Mock(spec=TheWellDataModule, train_dataset=dataset),
    )

    # Training stays in model units; predict_step restores physical units once.
    torch.testing.assert_close(model(batch), batch.input_fields)
    expected = batch.input_fields
    if use_normalization:
        expected = expected * torch.tensor([2.0, 3.0]) + torch.tensor([10.0, 20.0])
    torch.testing.assert_close(model.predict_step(batch, 0), expected)
