from unittest.mock import Mock

import fsspec
import pytest
import torch
import yaml
from einops import repeat
from omegaconf import OmegaConf
from the_well.data.normalization import RMSNormalization, ZScoreNormalization

from autocast.data.datamodule import TheWellDataModule
from autocast.data.dataset import TheWell
from autocast.scripts.setup import setup_epd_model


@pytest.mark.parametrize("use_normalization", [True, False])
@pytest.mark.parametrize("delta_source", [None, "state", "local", "filesystem"])
@pytest.mark.parametrize("n_members", [1, 3])
@pytest.mark.parametrize("normalization_type", [ZScoreNormalization, RMSNormalization])
def test_well_model_setup_preserves_output_units(
    tmp_path,
    monkeypatch,
    make_toy_batch,
    use_normalization,
    delta_source,
    n_members,
    normalization_type,
):
    """Use native statistics, filesystem and field order for either state mode."""
    # Native Well statistics have no wrapper or field-name list. Metadata orders
    # the vector before the scalar, unlike the YAML mapping's alphabetical order.
    stats = {
        "mean": {"a": 10.0, "velocity": [20.0, -5.0]},
        "std": {"a": 2.0, "velocity": [3.0, 4.0]},
        "mean_delta": {"a": 0.5, "velocity": [-1.0, 2.0]},
        "std_delta": {"a": 0.25, "velocity": [2.0, 3.0]},
        "rms": {"a": 12.0, "velocity": [20.0, 6.0]},
        "rms_delta": {"a": 0.75, "velocity": [4.0, 5.0]},
    }
    field_names = ["velocity", "a"]
    normalizer = normalization_type(stats, field_names, [])
    fs = fsspec.filesystem("memory")
    stats_path = f"memory://{tmp_path.name}/stats.yaml"
    fs.pipe(stats_path, yaml.safe_dump(stats).encode())
    if delta_source == "local":
        local_stats = tmp_path / "stats.yaml"
        local_stats.write_text(yaml.safe_dump(stats))
        stats_path = str(local_stats)
    residual_prediction = delta_source is not None
    delta_file = delta_source in {"local", "filesystem"}
    # Replace external HDF5 discovery, retaining real filesystem/normalizers.
    loader = Mock(
        norm=normalizer if use_normalization else None,
        metadata=None,
        fs=fs,
        core_field_names=field_names,
        core_constant_field_names=[],
        normalization_type=normalization_type,
    )
    monkeypatch.setattr("autocast.data.dataset.WellDataset", Mock(return_value=loader))
    dataset = TheWell(path="unused", use_normalization=use_normalization)
    batch = make_toy_batch(batch_size=1, t_in=1, w=2, h=2, c=3)
    config = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "_target_": "autocast.encoders.identity.IdentityEncoder",
                    "in_channels": 3,
                },
                "decoder": {
                    "_target_": "autocast.decoders.identity.IdentityDecoder",
                    "in_channels": 3,
                },
                "processor": {"_target_": "conftest.CondCaptureProcessor"},
                "n_members": n_members,
                "residual_prediction": residual_prediction,
                "residual_delta_stats_path": stats_path if delta_file else None,
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
    baseline = batch.input_fields
    if normalization_type is ZScoreNormalization:
        state_mean = torch.tensor([20.0, -5.0, 10.0])
        state_scale = torch.tensor([3.0, 4.0, 2.0])
        increment = baseline * torch.tensor([2.0, 3.0, 0.25])
        increment = increment + torch.tensor([-1.0, 2.0, 0.5])
    else:
        state_mean = torch.zeros(3)
        state_scale = torch.tensor([20.0, 6.0, 12.0])
        increment = baseline * torch.tensor([4.0, 5.0, 0.75])
    expected = baseline
    if residual_prediction:
        if delta_file or use_normalization:
            expected = baseline + (
                increment / state_scale if use_normalization else increment
            )
        else:
            expected = baseline + baseline
    physical = expected
    if use_normalization:
        physical = expected * state_scale + state_mean
    if n_members > 1:
        expected = repeat(expected, "... -> ... m", m=n_members)
        physical = repeat(physical, "... -> ... m", m=n_members)
    torch.testing.assert_close(model(batch), expected)
    torch.testing.assert_close(model.predict_step(batch, 0), physical)
