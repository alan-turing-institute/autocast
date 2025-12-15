from pathlib import Path

import pytest
import torch
import yaml
from the_well.data.normalization import ZScoreNormalization

from autocast.data.datamodule import SpatioTemporalDataModule
from autocast.data.dataset import ReactionDiffusionDataset


@pytest.fixture
def stats_dict():
    return {
        "stats": {
            "mean": {"U": 2.0, "V": 4.0},
            "std": {"U": 1.0, "V": 2.0},
            "mean_delta": {"U": 0.0, "V": 0.0},
            "std_delta": {"U": 0.1, "V": 0.2},
        },
        "core_field_names": ["U", "V"],
        "constant_field_names": [],
    }


@pytest.fixture
def stats_file(tmp_path: Path, stats_dict):
    """Create a temporary stats.yaml file."""
    stats_path = tmp_path / "stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats_dict, f)
    return stats_path


@pytest.fixture
def deterministic_data():
    """Create deterministic spatiotemporal data for normalization testing.

    Creates data with known values to test normalization:
    - Channel 0 (U): values around mean=2.0, std=1.0
    - Channel 1 (V): values around mean=4.0, std=2.0
    """
    # Create 2 trajectories, 10 timesteps, 2x2 spatial grid, 2 channels
    # Generate data with specific mean and std for each channel
    data_U = torch.randn(2, 10, 2, 2) * 1.0 + 2.0  # mean=2.0, std=1.0
    data_V = torch.randn(2, 10, 2, 2) * 2.0 + 4.0  # mean=4.0, std=2.0

    # Stack channels: [2, 10, 2, 2, 2]
    data = torch.stack([data_U, data_V], dim=-1)

    return {
        "data": data,
        "constant_scalars": torch.tensor([[0.5, 1.0], [0.5, 1.0]]),
        "constant_fields": None,
    }


# Normalization setup tests


def test_normalization_from_file(deterministic_data, stats_file):
    """Test loading normalization stats from file."""
    dataset = ReactionDiffusionDataset(
        data_path=None,
        data=deterministic_data,
        n_steps_input=2,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        normalization_path=str(stats_file),
    )

    assert dataset.norm is not None
    assert isinstance(dataset.norm, ZScoreNormalization)


def test_normalization_from_dict(deterministic_data, stats_dict):
    """Test loading normalization stats from dict."""
    dataset = ReactionDiffusionDataset(
        data_path=None,
        data=deterministic_data,
        n_steps_input=2,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        normalization_stats=stats_dict,
    )

    assert dataset.norm is not None
    assert isinstance(dataset.norm, ZScoreNormalization)


# Normalization behavior tests


def test_unnormalized_data_returns_original_values(deterministic_data):
    """Test that without normalization, data is unchanged."""

    dataset = ReactionDiffusionDataset(
        data_path=None,
        data=deterministic_data,
        n_steps_input=2,
        n_steps_output=1,
        use_normalization=False,  # No normalization
    )

    # Compare the first sample of the dataset to the first 2 inputs of the original
    assert torch.allclose(dataset[0].input_fields, deterministic_data["data"][0][:2])


def test_normalized_data_is_transformed(deterministic_data, stats_dict):
    """Test that with normalization, data is transformed according to stats."""

    dataset = ReactionDiffusionDataset(
        data_path=None,
        data=deterministic_data,
        n_steps_input=2,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        normalization_stats=stats_dict,
    )

    assert dataset.norm is not None

    # Check normalization is applied correctly to each channel
    assert torch.allclose(
        dataset[0].input_fields[..., 0], deterministic_data["data"][0][:2, ..., 0] - 2.0
    )
    assert torch.allclose(
        dataset[0].input_fields[..., 1],
        (deterministic_data["data"][0][:2, ..., 1] - 4.0) / 2.0,
    )


def test_datamodule_with_and_without_normalization(deterministic_data, stats_dict):
    """Test DataModule can be configured with or without normalization."""

    # Test without normalization
    dm_no_norm = SpatioTemporalDataModule(
        data_path=None,
        data={
            "train": deterministic_data,
            "valid": deterministic_data,
            "test": deterministic_data,
        },
        dataset_cls=ReactionDiffusionDataset,
        n_steps_input=2,
        n_steps_output=1,
        batch_size=1,
        use_normalization=False,
    )

    assert dm_no_norm.train_dataset.norm is None
    assert dm_no_norm.val_dataset.norm is None

    # Test with normalization
    dm_with_norm = SpatioTemporalDataModule(
        data_path=None,
        data={
            "train": deterministic_data,
            "valid": deterministic_data,
            "test": deterministic_data,
        },
        dataset_cls=ReactionDiffusionDataset,
        n_steps_input=2,
        n_steps_output=1,
        batch_size=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        normalization_stats=stats_dict,
    )

    assert dm_with_norm.train_dataset.norm is not None
    assert dm_with_norm.val_dataset.norm is not None
