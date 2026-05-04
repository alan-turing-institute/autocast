import importlib.util
import json
import subprocess
import sys
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


def test_channel_idxs_slices_data_and_subsets_norm(deterministic_data, stats_dict):
    """`channel_idxs` should slice data channels and align norm field names."""
    dataset = ReactionDiffusionDataset(
        data_path=None,
        data=deterministic_data,
        n_steps_input=2,
        n_steps_output=1,
        channel_idxs=(1,),
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        normalization_stats=stats_dict,
    )

    # Sliced data keeps only channel 1 (V).
    assert dataset.data.shape[-1] == 1
    assert dataset[0].input_fields.shape[-1] == 1

    # Norm field names subset to match sliced channels.
    assert dataset.norm is not None
    assert dataset.norm.core_field_names == ["V"]

    # Normalization uses V stats (mean=4.0, std=2.0) against the original V channel.
    expected = (deterministic_data["data"][0][:2, ..., 1] - 4.0) / 2.0
    assert torch.allclose(dataset[0].input_fields[..., 0], expected)


def test_channel_idxs_none_is_noop(deterministic_data):
    """`channel_idxs=None` should leave all channels intact."""
    dataset = ReactionDiffusionDataset(
        data_path=None,
        data=deterministic_data,
        n_steps_input=2,
        n_steps_output=1,
        channel_idxs=None,
        use_normalization=False,
    )
    assert dataset.data.shape[-1] == 2
    assert dataset[0].input_fields.shape[-1] == 2


def test_datamodule_threads_channel_idxs(deterministic_data, stats_dict):
    """DataModule should propagate `channel_idxs` to all sub-datasets."""
    dm = SpatioTemporalDataModule(
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
        channel_idxs=(0,),
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        normalization_stats=stats_dict,
    )

    for ds in (
        dm.train_dataset,
        dm.val_dataset,
        dm.test_dataset,
        dm.rollout_val_dataset,
        dm.rollout_test_dataset,
    ):
        assert ds.data.shape[-1] == 1
        assert ds.norm is not None
        assert ds.norm.core_field_names == ["U"]


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


@pytest.mark.skipif(
    sys.platform.startswith("win") and importlib.util.find_spec("psutil") is None,
    reason="On Windows, this RSS probe requires psutil",
)
def test_normalized_dataloader_rss_growth_is_bounded() -> None:
    """Normalized dataloader iteration should not show runaway RSS growth.

    Runs in a subprocess so measurement is isolated from pytest process memory.
    """
    code = r"""
import json
import os
import sys

import torch

from autocast.data.datamodule import SpatioTemporalDataModule
from autocast.data.dataset import ReactionDiffusionDataset


def rss_mb() -> float:
    # Preferred cross-platform path.
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass

    # Stdlib fallback (macOS/Linux). Note: ru_maxrss is peak RSS.
    try:
        import resource

        rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss_raw / (1024.0 * 1024.0)
        return rss_raw / 1024.0
    except Exception:
        pass

    # Linux procfs fallback for current RSS.
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass

    raise RuntimeError("Unable to determine RSS on this platform")


def build_data() -> dict:
    data = torch.randn(16, 24, 16, 16, 2, dtype=torch.float32)
    return {
        "data": data,
        "constant_scalars": torch.randn(16, 2, dtype=torch.float32),
        "constant_fields": None,
    }


stats = {
    "stats": {
        "mean": {"U": 2.0, "V": 4.0},
        "std": {"U": 1.0, "V": 2.0},
        "mean_delta": {"U": 0.0, "V": 0.0},
        "std_delta": {"U": 0.1, "V": 0.2},
    },
    "core_field_names": ["U", "V"],
    "constant_field_names": [],
}


def probe(use_norm: bool) -> list[float]:
    data = build_data()
    dm = SpatioTemporalDataModule(
        data_path=None,
        data={"train": data, "valid": data, "test": data},
        dataset_cls=ReactionDiffusionDataset,
        n_steps_input=2,
        n_steps_output=2,
        batch_size=8,
        num_workers=0,
        use_normalization=use_norm,
        normalization_stats=stats,
    )

    loader = dm.train_dataloader()
    it = iter(loader)
    points = []
    for i in range(1200):
        try:
            _ = next(it)
        except StopIteration:
            it = iter(loader)
            _ = next(it)
        if i % 100 == 0:
            points.append(rss_mb())
    return points


off = probe(False)
on = probe(True)
print(json.dumps({"off": off, "on": on}))
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout.strip())
    off_points = payload["off"]
    on_points = payload["on"]

    off_growth = off_points[-1] - off_points[0]
    on_growth = on_points[-1] - on_points[0]

    # Keep generous bounds for CI variance while rejecting strong linear drift.
    assert on_growth < 64.0, (
        f"Normalized dataloader RSS grew by {on_growth:.1f} MB (points={on_points})"
    )
    assert on_growth <= off_growth + 32.0, (
        "Normalized RSS growth exceeded baseline by too much: "
        f"on={on_growth:.1f} MB, off={off_growth:.1f} MB, "
        f"on_points={on_points}, off_points={off_points}"
    )
