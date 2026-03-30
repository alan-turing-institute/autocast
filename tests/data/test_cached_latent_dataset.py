"""Tests for CachedLatentDataset and its integration with EncodedDataModule."""

import pytest
import torch

from autocast.data.encoded_dataset import (
    CachedLatentDataset,
    EncodedDataModule,
)
from autocast.types.batch import EncodedBatch, EncodedSample
from autocast.types.collation import collate_encoded_samples


def _create_fake_trajectories(
    tmp_path, split, n_trajectories=3, n_timesteps=10, shape_spatial=(8, 8, 4)
):
    """Create fake cached trajectory .pt files for testing."""
    split_dir = tmp_path / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for traj_idx in range(n_trajectories):
        traj = {
            "encoded_fields": torch.randn(n_timesteps, *shape_spatial),
            "global_cond": torch.randn(8),
        }
        torch.save(traj, split_dir / f"traj_{traj_idx:06d}.pt")
    return split_dir


# --- CachedLatentDataset tests ---


def test_cached_dataset_loads_and_indexes_correctly(tmp_path):
    n_traj, n_t = 3, 10
    _create_fake_trajectories(tmp_path, "train", n_trajectories=n_traj, n_timesteps=n_t)

    dataset = CachedLatentDataset(
        cache_dir=tmp_path / "train", n_steps_input=2, n_steps_output=2, stride=2
    )
    # window_size=4, (10-4)//2 + 1 = 4 windows per trajectory, 3 trajectories
    assert len(dataset) == n_traj * 4


def test_cached_dataset_windowing_changes_length(tmp_path):
    n_t = 10
    _create_fake_trajectories(tmp_path, "train", n_trajectories=1, n_timesteps=n_t)

    ds_small = CachedLatentDataset(
        cache_dir=tmp_path / "train", n_steps_input=1, n_steps_output=1, stride=1
    )
    ds_large = CachedLatentDataset(
        cache_dir=tmp_path / "train", n_steps_input=4, n_steps_output=4, stride=4
    )
    # stride=1, window=2: (10-2)//1 + 1 = 9
    assert len(ds_small) == 9
    # stride=4, window=8: (10-8)//4 + 1 = 1
    assert len(ds_large) == 1


def test_cached_dataset_getitem_returns_encoded_sample(tmp_path):
    spatial = (8, 8, 4)
    _create_fake_trajectories(
        tmp_path, "train", n_trajectories=1, n_timesteps=6, shape_spatial=spatial
    )

    dataset = CachedLatentDataset(
        cache_dir=tmp_path / "train", n_steps_input=2, n_steps_output=3
    )
    sample = dataset[0]

    assert isinstance(sample, EncodedSample)
    assert sample.encoded_inputs.shape == (2, *spatial)
    assert sample.encoded_output_fields.shape == (3, *spatial)
    assert sample.global_cond is not None
    assert isinstance(sample.encoded_info, dict)


def test_cached_dataset_without_global_cond(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)
    traj = {"encoded_fields": torch.randn(5, 8, 8, 4)}
    torch.save(traj, split_dir / "traj_000000.pt")

    dataset = CachedLatentDataset(
        cache_dir=split_dir, n_steps_input=2, n_steps_output=2
    )
    sample = dataset[0]
    assert sample.global_cond is None


def test_cached_dataset_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        CachedLatentDataset(cache_dir=tmp_path / "nonexistent")


def test_cached_dataset_empty_directory_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No traj_"):
        CachedLatentDataset(cache_dir=empty_dir)


def test_cached_dataset_collation_produces_encoded_batch(tmp_path):
    spatial = (8, 8, 4)
    _create_fake_trajectories(
        tmp_path, "train", n_trajectories=1, n_timesteps=10, shape_spatial=spatial
    )
    dataset = CachedLatentDataset(
        cache_dir=tmp_path / "train", n_steps_input=2, n_steps_output=2
    )

    samples = [dataset[i] for i in range(3)]
    batch = collate_encoded_samples(samples)

    assert isinstance(batch, EncodedBatch)
    assert batch.encoded_inputs.shape == (3, 2, *spatial)
    assert batch.encoded_output_fields.shape == (3, 2, *spatial)


# --- EncodedDataModule with CachedLatentDataset tests ---


def test_encoded_datamodule_setup_creates_datasets(tmp_path):
    for split in ("train", "valid", "test"):
        _create_fake_trajectories(tmp_path, split, n_trajectories=2, n_timesteps=8)

    dm = EncodedDataModule(
        data_path=str(tmp_path),
        dataset_cls=CachedLatentDataset,
        n_steps_input=2,
        n_steps_output=2,
        stride=2,
        batch_size=2,
    )
    dm.setup(stage="fit")

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    # window_size=4, stride=2: (8-4)//2 + 1 = 3 windows per traj, 2 trajs
    assert len(dm.train_dataset) == 6  # type: ignore[arg-type]


def test_encoded_datamodule_dataloaders_produce_encoded_batch(tmp_path):
    for split in ("train", "valid", "test"):
        _create_fake_trajectories(tmp_path, split, n_trajectories=1, n_timesteps=6)

    dm = EncodedDataModule(
        data_path=str(tmp_path),
        dataset_cls=CachedLatentDataset,
        n_steps_input=2,
        n_steps_output=2,
        batch_size=2,
    )

    dm.setup(stage="fit")
    train_batch = next(iter(dm.train_dataloader()))
    assert isinstance(train_batch, EncodedBatch)
    assert train_batch.encoded_inputs.shape[1] == 2  # n_steps_input

    val_batch = next(iter(dm.val_dataloader()))
    assert isinstance(val_batch, EncodedBatch)

    dm.setup(stage="test")
    test_batch = next(iter(dm.test_dataloader()))
    assert isinstance(test_batch, EncodedBatch)


def test_encoded_datamodule_missing_split_is_none(tmp_path):
    _create_fake_trajectories(tmp_path, "train", n_trajectories=1, n_timesteps=6)

    dm = EncodedDataModule(
        data_path=str(tmp_path),
        dataset_cls=CachedLatentDataset,
        n_steps_input=2,
        n_steps_output=2,
        batch_size=2,
    )
    dm.setup(stage="fit")

    assert dm.train_dataset is not None
    assert dm.val_dataset is None
