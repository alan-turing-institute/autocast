"""Trajectory selection precedes windowing and preserves source identities."""

from pathlib import Path

import h5py
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import autocast
from autocast.data.datamodule import SpatioTemporalDataModule
from autocast.data.dataset import SpatioTemporalDataset
from autocast.metrics.trajectory import build_trajectory_metadata


@pytest.fixture
def payload():
    return {
        "data": torch.arange(4 * 6 * 2 * 2 * 2, dtype=torch.float32).reshape(
            4, 6, 2, 2, 2
        ),
        "constant_scalars": torch.arange(100.0, 104.0).reshape(4, 1),
        "constant_fields": torch.arange(16.0).reshape(4, 2, 2, 1),
    }


@pytest.mark.parametrize(
    ("selection", "indices"), [(None, [0, 1, 2, 3]), (2, [0, 1]), ([3, 1], [3, 1])]
)
def test_selection_keeps_complete_trajectories_and_constants(
    payload, selection, indices
):
    dataset = SpatioTemporalDataset(
        data_path=None, data=payload, trajectories=selection, stride=2
    )
    assert dataset.n_trajectories == len(indices)
    assert len(dataset) == len(indices) * 3
    assert dataset.trajectory_indices == indices
    assert dataset.sample_trajectory_indices == [i for i in indices for _ in range(3)]
    assert dataset.sample_window_indices == [0, 1, 2] * len(indices)
    for local_idx, source_idx in enumerate(indices):
        sample = dataset[local_idx * 3]
        assert torch.equal(sample.input_fields, payload["data"][source_idx, :1])
        assert torch.equal(sample.output_fields, payload["data"][source_idx, 1:2])
        assert sample.constant_scalars is not None
        assert sample.constant_fields is not None
        assert torch.equal(
            sample.constant_scalars, payload["constant_scalars"][source_idx]
        )
        assert torch.equal(
            sample.constant_fields, payload["constant_fields"][source_idx]
        )
    assert payload["data"].shape[0] == 4


@pytest.mark.parametrize(
    ("selection", "error"),
    [
        (True, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (5, ValueError),
        (1.5, TypeError),
        ("0,1", TypeError),
        ([], ValueError),
        ([1, 1], ValueError),
        ([-1], ValueError),
        ([4], ValueError),
        ([False], TypeError),
        ([0, 1.0], TypeError),
    ],
)
def test_invalid_selections_fail_before_windowing(
    payload, selection, error, validation_error
):
    with validation_error(error, match="trajector"):
        SpatioTemporalDataset(data_path=None, data=payload, trajectories=selection)


@pytest.mark.parametrize("autoencoder_mode", [False, True])
def test_datamodule_selects_only_valid_and_test(payload, autoencoder_mode):
    dm = SpatioTemporalDataModule(
        data_path=None,
        data=dict.fromkeys(("train", "valid", "test"), payload),
        valid_trajectories=[3, 1],
        test_trajectories=2,
        autoencoder_mode=autoencoder_mode,
        num_workers=0,
        pin_memory=False,
    )

    assert dm.train_dataset.trajectory_indices == [0, 1, 2, 3]
    assert dm.val_dataset.trajectory_indices == [3, 1]
    assert dm.test_dataset.trajectory_indices == [0, 1]
    if not autoencoder_mode:
        valid_rollout = dm.rollout_valid_dataloader().dataset
        test_rollout = dm.rollout_test_dataloader().dataset
        assert isinstance(valid_rollout, SpatioTemporalDataset)
        assert isinstance(test_rollout, SpatioTemporalDataset)
        assert valid_rollout.sample_trajectory_indices == [3, 1]
        assert test_rollout.sample_trajectory_indices == [0, 1]
        assert dm.rollout_val_dataset.sample_trajectory_indices == [0, 1, 2, 3]
        assert test_rollout.data is dm.test_dataset.data
        assert torch.equal(valid_rollout[0].input_fields, payload["data"][3, :1])


def test_selection_and_preloaded_reuse_preserve_frames_channels_and_ids(payload):
    dataset = SpatioTemporalDataset(
        data_path=None,
        data=payload,
        trajectories=[3, 1],
        channel_idxs=(1,),
        start_frame=2,
    )
    rollout = SpatioTemporalDataset(
        data_path=None,
        data=dataset.to_preloaded_data(),
        full_trajectory_mode=True,
        channel_idxs=(1,),
        start_frame=2,
    )

    assert rollout.data is dataset.data
    assert rollout.sample_trajectory_indices == [3, 1]
    assert torch.equal(rollout[0].input_fields, payload["data"][3, 2:3, ..., 1:])
    assert torch.equal(rollout[0].output_fields, payload["data"][3, 3:, ..., 1:])
    assert build_trajectory_metadata(rollout) == {3: {"cs0": 103.0}, 1: {"cs0": 101.0}}


def test_reselecting_preloaded_data_uses_source_indices(payload):
    selected = SpatioTemporalDataset(data_path=None, data=payload, trajectories=[3, 1])
    reused = SpatioTemporalDataset(
        data_path=None, data=selected.to_preloaded_data(), trajectories=[1]
    )

    assert reused.trajectory_indices == [1]
    assert torch.equal(reused.data, payload["data"][1:2])
    with pytest.raises(ValueError, match="Unavailable trajectory indices"):
        SpatioTemporalDataset(
            data_path=None, data=selected.to_preloaded_data(), trajectories=[0]
        )


def test_selected_splits_keep_training_normalization(payload):
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
    dm = SpatioTemporalDataModule(
        data_path=None,
        data=dict.fromkeys(("train", "valid", "test"), payload),
        valid_trajectories=[3, 1],
        test_trajectories=2,
        use_normalization=True,
        normalization_stats=stats,
        num_workers=0,
        pin_memory=False,
    )

    for dataset in (
        dm.train_dataset,
        dm.val_dataset,
        dm.test_dataset,
        dm.rollout_valid_dataloader().dataset,
        dm.rollout_test_dataset,
    ):
        assert isinstance(dataset, SpatioTemporalDataset)
        source_idx = dataset.trajectory_indices[0]
        raw = payload["data"][source_idx, :1]
        expected = (raw - torch.tensor([2.0, 4.0])) / torch.tensor([1.0, 2.0])
        assert torch.equal(dataset[0].input_fields, expected)


@pytest.mark.parametrize("suffix", ["pt", "h5"])
def test_saved_selection_preserves_source_ids(tmp_path, payload, suffix):
    selected = SpatioTemporalDataset(data_path=None, data=payload, trajectories=[3, 1])
    path = tmp_path / f"selected.{suffix}"
    if suffix == "pt":
        torch.save(selected.to_preloaded_data(), path)
    else:
        with h5py.File(path, "w") as handle:
            for name, value in selected.to_preloaded_data().items():
                if value is not None:
                    handle.create_dataset(name, data=value)

    reused = SpatioTemporalDataset(data_path=str(path), full_trajectory_mode=True)

    assert reused.sample_trajectory_indices == [3, 1]
    assert build_trajectory_metadata(reused) == {3: {"cs0": 103.0}, 1: {"cs0": 101.0}}


@pytest.mark.parametrize(
    ("overrides", "valid_ids", "test_ids"),
    [
        ([], [0, 1, 2, 3], [0, 1, 2, 3]),
        (
            [
                "+datamodule.valid_trajectories=null",
                "+datamodule.test_trajectories=null",
            ],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ),
        (
            ["+datamodule.valid_trajectories=2", "+datamodule.test_trajectories=[3,1]"],
            [0, 1],
            [3, 1],
        ),
        (
            ["+datamodule.valid_trajectories=[3,1]", "+datamodule.test_trajectories=2"],
            [3, 1],
            [0, 1],
        ),
    ],
)
def test_hydra_config_accepts_optional_trajectory_overrides(
    tmp_path, payload, overrides, valid_ids, test_ids
):
    for split in ("train", "valid", "test"):
        directory = tmp_path / split
        directory.mkdir()
        torch.save(payload, directory / "data.pt")
    config_dir = Path(autocast.__file__).parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="datamodule/autosim",
            overrides=[f"datamodule.data_path={tmp_path}", *overrides],
        )

    dm = instantiate(config.datamodule)

    assert dm.train_dataset.trajectory_indices == [0, 1, 2, 3]
    assert dm.val_dataset.sample_trajectory_indices == [
        index for index in valid_ids for _ in range(2)
    ]
    assert dm.test_dataset.sample_trajectory_indices == [
        index for index in test_ids for _ in range(2)
    ]
