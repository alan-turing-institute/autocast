"""Trajectory selection precedes windowing and preserves source identities."""

from functools import partial
from pathlib import Path

import h5py
import numpy as np
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


@pytest.fixture
def make_dataset(payload):
    return partial(SpatioTemporalDataset, data_path=None, data=payload)


def _save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pt":
        torch.save(data, path)
    else:
        with h5py.File(path, "w") as handle:
            for name, value in data.items():
                if value is not None:
                    handle.create_dataset(name, data=value)


@pytest.mark.parametrize(
    ("selection", "indices"), [(None, [0, 1, 2, 3]), (2, [0, 1]), ([3, 1], [3, 1])]
)
def test_selection_keeps_complete_trajectories_and_constants(
    payload, make_dataset, selection, indices
):
    dataset = make_dataset(trajectories=selection, stride=2)
    assert dataset.n_trajectories == len(indices)
    assert len(dataset) == len(indices) * 3
    assert dataset.trajectory_indices == indices
    assert dataset.sample_trajectory_indices == [i for i in indices for _ in range(3)]
    assert dataset.sample_window_indices == [0, 1, 2] * len(indices)
    for local_idx, source_idx in enumerate(indices):
        sample = dataset[local_idx * 3]
        for field, expected in (
            ("input_fields", payload["data"][source_idx, :1]),
            ("output_fields", payload["data"][source_idx, 1:2]),
            ("constant_scalars", payload["constant_scalars"][source_idx]),
            ("constant_fields", payload["constant_fields"][source_idx]),
        ):
            assert torch.equal(getattr(sample, field), expected), field
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
    make_dataset, selection, error, validation_error
):
    with validation_error(error, match="trajector"):
        make_dataset(trajectories=selection)


@pytest.mark.parametrize("autoencoder_mode", [False, True])
@pytest.mark.parametrize("use_normalization", [False, True])
def test_datamodule_selection_and_training_normalization(
    payload, autoencoder_mode, use_normalization
):
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
        autoencoder_mode=autoencoder_mode,
        use_normalization=use_normalization,
        normalization_stats=stats,
        num_workers=0,
        pin_memory=False,
    )

    assert dm.train_dataset.trajectory_indices == [0, 1, 2, 3]
    assert dm.val_dataset.trajectory_indices == [3, 1]
    assert dm.test_dataset.trajectory_indices == [0, 1]
    datasets = [dm.train_dataset, dm.val_dataset, dm.test_dataset]
    if not autoencoder_mode:
        valid_rollout = dm.rollout_valid_dataloader().dataset
        test_rollout = dm.rollout_test_dataloader().dataset
        assert isinstance(valid_rollout, SpatioTemporalDataset)
        assert isinstance(test_rollout, SpatioTemporalDataset)
        assert valid_rollout.sample_trajectory_indices == [3, 1]
        assert test_rollout.sample_trajectory_indices == [0, 1]
        assert dm.rollout_val_dataset.sample_trajectory_indices == [0, 1, 2, 3]
        assert test_rollout.data is dm.test_dataset.data
        datasets += [valid_rollout, test_rollout]
    for dataset in datasets:
        raw = payload["data"][dataset.trajectory_indices[0], :1]
        expected = (
            (raw - torch.tensor([2.0, 4.0])) / torch.tensor([1.0, 2.0])
            if use_normalization
            else raw
        )
        assert torch.equal(dataset[0].input_fields, expected)


@pytest.mark.parametrize(
    ("selection", "indices"), [(None, [0, 1, 2, 3]), (2, [0, 1]), ([3, 1], [3, 1])]
)
def test_custom_reader_needs_no_trajectory_metadata(payload, selection, indices):
    class CustomDataset(SpatioTemporalDataset):
        def read_data(self, data_path):
            self.data_path = data_path
            self.data = payload["data"]
            self.constant_scalars = payload["constant_scalars"]
            self.constant_fields = payload["constant_fields"]

    dm = SpatioTemporalDataModule(
        data_path="custom_dataset",
        dataset_cls=CustomDataset,
        valid_trajectories=selection,
        test_trajectories=selection,
        num_workers=0,
        pin_memory=False,
    )

    assert dm.train_dataset.trajectory_indices == [0, 1, 2, 3]
    assert dm.val_dataset.trajectory_indices == indices
    assert dm.test_dataset.trajectory_indices == indices
    valid_rollout = dm.rollout_valid_dataloader().dataset
    test_rollout = dm.rollout_test_dataset
    assert isinstance(valid_rollout, SpatioTemporalDataset)
    assert valid_rollout.sample_trajectory_indices == indices
    assert test_rollout.sample_trajectory_indices == indices
    assert torch.equal(valid_rollout[0].input_fields, payload["data"][indices[0], :1])


def test_custom_parser_can_crop_rows_after_calling_super(payload):
    class CroppingDataset(SpatioTemporalDataset):
        def parse_data(self, data):
            super().parse_data(data)
            self.data = self.data[:2]

    dataset = CroppingDataset(data_path=None, data={"data": payload["data"]})

    assert dataset.trajectory_indices == [0, 1]
    assert dataset.sample_trajectory_indices == [0] * 5 + [1] * 5
    assert torch.equal(dataset.data, payload["data"][:2])


@pytest.mark.parametrize(("selection", "indices"), [(None, [3, 1]), ([1], [1])])
def test_preloaded_reuse_preserves_frames_channels_and_source_ids(
    payload, make_dataset, selection, indices
):
    dataset = make_dataset(trajectories=[3, 1], channel_idxs=(1,), start_frame=2)
    rollout = make_dataset(
        data=dataset.to_preloaded_data(),
        trajectories=selection,
        full_trajectory_mode=True,
        channel_idxs=(1,),
        start_frame=2,
    )
    if selection is None:
        assert rollout.data is dataset.data
    expected = payload["data"][indices, 2:, ..., 1:]
    assert torch.equal(rollout.data, expected)
    assert rollout.sample_trajectory_indices == indices
    assert torch.equal(rollout[0].input_fields, expected[0, :1])
    assert torch.equal(rollout[0].output_fields, expected[0, 1:])
    assert build_trajectory_metadata(rollout) == {
        index: {"cs0": 100.0 + index} for index in indices
    }
    with pytest.raises(ValueError, match="Unavailable trajectory indices"):
        make_dataset(data=dataset.to_preloaded_data(), trajectories=[0], start_frame=2)


@pytest.mark.parametrize(
    ("source", "ids_type"),
    [
        ("memory", None),
        ("pt", None),
        ("memory", np.array),
        ("h5", np.array),
        ("memory", torch.tensor),
        ("pt", torch.tensor),
        ("pt", list),
        ("h5", list),
    ],
)
def test_preloaded_ids_and_file_round_trips(
    tmp_path, payload, make_dataset, source, ids_type
):
    selected = make_dataset(trajectories=[3, 1])
    data = selected.to_preloaded_data()
    data["_trajectory_indices"] = None if ids_type is None else ids_type([3, 1])
    indices = [0, 1] if ids_type is None else [3, 1]
    if source == "memory":
        dataset = SpatioTemporalDataset(
            data_path=None, data=data, full_trajectory_mode=True
        )
    else:
        path = tmp_path / f"data.{source}"
        _save(data, path)
        dataset = SpatioTemporalDataset(data_path=str(path), full_trajectory_mode=True)
    assert dataset.sample_trajectory_indices == indices
    assert torch.equal(dataset.data, payload["data"][[3, 1]])
    assert build_trajectory_metadata(dataset) == {
        index: {"cs0": value}
        for index, value in zip(indices, [103.0, 101.0], strict=True)
    }


@pytest.mark.parametrize(
    ("valid", "test", "valid_ids", "test_ids"),
    [
        (None, None, [0, 1, 2, 3], [0, 1, 2, 3]),
        ("null", "null", [0, 1, 2, 3], [0, 1, 2, 3]),
        ("2", "[3,1]", [0, 1], [3, 1]),
        ("[3,1]", "2", [3, 1], [0, 1]),
    ],
)
def test_hydra_config_accepts_optional_trajectory_overrides(
    tmp_path, payload, valid, test, valid_ids, test_ids
):
    for split in ("train", "valid", "test"):
        _save(payload, tmp_path / split / "data.pt")
    overrides = [
        f"+datamodule.{split}_trajectories={value}"
        for split, value in (("valid", valid), ("test", test))
        if value is not None
    ]
    config_dir = Path(autocast.__file__).parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="datamodule/autosim",
            overrides=[f"datamodule.data_path={tmp_path}", *overrides],
        )

    dm = instantiate(config.datamodule)

    assert dm.train_dataset.trajectory_indices == [0, 1, 2, 3]
    for dataset, indices in ((dm.val_dataset, valid_ids), (dm.test_dataset, test_ids)):
        assert dataset.sample_trajectory_indices == [
            index for index in indices for _ in range(2)
        ]
