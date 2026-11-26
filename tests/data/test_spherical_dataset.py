"""A spherical payload is mapped from disk, and its windows keep the grid name."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from autocast.data.datamodule import SpatioTemporalDataModule
from autocast.data.dataset import ReactionDiffusionDataset, SpatioTemporalDataset
from autocast.types import (
    Batch,
    GridType,
    Sample,
    SphericalBatch,
    SphericalSample,
    collate_batches,
)

N_RUNS, N_FRAMES, N_LAT, N_LON, N_CHANNELS = 3, 7, 4, 8, 3


@pytest.fixture
def payload():
    size = N_RUNS * N_FRAMES * N_LAT * N_LON * N_CHANNELS
    return {
        "data": torch.arange(size, dtype=torch.float32).reshape(
            N_RUNS, N_FRAMES, N_LAT, N_LON, N_CHANNELS
        ),
        "constant_scalars": torch.arange(float(N_RUNS * 2)).reshape(N_RUNS, 2),
        "constant_fields": torch.arange(float(N_RUNS * N_LAT * N_LON * 2)).reshape(
            N_RUNS, N_LAT, N_LON, 2
        ),
        "grid_type": "legendre-gauss",
    }


@pytest.fixture
def cartesian_payload(payload):
    return {key: value for key, value in payload.items() if key != "grid_type"}


@pytest.fixture
def shard(tmp_path, payload):
    path = tmp_path / "shard.pt"
    torch.save(payload, path)
    return path


def test_pt_file_is_loaded_with_mmap(shard, monkeypatch):
    calls = []
    load = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    SpatioTemporalDataset(data_path=str(shard))

    assert calls[0].get("mmap") is True


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="Linux only")
def test_file_stays_mapped_after_construction(shard):
    dataset = SpatioTemporalDataset(data_path=str(shard))

    assert str(shard.resolve()) in Path("/proc/self/maps").read_text()
    assert len(dataset) > 0


def test_windows_match_unfold(shard, payload):
    n_in, n_out, stride = 2, 3, 2
    dataset = SpatioTemporalDataset(
        data_path=str(shard), n_steps_input=n_in, n_steps_output=n_out, stride=stride
    )
    per_run = [
        payload["data"][run].unfold(0, n_in + n_out, stride).permute(0, -1, 1, 2, 3)
        for run in range(N_RUNS)
    ]
    windows = torch.cat(per_run)
    n_windows = per_run[0].shape[0]

    assert len(dataset) == windows.shape[0] == N_RUNS * n_windows
    for index in range(len(dataset)):
        sample = dataset[index]
        assert torch.equal(sample.input_fields, windows[index, :n_in])
        assert torch.equal(sample.output_fields, windows[index, n_in:])
    assert torch.equal(dataset[-1].output_fields, windows[-1, n_in:])
    assert dataset.sample_trajectory_indices == [
        run for run in range(N_RUNS) for _ in range(n_windows)
    ]
    assert dataset.sample_window_indices == list(range(n_windows)) * N_RUNS


def test_item_holds_its_window_only(shard):
    dataset = SpatioTemporalDataset(
        data_path=str(shard), n_steps_input=1, n_steps_output=2
    )
    sample = dataset[4]
    window_bytes = 3 * N_LAT * N_LON * N_CHANNELS * 4

    assert sample.input_fields.untyped_storage().nbytes() == window_bytes
    assert (
        sample.input_fields.untyped_storage().data_ptr()
        != dataset.data.untyped_storage().data_ptr()
    )
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_samples_and_batches_keep_the_grid(shard, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dataset = SpatioTemporalDataset(data_path=str(shard))
    sample = dataset[0]

    assert dataset.grid_type is GridType.LEGENDRE_GAUSS
    assert type(sample) is SphericalSample
    assert sample.grid_type is GridType.LEGENDRE_GAUSS

    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_batches)))

    assert type(batch) is SphericalBatch
    assert batch.grid_type is GridType.LEGENDRE_GAUSS
    assert batch.input_fields.shape == (4, 1, N_LAT, N_LON, N_CHANNELS)
    assert batch.constant_fields is not None
    assert batch.constant_fields.shape == (4, N_LAT, N_LON, 2)
    for moved in (batch.to("cpu"), batch.pin_memory(), batch.repeat(2)):
        assert type(moved) is SphericalBatch
        assert moved.grid_type is GridType.LEGENDRE_GAUSS
    assert batch.repeat(2).input_fields.shape[0] == 8


def test_rollout_datasets_keep_the_grid(tmp_path, payload):
    for split in ("train", "valid", "test"):
        (tmp_path / split).mkdir()
        torch.save(payload, tmp_path / split / "data.pt")
    datamodule = SpatioTemporalDataModule(
        data_path=str(tmp_path), n_steps_input=1, n_steps_output=1, num_workers=0
    )

    assert datamodule.train_dataset.grid_type is GridType.LEGENDRE_GAUSS
    assert datamodule.rollout_val_dataset.grid_type is GridType.LEGENDRE_GAUSS
    assert datamodule.rollout_test_dataset.data is datamodule.test_dataset.data
    assert type(next(iter(datamodule.rollout_test_dataloader()))) is SphericalBatch


def test_preloaded_payload_saves_and_loads_with_weights_only(tmp_path, shard):
    preloaded = SpatioTemporalDataset(data_path=str(shard)).to_preloaded_data()
    path = tmp_path / "preloaded.pt"
    torch.save(preloaded, path)

    assert type(preloaded["grid_type"]) is str
    assert SpatioTemporalDataset(data_path=str(path)).grid_type is (
        GridType.LEGENDRE_GAUSS
    )


@pytest.mark.parametrize("grid_type", [None, "cartesian"])
def test_payload_off_the_sphere_gives_plain_types(cartesian_payload, grid_type):
    if grid_type is not None:
        cartesian_payload["grid_type"] = grid_type
    dataset = SpatioTemporalDataset(data_path=None, data=cartesian_payload)

    assert dataset.grid_type is None
    assert type(dataset[0]) is Sample
    assert type(collate_batches([dataset[0], dataset[1]])) is Batch


def test_unknown_grid_name_is_refused(payload):
    payload["grid_type"] = "gauss_legendre"

    with pytest.raises(ValueError, match="Unknown grid_type"):
        SpatioTemporalDataset(data_path=None, data=payload)


def test_samples_on_different_grids_are_not_collated(payload, cartesian_payload):
    sphere = SpatioTemporalDataset(data_path=None, data=payload)
    other = SpatioTemporalDataset(
        data_path=None, data=dict(payload, grid_type="equiangular")
    )
    plain = SpatioTemporalDataset(data_path=None, data=cartesian_payload)

    for samples in ([sphere[0], other[0]], [sphere[0], plain[0]]):
        with pytest.raises(ValueError, match="different grids"):
            collate_batches(samples)


def test_contiguous_selections_are_views(payload):
    dataset = SpatioTemporalDataset(
        data_path=None, data=payload, channel_idxs=(1, 2), trajectories=[0, 1]
    )

    assert torch.equal(dataset.data, payload["data"][:2, ..., 1:3])
    assert (
        dataset.data.untyped_storage().data_ptr()
        == payload["data"].untyped_storage().data_ptr()
    )


def test_irregular_selections_copy_the_selected_entries(payload):
    dataset = SpatioTemporalDataset(
        data_path=None, data=payload, channel_idxs=(2, 0), trajectories=[2, 0]
    )

    assert torch.equal(dataset.data, payload["data"][[2, 0]][..., [2, 0]])


def test_negative_channel_index_selects_from_the_end(payload):
    dataset = SpatioTemporalDataset(data_path=None, data=payload, channel_idxs=(-1,))

    assert torch.equal(dataset.data, payload["data"][..., [-1]])


def test_stride_must_be_positive(payload):
    with pytest.raises(ValueError, match="stride"):
        SpatioTemporalDataset(data_path=None, data=payload, stride=0)


def test_cartesian_dataset_metadata_uses_grid_type(cartesian_payload):
    dataset = ReactionDiffusionDataset(data_path=None, data=cartesian_payload)

    assert dataset.metadata.grid_type is GridType.CARTESIAN
