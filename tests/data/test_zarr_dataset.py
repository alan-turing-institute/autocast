from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

zarr = pytest.importorskip("zarr")

from autocast.data.zarr_dataset import ZarrDataModule, ZarrDataset  # noqa: E402
from autocast.types import Batch  # noqa: E402


def _write_store(
    path: Path,
    values: np.ndarray,
    *,
    chunk_steps: int = 1,
    shard_steps: int = 4,
    complete: bool = True,
) -> None:
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.create_array(
        "data",
        data=values,
        chunks=(1, chunk_steps, *values.shape[2:]),
        shards=(1, shard_steps, *values.shape[2:]),
        compressors=None,
    )
    group.attrs.update(
        schema="autosim.spatiotemporal",
        schema_version=1,
        field_names=[f"field_{index}" for index in range(values.shape[-1])],
        provenance={"kind": "synthetic"},
        complete=complete,
    )


@pytest.fixture
def values() -> np.ndarray:
    return np.arange(2 * 20 * 3 * 4 * 3, dtype=np.float32).reshape(2, 20, 3, 4, 3)


@pytest.mark.parametrize(("chunks", "shards"), [(1, 4), (2, 8), (4, 12)])
def test_windows_cross_shards_without_changing_samples(
    tmp_path, values, chunks, shards
):
    path = tmp_path / "data.zarr"
    _write_store(path, values, chunk_steps=chunks, shard_steps=shards)
    dataset = ZarrDataset(path, n_steps_input=2, n_steps_output=5)
    assert len(dataset) == 2 * 14
    for index in range(len(dataset)):
        trajectory, start = divmod(index, 14)
        sample = dataset[index]
        torch.testing.assert_close(
            sample.input_fields, torch.from_numpy(values[trajectory, start : start + 2])
        )
        torch.testing.assert_close(
            sample.output_fields,
            torch.from_numpy(values[trajectory, start + 2 : start + 7]),
        )


def test_window_start_stride_and_frame_offset(tmp_path, values):
    path = tmp_path / "data.zarr"
    _write_store(path, values)
    dataset = ZarrDataset(
        path, n_steps_input=2, n_steps_output=3, stride=3, start_frame=2
    )
    assert len(dataset) == 10
    assert dataset.sample_trajectory_indices == [0] * 5 + [1] * 5
    assert dataset.sample_window_indices == list(range(5)) * 2
    torch.testing.assert_close(
        dataset[1].input_fields, torch.from_numpy(values[0, 5:7])
    )
    torch.testing.assert_close(
        dataset[-1].output_fields, torch.from_numpy(values[1, 16:19])
    )
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_initialization_is_metadata_only_and_reads_one_window(tmp_path, monkeypatch):
    path = tmp_path / "data.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.create_array(
        "data",
        shape=(1, 58440, 121, 240, 69),
        dtype="float32",
        chunks=(1, 1, 121, 240, 69),
        shards=(1, 32, 121, 240, 69),
        compressors=None,
    )
    group.attrs.update(
        schema="autosim.spatiotemporal",
        schema_version=1,
        field_names=[f"field_{index}" for index in range(69)],
        complete=True,
    )
    calls = []
    original = zarr.Array.__getitem__

    def read_window(array, selection):
        calls.append(selection)
        return original(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", read_window)
    dataset = ZarrDataset(path, n_steps_input=1, n_steps_output=1)
    assert calls == []
    assert dataset._array is None
    assert len(dataset) == 58439
    sample = dataset[32]
    assert sample.input_fields.shape == (1, 121, 240, 69)
    assert calls == [(0, slice(32, 34))]


def test_channel_selection_uses_training_stats(tmp_path, values):
    path = tmp_path / "valid" / "data.zarr"
    _write_store(path, values)
    stats = {
        "stats": {
            "mean": {"field_0": 1.0, "field_1": 10.0, "field_2": 100.0},
            "std": {"field_0": 2.0, "field_1": 4.0, "field_2": 5.0},
            "mean_delta": {f"field_{index}": 0.0 for index in range(3)},
            "std_delta": {f"field_{index}": 1.0 for index in range(3)},
        },
        "core_field_names": ["field_0", "field_1", "field_2"],
        "constant_field_names": [],
    }
    (tmp_path / "stats.yml").write_text(yaml.safe_dump(stats))
    dataset = ZarrDataset(
        path, channel_idxs=[2, 0], use_normalization=True, n_steps_input=2
    )
    expected = torch.from_numpy(values[0, :2])[..., [2, 0]]
    expected = (expected - torch.tensor([100.0, 1.0])) / torch.tensor([5.0, 2.0])
    torch.testing.assert_close(dataset[0].input_fields, expected)
    assert dataset.norm is not None
    assert dataset.norm.core_field_names == ["field_2", "field_0"]


def test_autoencoder_mode_reads_one_frame(tmp_path, values):
    path = tmp_path / "data.zarr"
    _write_store(path, values)
    dataset = ZarrDataset(path, autoencoder_mode=True)
    assert len(dataset) == 40
    sample = dataset[21]
    assert sample.input_fields is sample.output_fields
    torch.testing.assert_close(sample.input_fields, torch.from_numpy(values[1, 1:2]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_normalization_preserves_requested_dtype(tmp_path, values, dtype):
    path = tmp_path / "data.zarr"
    _write_store(path, values)
    means = {"field_0": 0.1, "field_1": 0.2, "field_2": 0.3}
    stds = {"field_0": 3.0, "field_1": 7.0, "field_2": 11.0}
    stats = {
        "stats": {
            "mean": means,
            "std": stds,
            "mean_delta": dict.fromkeys(means, 0.0),
            "std_delta": dict.fromkeys(means, 1.0),
        }
    }
    dataset = ZarrDataset(
        path,
        n_steps_input=2,
        dtype=dtype,
        use_normalization=True,
        normalization_stats=stats,
    )
    expected = (
        (torch.from_numpy(values[0, :3]) - torch.tensor(list(means.values())))
        / torch.tensor(list(stds.values()))
    ).to(dtype)
    sample = dataset[0]
    torch.testing.assert_close(sample.input_fields, expected[:2], rtol=0, atol=0)
    torch.testing.assert_close(sample.output_fields, expected[2:], rtol=0, atol=0)


def test_handles_reopen_after_serialization_or_process_change(tmp_path, values):
    path = tmp_path / "data.zarr"
    _write_store(path, values)
    dataset = ZarrDataset(path)
    expected = dataset[0].input_fields
    original_array = dataset._array
    clone = pickle.loads(pickle.dumps(dataset))
    assert clone._array is None
    torch.testing.assert_close(clone[0].input_fields, expected)
    dataset._pid = -1
    torch.testing.assert_close(dataset[0].input_fields, expected)
    assert dataset._array is not original_array


@pytest.mark.parametrize("num_workers", [0, 2])
def test_datamodule_returns_batches_and_bounded_rollouts(tmp_path, values, num_workers):
    for split in ("train", "valid", "test"):
        _write_store(tmp_path / split / "data.zarr", values)
    dm = ZarrDataModule(
        tmp_path,
        n_steps_input=2,
        n_steps_output=1,
        batch_size=2,
        num_workers=num_workers,
        use_normalization=False,
        rollout_steps=5,
        pin_memory=False,
        persistent_workers=False,
    )
    dm.setup("fit")
    assert dm.train_dataset is not None
    assert dm.train_dataset._array is None
    iterator = iter(dm.val_dataloader())
    batch = next(iterator)
    assert isinstance(batch, Batch)
    assert batch.input_fields.shape == (2, 2, 3, 4, 3)
    assert batch.output_fields.shape == (2, 1, 3, 4, 3)
    del iterator
    iterator = iter(dm.rollout_test_dataloader(batch_size=1))
    rollout = next(iterator)
    assert rollout.output_fields.shape == (1, 5, 3, 4, 3)
    del iterator


def test_rollout_window_stride_only_changes_sample_starts(tmp_path, values):
    _write_store(tmp_path / "test" / "data.zarr", values)
    dm = ZarrDataModule(
        tmp_path,
        n_steps_input=2,
        n_steps_output=1,
        rollout_steps=5,
        rollout_window_stride=3,
        use_normalization=False,
        pin_memory=False,
    )
    loader = dm.rollout_test_dataloader(batch_size=1)
    assert isinstance(loader.dataset, ZarrDataset)
    assert len(loader.dataset) == 10
    iterator = iter(loader)
    first, second = next(iterator), next(iterator)
    assert first.output_fields.shape == second.output_fields.shape == (1, 5, 3, 4, 3)
    torch.testing.assert_close(first.input_fields[0], torch.from_numpy(values[0, :2]))
    torch.testing.assert_close(second.input_fields[0], torch.from_numpy(values[0, 3:5]))
    torch.testing.assert_close(
        second.output_fields[0], torch.from_numpy(values[0, 5:10])
    )


@pytest.mark.parametrize(
    ("total_steps", "n_steps_input", "start_frame", "n_steps_output", "expected"),
    [(4, 1, 0, 1, 3), (6, 2, 1, 2, 3), (20, 2, 1, 1, 12)],
)
def test_rollout_horizon_uses_available_frames(
    tmp_path, values, total_steps, n_steps_input, start_frame, n_steps_output, expected
):
    _write_store(tmp_path / "test" / "data.zarr", values[:, :total_steps])
    dm = ZarrDataModule(
        tmp_path,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        start_frame=start_frame,
        use_normalization=False,
        pin_memory=False,
    )
    loader = dm.rollout_test_dataloader(batch_size=1)
    assert isinstance(loader.dataset, ZarrDataset)
    assert loader.dataset._array is None
    batch = next(iter(loader))
    target_start = start_frame + n_steps_input
    torch.testing.assert_close(
        batch.output_fields[0],
        torch.from_numpy(values[0, target_start : target_start + expected]),
    )


def test_rollout_horizon_must_fit_one_prediction_block(tmp_path):
    with pytest.raises(
        ValueError, match="rollout_steps must be at least n_steps_output"
    ):
        ZarrDataModule(tmp_path, n_steps_output=16, rollout_steps=12)


def test_autoencoder_does_not_require_a_rollout_horizon(tmp_path, values):
    _write_store(tmp_path / "test" / "data.zarr", values[:, :1])
    dm = ZarrDataModule(
        tmp_path,
        n_steps_output=16,
        autoencoder_mode=True,
        use_normalization=False,
        pin_memory=False,
    )
    batch = next(iter(dm.test_dataloader()))
    assert batch.output_fields.shape[1] == 1
    with pytest.raises(ValueError, match="unavailable in autoencoder mode"):
        dm.rollout_test_dataloader()


@pytest.mark.parametrize(
    "options",
    [
        {"stride": 0},
        {"n_steps_input": 0},
        {"n_steps_output": 0},
        {"start_frame": 20},
        {"channel_idxs": [3]},
        {"full_trajectory_mode": True},
    ],
)
def test_invalid_windows_fail_before_loading_values(tmp_path, values, options):
    path = tmp_path / "data.zarr"
    _write_store(path, values)
    with pytest.raises(
        ValueError,
        match=r"stride|n_steps_|too few|channel_idxs|Full-trajectory",
    ):
        ZarrDataset(path, **options)


def test_rejects_incomplete_store_and_unbounded_datamodule(tmp_path, values):
    path = tmp_path / "data.zarr"
    _write_store(path, values, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        ZarrDataset(path)
    with pytest.raises(ValueError, match="full_trajectory_mode"):
        ZarrDataModule(tmp_path, full_trajectory_mode=True)
