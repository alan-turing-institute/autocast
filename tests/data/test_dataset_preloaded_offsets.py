"""Frame offsets remain absolute when reusing preloaded dataset tensors."""

import h5py
import pytest
import torch

from autocast.data.dataset import SpatioTemporalDataset


def _dataset(start_frame: int = 0) -> SpatioTemporalDataset:
    return SpatioTemporalDataset(
        data_path=None,
        data={"data": torch.arange(8.0).reshape(1, 8, 1, 1, 1)},
        start_frame=start_frame,
    )


@pytest.mark.parametrize(("initial", "requested"), [(0, 2), (2, 4), (2, 2)])
def test_preloaded_data_honors_absolute_frame_offset(initial, requested):
    original = _dataset(initial)
    reused = SpatioTemporalDataset(
        data_path=None,
        data=original.to_preloaded_data(),
        start_frame=requested,
    )

    assert reused[0].input_fields.item() == requested
    assert reused[0].output_fields.item() == requested + 1
    assert reused.data.shape[1] == 8 - requested
    assert reused.start_frame == requested
    assert original[0].input_fields.item() == initial
    if initial == requested:
        assert reused.data is original.data


@pytest.mark.parametrize("requested", [0, 2])
def test_preloaded_data_rejects_unavailable_earlier_frames(requested):
    original = _dataset(3)

    with pytest.raises(ValueError, match="already starts at frame 3"):
        SpatioTemporalDataset(
            data_path=None, data=original.to_preloaded_data(), start_frame=requested
        )


@pytest.mark.parametrize("requested", [7, 8])
def test_preloaded_data_rejects_offsets_without_a_complete_sample(requested):
    original = _dataset(2)

    with pytest.raises(ValueError, match="start_frame"):
        SpatioTemporalDataset(
            data_path=None, data=original.to_preloaded_data(), start_frame=requested
        )


def test_preloaded_data_preserves_offset_across_repeated_reuse():
    dataset = _dataset()
    for start_frame in (2, 4, 4):
        dataset = SpatioTemporalDataset(
            data_path=None, data=dataset.to_preloaded_data(), start_frame=start_frame
        )
        assert dataset[0].input_fields.item() == start_frame
        assert dataset.data.shape[1] == 8 - start_frame


@pytest.mark.parametrize("suffix", ["pt", "h5"])
def test_saved_preloaded_data_preserves_frame_offset(tmp_path, suffix):
    payload = _dataset(2).to_preloaded_data()
    path = tmp_path / f"preloaded.{suffix}"
    if suffix == "pt":
        torch.save(payload, path)
    else:
        with h5py.File(path, "w") as handle:
            for name, value in payload.items():
                if value is not None:
                    handle.create_dataset(name, data=value)

    reused = SpatioTemporalDataset(data_path=str(path), start_frame=4)

    assert reused[0].input_fields.item() == 4
    assert reused[0].output_fields.item() == 5
    assert reused.data.shape[1] == 4
