"""Check sparse validation coverage and the independent historical test path."""

import pytest
import torch

from autocast.data.datamodule import SpatioTemporalDataModule


def _payload(n_trajectories, offset=0):
    return {
        "data": torch.arange(n_trajectories * 21 * 4, dtype=torch.float32).reshape(
            n_trajectories, 21, 2, 2, 1
        )
        + offset,
        "constant_scalars": None,
        "constant_fields": None,
    }


@pytest.mark.parametrize(
    ("validation_stride", "starts"),
    [(None, tuple(range(17))), (5, (0, 5, 10, 15)), (10, (0, 10))],
)
def test_validation_thins_windows_across_every_trajectory(validation_stride, starts):
    data = {"train": _payload(4), "valid": _payload(6), "test": _payload(2)}
    dm = SpatioTemporalDataModule(
        data_path=None,
        data=data,
        n_steps_output=4,
        validation_stride=validation_stride,
        num_workers=0,
    )
    assert len(dm.train_dataset) == 4 * 17
    assert len(dm.test_dataset) == 2 * 17
    assert len(dm.val_dataset) == 6 * len(starts)
    assert set(dm.val_dataset.sample_trajectory_indices) == set(range(6))
    first_trajectory = [dm.val_dataset[i] for i in range(len(starts))]
    for sample, start in zip(first_trajectory, starts, strict=True):
        assert torch.equal(
            sample.input_fields, data["valid"]["data"][0, start : start + 1]
        )
        assert torch.equal(
            sample.output_fields, data["valid"]["data"][0, start + 1 : start + 5]
        )


@pytest.mark.parametrize("validation_stride", [5, 10])
def test_autoencoder_validation_thins_frames_across_all_trajectories(validation_stride):
    dm = SpatioTemporalDataModule(
        data_path=None,
        data={"train": _payload(4), "valid": _payload(6), "test": _payload(2)},
        autoencoder_mode=True,
        validation_stride=validation_stride,
        num_workers=0,
    )
    assert len(dm.train_dataset) == 4 * 21
    assert len(dm.val_dataset) == 6 * len(range(0, 21, validation_stride))
    assert set(dm.val_dataset.sample_trajectory_indices) == set(range(6))


def test_original_test_override_applies_to_windows_and_rollouts(tmp_path):
    generated = tmp_path / "generated"
    for split in ("train", "valid", "test"):
        (generated / split).mkdir(parents=True)
        torch.save(_payload(6, offset=1000), generated / split / "data.pt")
    original_test = tmp_path / "original_test.pt"
    original = _payload(2, offset=-1000)
    torch.save(original, original_test)
    dm = SpatioTemporalDataModule(
        data_path=str(generated),
        test_data_path=str(original_test),
        n_steps_output=4,
        validation_stride=10,
        num_workers=0,
    )
    assert dm.train_dataset.n_trajectories == 6
    assert dm.test_dataset.n_trajectories == 2
    assert dm.rollout_test_dataset.n_trajectories == 2
    assert dm.test_dataset.stride == 1
    assert torch.equal(dm.test_dataset.data, original["data"])
    assert torch.equal(dm.rollout_test_dataset.data, original["data"])
