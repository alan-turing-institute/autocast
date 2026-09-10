from types import SimpleNamespace

import pytest
import torch

from autocast.data.datamodule import SpatioTemporalDataModule, TheWellDataModule


@pytest.mark.parametrize("full_trajectory_mode", [False, True])
def test_well_datamodule_separates_windowed_and_rollout_modes(
    monkeypatch, full_trajectory_mode
):
    monkeypatch.setattr("autocast.data.datamodule.TheWell", SimpleNamespace)

    dm = TheWellDataModule(
        well_dataset_name="rayleigh_benard",
        full_trajectory_mode=full_trajectory_mode,
        num_workers=0,
    )

    for dataset in (dm.train_dataset, dm.val_dataset, dm.test_dataset):
        assert dataset.full_trajectory_mode is full_trajectory_mode
    assert dm.rollout_val_dataset.full_trajectory_mode is True
    assert dm.rollout_test_dataset.full_trajectory_mode is True


def test_file_backed_rollout_datasets_apply_channel_idxs_once(tmp_path):
    payload = {"data": torch.randn(2, 6, 3, 4, 2)}
    for split in ("train", "valid", "test"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        torch.save(payload, split_dir / "data.pt")

    dm = SpatioTemporalDataModule(
        data_path=str(tmp_path),
        n_steps_input=2,
        n_steps_output=2,
        channel_idxs=(1,),
        ftype="torch",
    )

    assert dm.train_dataset.data.shape[-1] == 1
    assert dm.rollout_val_dataset.data.shape[-1] == 1
    assert dm.rollout_test_dataset.data.shape[-1] == 1
    assert dm.rollout_val_dataset.data is dm.train_dataset.data
    assert dm.rollout_test_dataset.data is dm.test_dataset.data
