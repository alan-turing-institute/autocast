import torch

from autocast.data.datamodule import SpatioTemporalDataModule


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
