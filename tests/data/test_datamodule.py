import pytest
import torch

from autocast.data.datamodule import SpatioTemporalDataModule


@pytest.mark.parametrize("pin_memory", [False, True])
def test_rollout_valid_dataloader_respects_pin_memory(pin_memory):
    payload = {"data": torch.randn(2, 4, 3, 4, 1)}
    dm = SpatioTemporalDataModule(
        data_path=None,
        data=dict.fromkeys(("train", "valid", "test"), payload),
        num_workers=0,
        pin_memory=pin_memory,
    )

    assert dm.rollout_valid_dataloader().pin_memory is pin_memory


@pytest.mark.parametrize("start_frame", [0, 2, 4])
def test_file_backed_rollout_datasets_apply_slices_once(tmp_path, start_frame):
    payload = {"data": torch.randn(2, 6, 3, 4, 2)}
    for split in ("train", "valid", "test"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        torch.save(payload, split_dir / "data.pt")

    dm = SpatioTemporalDataModule(
        data_path=str(tmp_path),
        n_steps_input=1,
        n_steps_output=1,
        channel_idxs=(1,),
        start_frame=start_frame,
        ftype="torch",
    )

    assert dm.train_dataset.data.shape[-1] == 1
    assert dm.rollout_val_dataset.data.shape[-1] == 1
    assert dm.rollout_test_dataset.data.shape[-1] == 1
    assert dm.rollout_val_dataset.data is dm.train_dataset.data
    assert dm.rollout_test_dataset.data is dm.test_dataset.data
    assert torch.equal(dm.test_dataset.data, payload["data"][:, start_frame:, :, :, 1:])
    assert torch.equal(
        dm.rollout_test_dataset[0].input_fields,
        payload["data"][0, start_frame : start_frame + 1, :, :, 1:],
    )
    assert torch.equal(
        dm.rollout_test_dataset[0].output_fields,
        payload["data"][0, start_frame + 1 :, :, :, 1:],
    )
