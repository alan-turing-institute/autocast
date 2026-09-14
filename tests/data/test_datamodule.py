import pytest
import torch

from autocast.data.datamodule import SpatioTemporalDataModule


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


def test_datamodule_forwards_independent_channel_selectors():
    data = torch.randn(2, 5, 3, 4, 3)
    dm = SpatioTemporalDataModule(
        data_path=None,
        data={split: {"data": data} for split in ("train", "valid", "test")},
        n_steps_input=2,
        n_steps_output=1,
        input_channel_idxs=(0, 2),
        output_channel_idxs=(1,),
        batch_size=2,
        num_workers=0,
    )

    batch = next(iter(dm.train_dataloader()))

    assert batch.input_fields.shape == (2, 2, 3, 4, 2)
    assert batch.output_fields.shape == (2, 1, 3, 4, 1)
    assert dm.rollout_test_dataset.input_channel_idxs == (0, 2)
    assert dm.rollout_test_dataset.output_channel_idxs == (1,)
