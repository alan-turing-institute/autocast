import torch

from autocast.data.dataset import SpatioTemporalDataset


def _dataset(*, full_trajectory_mode: bool = False) -> SpatioTemporalDataset:
    data = torch.arange(2 * 8 * 2 * 2, dtype=torch.float32).reshape(2, 8, 2, 2, 1)
    return SpatioTemporalDataset(
        data_path=None,
        data={"data": data},
        n_steps_input=1,
        n_steps_output=2,
        stride=2,
        full_trajectory_mode=full_trajectory_mode,
    )


def test_windowed_dataset_records_trajectory_and_window_indices():
    dataset = _dataset()

    assert dataset.sample_trajectory_indices == [0, 0, 0, 1, 1, 1]
    assert dataset.sample_window_indices == [0, 1, 2, 0, 1, 2]
    assert len(dataset.sample_trajectory_indices) == len(dataset)
    assert len(dataset.sample_window_indices) == len(dataset)


def test_full_trajectory_dataset_has_one_index_per_trajectory():
    dataset = _dataset(full_trajectory_mode=True)

    assert dataset.sample_trajectory_indices == [0, 1]
    assert dataset.sample_window_indices == [0, 0]
