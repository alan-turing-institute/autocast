"""Tests for time-varying-scalar windowing in SpatioTemporalDataset."""

import pytest
import torch

from autocast.data.dataset import SpatioTemporalDataset


def _make_data(n_traj=1, t=12, w=2, h=2, c=1, c_tvs=3):
    data = torch.randn(n_traj, t, w, h, c)
    # Distinct, easily-checked values per (traj, time, channel).
    n_tvs = n_traj * t * c_tvs
    tvs = torch.arange(n_tvs, dtype=torch.float32).reshape(n_traj, t, c_tvs)
    return {"data": data, "time_varying_scalars": tvs}


def test_tvs_window_is_input_aligned_in_normal_mode():
    data = _make_data(t=12, c_tvs=3)
    n_steps_input, n_steps_output, stride = 2, 3, 1
    ds = SpatioTemporalDataset(
        data_path=None,
        data=data,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        stride=stride,
    )
    window = n_steps_input + n_steps_output
    n_windows = (12 - window) // stride + 1
    assert len(ds) == n_windows
    for i in range(n_windows):
        sample = ds[i]
        # Normal mode: TVS spans only the input window.
        assert sample.time_varying_scalars.shape == (n_steps_input, 3)
        start = i * stride
        expected = data["time_varying_scalars"][0, start : start + n_steps_input]
        assert torch.equal(sample.time_varying_scalars, expected)


def test_tvs_window_spans_input_plus_output_in_full_trajectory_mode():
    data = _make_data(t=12, c_tvs=3)
    n_steps_input = 2
    ds = SpatioTemporalDataset(
        data_path=None,
        data=data,
        n_steps_input=n_steps_input,
        n_steps_output=5,  # ignored; full-trajectory expands to T - n_steps_input
        full_trajectory_mode=True,
    )
    assert len(ds) == 1
    sample = ds[0]
    # Spans the whole trajectory so the rollout can stride forward.
    assert sample.time_varying_scalars.shape == (12, 3)
    assert torch.equal(sample.time_varying_scalars, data["time_varying_scalars"][0])


def test_subtrajectory_mode_windows_at_explicit_starts():
    data = _make_data(t=12, c_tvs=3)
    n_steps_input, n_steps_output = 2, 3
    starts = [0, 4, 7]
    window = n_steps_input + n_steps_output
    ds = SpatioTemporalDataset(
        data_path=None,
        data=data,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        subtrajectory_mode=True,
        subtrajectory_start_idxs=starts,
    )
    assert len(ds) == len(starts)
    for local_idx, start in enumerate(starts):
        sample = ds[local_idx]
        assert sample.input_fields.shape == (n_steps_input, 2, 2, 1)
        assert sample.output_fields.shape == (n_steps_output, 2, 2, 1)
        assert torch.equal(
            sample.input_fields, data["data"][0, start : start + n_steps_input]
        )
        assert torch.equal(
            sample.output_fields,
            data["data"][0, start + n_steps_input : start + window],
        )
        # Subtrajectory mode spans input + output for rollout striding.
        assert sample.time_varying_scalars.shape == (window, 3)
        assert torch.equal(
            sample.time_varying_scalars,
            data["time_varying_scalars"][0, start : start + window],
        )


def test_subtrajectory_mode_requires_start_idxs():
    data = _make_data()
    with pytest.raises(ValueError, match="subtrajectory_start_idxs"):
        SpatioTemporalDataset(
            data_path=None, data=data, subtrajectory_mode=True
        )


def test_subtrajectory_mode_rejects_out_of_range_start():
    data = _make_data(t=12)
    with pytest.raises(ValueError, match="out-of-range"):
        SpatioTemporalDataset(
            data_path=None,
            data=data,
            n_steps_input=2,
            n_steps_output=3,
            subtrajectory_mode=True,
            subtrajectory_start_idxs=[8],  # 8 + 5 = 13 > 12
        )


@pytest.mark.parametrize("other", ["full_trajectory_mode", "autoencoder_mode"])
def test_subtrajectory_mode_is_mutually_exclusive(other):
    data = _make_data()
    with pytest.raises(ValueError, match="cannot both be True"):
        SpatioTemporalDataset(
            data_path=None,
            data=data,
            subtrajectory_mode=True,
            subtrajectory_start_idxs=[0],
            **{other: True},
        )
