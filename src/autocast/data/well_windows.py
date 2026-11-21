"""Resolve fixed The Well test windows against their source trajectories."""

from bisect import bisect_right
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from autocast.data.transforms import LogScalars


def validate_well_window_selection(
    dataset: Any, selection: Mapping[str, Any]
) -> list[int]:
    """Reject changed file ordering, frame counts, strides, or window identities."""
    well = dataset.well_dataset
    dataset_name = selection.get("dataset")
    if dataset_name is not None and dataset_name != well.metadata.dataset_name:
        raise ValueError(f"Paired test selection requires {dataset_name}")
    fields = selection.get("fields")
    if fields is not None:
        actual_fields = [
            field for group in well.metadata.field_names.values() for field in group
        ]
        if list(fields) != actual_fields:
            msg = "Test field order differs from the saved selection"
            raise ValueError(msg)
    expected_files = list(selection["files"])
    actual_files = [Path(path).name for path in well.files_paths]
    if actual_files != expected_files:
        msg = "Test file inventory/order differs from the saved selection"
        raise ValueError(msg)
    if list(well.n_trajectories_per_file) != list(selection["files"].values()):
        msg = "Test trajectory counts differ from the saved selection"
        raise ValueError(msg)
    stride = selection.get("stride", 1)
    if (
        well.n_steps_input,
        well.n_steps_output,
        well.min_dt_stride,
        well.max_dt_stride,
    ) != (1, 4, stride, stride):
        raise ValueError(
            f"Paired evaluation requires 1-to-4 windows at stride {stride}"
        )
    frames = selection["frames_per_trajectory"]
    starts = frames - 4 * stride
    if any(n != frames for n in well.n_steps_per_trajectory) or any(
        n != starts for n in well.n_windows_per_trajectory
    ):
        msg = "Frame/window counts differ from the saved selection"
        raise ValueError(msg)
    if well.transform is not None and not isinstance(well.transform, LogScalars):
        msg = "Paired test windows must not use random augmentation"
        raise ValueError(msg)
    identities = [
        (row["file"], row["trajectory_in_file"], row["context_frame"])
        for row in selection["windows"]
    ]
    windows_per_trajectory = selection["windows_per_trajectory"]
    expected_trajectories = selection.get("trajectory_count", 64)
    expected_windows = expected_trajectories * windows_per_trajectory
    if len(identities) != expected_windows or len(set(identities)) != expected_windows:
        raise ValueError(f"Expected exactly {expected_windows} distinct windows")
    counts = {}
    indices = []
    for row in selection["windows"]:
        file_idx = expected_files.index(row["file"])
        trajectory = (
            sum(well.n_trajectories_per_file[:file_idx]) + row["trajectory_in_file"]
        )
        expected = (
            max(int(well.file_index_offsets[file_idx]), 0)
            + row["trajectory_in_file"] * starts
            + row["context_frame"]
        )
        if not (
            0 <= row["trajectory_in_file"] < well.n_trajectories_per_file[file_idx]
            and 0 <= row["context_frame"] < starts
            and bisect_right(well.file_index_offsets, expected) - 1 == file_idx
            and row["trajectory_idx"] == trajectory
        ):
            raise ValueError(f"Window identity mismatch: {row}")
        indices.append(expected)
        counts[trajectory] = counts.get(trajectory, 0) + 1
    if len(counts) != expected_trajectories or set(counts.values()) != {
        windows_per_trajectory
    }:
        raise ValueError(f"Expected {windows_per_trajectory} windows per trajectory")
    return indices
