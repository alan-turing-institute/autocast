"""Checks for the RB experiment's frozen identities and physical scoring."""

import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from autocast.data.transforms import LogScalars

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "slurm_scripts/comparison/the_well/rayleigh_benard/paired_test_windows.py"
)
spec = importlib.util.spec_from_file_location("rb_paired_windows", SCRIPT)
assert spec is not None
assert spec.loader is not None
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def selection_and_dataset():
    source = yaml.safe_load((SCRIPT.parent / "test64_seed0.yaml").read_text())
    selection = paired.make_selection(source)
    counts = list(source["files"].values())
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count * 196)
    well = SimpleNamespace(
        files_paths=list(source["files"]),
        n_trajectories_per_file=counts,
        n_steps_input=1,
        n_steps_output=4,
        min_dt_stride=1,
        max_dt_stride=1,
        n_steps_per_trajectory=[200] * len(counts),
        n_windows_per_trajectory=[196] * len(counts),
        file_index_offsets=offsets,
        transform=None,
    )
    return selection, SimpleNamespace(well_dataset=well)


def test_frozen_windows_cover_selected_trajectories_without_replacement():
    selection, dataset = selection_and_dataset()
    indices = paired.validate_selection(dataset, selection)
    assert len(indices) == len(set(indices)) == 512
    assert set(
        Counter(row["trajectory_idx"] for row in selection["windows"]).values()
    ) == {8}
    source = yaml.safe_load((SCRIPT.parent / "test64_seed0.yaml").read_text())
    assert selection == paired.make_selection(source)
    assert selection != paired.make_selection(source, seed=1)


def test_frozen_identity_rejects_reordered_files_and_changed_context():
    selection, dataset = selection_and_dataset()
    dataset.well_dataset.files_paths.reverse()
    with pytest.raises(ValueError, match="inventory/order"):
        paired.validate_selection(dataset, selection)
    dataset.well_dataset.files_paths.reverse()
    selection["windows"][0]["context_frame"] = 196
    with pytest.raises(ValueError, match="identity mismatch"):
        paired.validate_selection(dataset, selection)


def test_deterministic_scalar_transform_preserves_window_selection():
    selection, dataset = selection_and_dataset()
    dataset.well_dataset.transform = LogScalars(scalar_names=["Rayleigh", "Prandtl"])
    assert len(paired.validate_selection(dataset, selection)) == 512
    dataset.well_dataset.transform = lambda x: x
    with pytest.raises(ValueError, match="random augmentation"):
        paired.validate_selection(dataset, selection)


def test_first_file_sentinel_keeps_physical_window_indices():
    selection, dataset = selection_and_dataset()
    reference = paired.validate_selection(dataset, selection)
    dataset.well_dataset.file_index_offsets[0] = -1
    assert paired.validate_selection(dataset, selection) == reference


def test_test_prediction_scores_physical_outputs_and_truth():
    batch = SimpleNamespace(output_fields=torch.zeros(1, 4, 8, 4))

    class Forecast:
        def predict_step(self, batch, batch_idx):
            assert batch_idx == 0
            return torch.full((*batch.output_fields.shape, 16), 12.0)

        def denormalize_tensor(self, value):
            return 3 * value + 10

        def __call__(self, _):
            msg = "Normalized forward output must not be scored"
            raise AssertionError(msg)

    pred, truth = paired.physical_prediction(Forecast(), batch)
    assert pred.shape == (*truth.shape, 16)
    assert torch.all(pred == 12)
    assert torch.all(truth == 10)
