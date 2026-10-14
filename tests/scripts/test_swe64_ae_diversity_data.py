"""Tiny tensor checks for the AE diversity view; no simulation or training."""

import copy
import json
import runpy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from autocast.data.dataset import SpatioTemporalDataset

ROOT = Path(__file__).resolve().parents[2]
HELPERS = runpy.run_path(str(ROOT / "scripts/swe64s/prepare_ae_diversity_data.py"))


def _save_payload(path, count, start):
    path.parent.mkdir(parents=True)
    values = torch.arange(count * 9 * 2 * 2 * 3, dtype=torch.float32).reshape(
        count, 9, 2, 2, 3
    )
    torch.save(
        {
            "data": values / 1000 + start,
            "constant_scalars": torch.full((count, 1), 0.1),
            "constant_fields": None,
        },
        path,
    )


@pytest.fixture
def inputs(tmp_path):
    baseline, additional = tmp_path / "baseline", tmp_path / "additional"
    baseline.mkdir()
    additional.mkdir()
    simulator = {
        "nx": 2,
        "ny": 2,
        "T": 2.0,
        "dt_save": 0.25,
        "skip_nt": 0,
        "forcing_energy_rate": 0.0015,
        "parameters_range": {"amp": [0.1, 0.1]},
    }
    first = {
        "simulator": simulator,
        "dataset": {"n_train": 2, "n_valid": 1, "n_test": 1},
        "seed": 0,
    }
    second = {
        "simulator": simulator,
        "dataset": {"n_train": 2, "n_valid": 0, "n_test": 0},
        "seed": 20261014,
    }
    OmegaConf.save(OmegaConf.create(first), baseline / "resolved_config.yaml")
    OmegaConf.save(OmegaConf.create(second), additional / "resolved_config.yaml")
    stats = {
        "core_field_names": ["h", "u", "v"],
        "constant_field_names": [],
        "stats": {
            "mean": dict.fromkeys(("h", "u", "v"), 0.0),
            "std": dict.fromkeys(("h", "u", "v"), 1.0),
            "mean_delta": dict.fromkeys(("h", "u", "v"), 0.0),
            "std_delta": dict.fromkeys(("h", "u", "v"), 1.0),
        },
    }
    OmegaConf.save(OmegaConf.create(stats), baseline / "stats.yml")
    for split, count, start in (("train", 2, 1), ("valid", 1, 3), ("test", 1, 4)):
        _save_payload(baseline / split / "data.pt", count, start)
    _save_payload(additional / "train/data.pt", 2, 5)
    contract = {
        "simulator": simulator,
        "splits": {"train": 2, "valid": 1, "test": 1},
        "frames": 9,
        "resolution": [2, 2],
        "field_names": ["h", "u", "v"],
        "saved_times": {"start": 0.0, "stop": 2.0, "interval": 0.25},
        "sha256": {},
    }
    for name in (
        "train/data.pt",
        "valid/data.pt",
        "test/data.pt",
        "stats.yml",
        "resolved_config.yaml",
    ):
        contract["sha256"][name] = HELPERS["_sha256"](baseline / name)
    return baseline, additional, tmp_path / "view", contract


def _prepare(inputs):
    baseline, additional, output, contract = inputs
    checked = HELPERS["_validate_sources"](baseline, additional, output, contract)
    return HELPERS["_write_view"](checked, contract, 4)


def test_even_frames_cover_whole_window():
    indices = HELPERS["selected_frame_indices"]()
    assert indices.shape == (160,)
    assert indices[0] == 0
    assert indices[-1] == 320
    assert indices.unique().numel() == 160
    assert set(indices.diff().tolist()) == {2, 3}


def test_materialized_view_preserves_sources_and_amp(inputs):
    baseline, additional, output, contract = inputs
    record = _prepare(inputs)
    assert record["shape"] == [4, 4, 2, 2, 3]
    assert record["source_frame_indices"] == [0, 3, 5, 8]
    assert record["source_saved_times"] == [0, 0.75, 1.25, 2.0]
    assert record["ae_only"] is True
    assert record["inputs_unchanged_after_write"] is True
    assert len(record["trajectory_mapping"]) == 4
    assert (output / "valid").resolve() == baseline / "valid"
    assert (output / "test").resolve() == baseline / "test"
    assert (output / "stats.yml").resolve() == baseline / "stats.yml"
    assert not (output / "train/.field-buffer.bin").exists()
    for name, digest in contract["sha256"].items():
        assert HELPERS["_sha256"](baseline / name) == digest
    dataset = SpatioTemporalDataset(
        data_path=str(output / "train/data.pt"),
        autoencoder_mode=True,
        use_normalization=True,
        normalization_path=str(output / "stats.yml"),
    )
    assert len(dataset) == 16
    first = dataset[0]
    assert torch.equal(first.input_fields, first.output_fields)
    assert first.constant_scalars is not None
    assert first.constant_scalars.item() == pytest.approx(0.1)
    saved = torch.load(output / "train/data.pt", weights_only=True)
    extra = torch.load(additional / "train/data.pt", weights_only=True)
    assert torch.equal(saved["data"][2], extra["data"][0, [0, 3, 5, 8]])
    assert (
        json.loads((output / "ae_diversity_manifest.json").read_text())["shape"]
        == record["shape"]
    )


def test_read_only_validation_writes_nothing(inputs):
    baseline, additional, output, contract = inputs
    record = HELPERS["_validate_sources"](baseline, additional, output, contract)
    assert not output.exists()
    assert len(record["retained_start_state_sha256"]["additional/train"]) == 2


@pytest.mark.parametrize("change", ["physics", "seed", "count"])
def test_config_changes_fail_before_writing(inputs, change):
    _, additional, output, _ = inputs
    cfg = OmegaConf.load(additional / "resolved_config.yaml")
    if change == "physics":
        cfg.simulator.forcing_energy_rate = 0.002
    elif change == "seed":
        cfg.seed = 0
    else:
        cfg.dataset.n_valid = 1
    OmegaConf.save(cfg, additional / "resolved_config.yaml")
    with pytest.raises(ValueError, match="Additional"):
        _prepare(inputs)
    assert not output.exists()


def test_existing_destination_and_broken_symlink_are_rejected(inputs):
    baseline, additional, output, contract = inputs
    output.symlink_to(output.parent / "absent")
    with pytest.raises(FileExistsError):
        HELPERS["_validate_sources"](baseline, additional, output, contract)
    assert output.is_symlink()


def test_nested_destination_is_rejected(inputs):
    baseline, additional, _, contract = inputs
    output = baseline / "new-view"
    with pytest.raises(ValueError, match="disjoint"):
        HELPERS["_validate_sources"](baseline, additional, output, contract)
    assert not output.exists()


@pytest.mark.parametrize(
    "change", ["duplicate_train", "duplicate_valid", "nonfinite", "amp", "shape"]
)
def test_bad_additional_tensors_fail_before_writing(inputs, change):
    baseline, additional, output, _ = inputs
    path = additional / "train/data.pt"
    payload = torch.load(path, weights_only=True)
    if change.startswith("duplicate"):
        split = "train" if change == "duplicate_train" else "valid"
        source = torch.load(baseline / split / "data.pt", weights_only=True)
        payload["data"][0, 0] = source["data"][0, 0]
    elif change == "nonfinite":
        payload["data"][0, 5, 0, 0, 0] = float("nan")
    elif change == "amp":
        payload["constant_scalars"][0, 0] = 0.2
    else:
        payload["data"] = payload["data"][:, :-1]
    torch.save(payload, path)
    with pytest.raises(
        ValueError, match=r"Duplicate|Nonfinite|Amp|Unexpected data shape"
    ):
        _prepare(inputs)
    assert not output.exists()


def test_modified_baseline_hash_fails_before_writing(inputs):
    baseline, _, output, _ = inputs
    with (baseline / "stats.yml").open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="Baseline hash mismatch"):
        _prepare(inputs)
    assert not output.exists()


def test_duplicate_starts_within_new_pool_are_rejected(inputs):
    _, additional, output, _ = inputs
    path = additional / "train/data.pt"
    payload = torch.load(path, weights_only=True)
    payload["data"][1, 0] = payload["data"][0, 0]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Duplicate retained start"):
        _prepare(inputs)
    assert not output.exists()


def test_disjoint_seed_range_check(inputs):
    baseline, additional, output, contract = inputs
    changed = copy.deepcopy(contract)
    cfg = OmegaConf.load(baseline / "resolved_config.yaml")
    cfg.seed = 20261000
    OmegaConf.save(cfg, baseline / "resolved_config.yaml")
    changed["sha256"]["resolved_config.yaml"] = HELPERS["_sha256"](
        baseline / "resolved_config.yaml"
    )
    with pytest.raises(ValueError, match="seed range overlaps"):
        HELPERS["_validate_sources"](baseline, additional, output, changed)
    assert not output.exists()


def test_autosim_empty_valid_test_splits_are_accepted(inputs):
    baseline, additional, output, contract = inputs
    for split in ("valid", "test"):
        path = additional / split / "data.pt"
        path.parent.mkdir()
        torch.save(
            {
                "data": torch.empty((0, 0, 2, 2, 3)),
                "constant_scalars": torch.empty((0, 1)),
                "constant_fields": None,
            },
            path,
        )
    checked = HELPERS["_validate_sources"](baseline, additional, output, contract)
    assert str(additional / "valid/data.pt") in checked["input_sha256"]
    assert str(additional / "test/data.pt") in checked["input_sha256"]
    assert not output.exists()


def test_unexpected_nonempty_additional_split_is_rejected(inputs):
    _, additional, output, _ = inputs
    _save_payload(additional / "valid/data.pt", 1, 9)
    with pytest.raises(ValueError, match="Unexpected data shape"):
        _prepare(inputs)
    assert not output.exists()


def test_preflight_validates_materialized_view_without_writes(inputs):
    _, _, output, contract = inputs
    record = _prepare(inputs)
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.rglob("*")
    }
    assert HELPERS["_validate_data_view"](output, contract, 4) == record
    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.rglob("*")
    }
    assert after == before


@pytest.mark.parametrize("change", ["train", "mapping", "stats", "valid_link"])
def test_preflight_rejects_changed_view(inputs, change):
    baseline, additional, output, contract = inputs
    _prepare(inputs)
    if change == "train":
        path = output / "train/data.pt"
        payload = torch.load(path, weights_only=True)
        payload["data"][0, 0, 0, 0, 0] += 1
        torch.save(payload, path)
        message = "View hash mismatch"
    elif change == "mapping":
        path = output / "ae_diversity_manifest.json"
        record = json.loads(path.read_text())
        record["trajectory_mapping"][0]["source_trajectory"] = 1
        path.write_text(json.dumps(record))
        message = "source trajectory identities"
    elif change == "stats":
        with (baseline / "stats.yml").open("a") as stream:
            stream.write("\n# changed\n")
        message = "Baseline hash mismatch"
    else:
        (output / "valid").unlink()
        (output / "valid").symlink_to(additional / "train", target_is_directory=True)
        message = "unchanged baseline split"
    with pytest.raises(ValueError, match=message):
        HELPERS["_validate_data_view"](output, contract, 4)
