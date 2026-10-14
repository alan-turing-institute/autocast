"""Build an AE-only, 400-trajectory view without generating simulator data.

The original 200 trajectories and an independently generated set of 200 each
contribute the same 160 selected saved frames. Validation, test and training
normalization remain references to the original dataset. The selected times
are not uniformly spaced, so this view must not be used for forecasting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "local_hydra/local_experiment/swe64s/dataset.json"
ADDITIONAL_SEED = 20261014
KEPT_FRAMES = 160
# Recorded in the frozen reconstruction audit; do not recompute normalization.
BASELINE_STATS_SHA256 = (
    "c04afa893ba3dffdadb1916161a1caeab833884ce4a27126825b32108e246feb"
)
BASELINE_CONFIG_SHA256 = (
    "906539c1b944f0657c9fe7f7c2dc4acb2964458e1336b8d2b8060820c2b82244"
)
_CHECKS = runpy.run_path(str(Path(__file__).with_name("check_inputs.py")))
_require = _CHECKS["_require"]
_sha256 = _CHECKS["_sha256"]


def _contract() -> dict[str, Any]:
    contract = json.loads(MANIFEST_PATH.read_text())
    contract["sha256"].update(
        {
            "stats.yml": BASELINE_STATS_SHA256,
            "resolved_config.yaml": BASELINE_CONFIG_SHA256,
        }
    )
    return contract


def selected_frame_indices(frames: int = 321, keep: int = KEPT_FRAMES) -> torch.Tensor:
    """Select distinct nearest-integer, evenly spaced indices, including endpoints."""
    _require(
        2 <= keep <= frames, "Need at least two selected frames, no more than available"
    )
    indices = torch.linspace(0, frames - 1, keep, dtype=torch.float64).round().long()
    _require(indices.unique().numel() == keep, "Selected frame indices must be unique")
    return indices


def _validate_paths(
    baseline: Path, additional: Path, output: Path
) -> tuple[Path, Path, Path]:
    original_output = output.expanduser().absolute()
    if os.path.lexists(original_output):
        msg = f"Refusing an existing output path, including symlinks: {original_output}"
        raise FileExistsError(msg)
    baseline, additional = (
        baseline.expanduser().resolve(),
        additional.expanduser().resolve(),
    )
    output = original_output.resolve()
    _require(
        baseline.is_dir() and additional.is_dir(), "Both source datasets must exist"
    )
    _require(output.parent.is_dir(), "The output parent directory must already exist")
    paths = (baseline, additional, output)
    for i, first in enumerate(paths):
        for second in paths[i + 1 :]:
            _require(
                first != second
                and first not in second.parents
                and second not in first.parents,
                "Source and output dataset directories must be disjoint, not nested",
            )
    return baseline, additional, output


def _load_config(path: Path) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        msg = f"Expected a string-keyed config mapping: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], config)


def _validate_configs(
    baseline: Path, additional: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = _load_config(baseline / "resolved_config.yaml")
    second = _load_config(additional / "resolved_config.yaml")
    _require(
        isinstance(first.get("simulator"), dict)
        and first["simulator"] == second.get("simulator"),
        "Additional simulator settings must exactly match the baseline, "
        "including runtime settings",
    )
    for key, expected in contract["simulator"].items():
        _require(
            first["simulator"].get(key) == expected,
            f"Unexpected baseline simulator setting: {key}",
        )
    for split, count in contract["splits"].items():
        _require(
            first.get("dataset", {}).get(f"n_{split}") == count,
            f"Unexpected baseline {split} count",
        )
        expected_additional = contract["splits"]["train"] if split == "train" else 0
        _require(
            second.get("dataset", {}).get(f"n_{split}") == expected_additional,
            "Additional data must contain only the approved number "
            "of training trajectories",
        )
    _require(
        second.get("seed") == ADDITIONAL_SEED,
        f"Additional data seed must be {ADDITIONAL_SEED}",
    )
    original_seed = first.get("seed")
    if not isinstance(original_seed, int) or isinstance(original_seed, bool):
        msg = "Baseline seed must be recorded"
        raise ValueError(msg)
    # AutoSim reserves max(100,20*n)+1 possible initial/retry seeds per split.
    stride = max(100, 20 * max(contract["splits"].values())) + 1
    extra_stride = max(100, 20 * contract["splits"]["train"]) + 1
    _require(
        original_seed >= ADDITIONAL_SEED + extra_stride
        or original_seed + 3 * stride <= ADDITIONAL_SEED,
        "Additional initial/retry seed range overlaps the original split ranges",
    )
    return first, second


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    _require(isinstance(payload, dict), f"Expected a tensor payload mapping: {path}")
    return payload


def _check_payload(
    path: Path,
    count: int,
    contract: dict[str, Any],
    seen_starts: dict[str, str],
) -> list[str]:
    payload = _load_payload(path)
    fields, scalars = payload.get("data"), payload.get("constant_scalars")
    # AutoSim writes zero temporal extent as well for an empty split.
    frames = contract["frames"] if count else 0
    expected = (count, frames, *contract["resolution"], 3)
    if not isinstance(fields, torch.Tensor) or tuple(fields.shape) != expected:
        msg = f"Unexpected data shape: {path}"
        raise ValueError(msg)
    _require(fields.dtype == torch.float32, f"Expected float32 fields: {path}")
    if not isinstance(scalars, torch.Tensor) or tuple(scalars.shape) != (count, 1):
        msg = f"Missing amp metadata: {path}"
        raise ValueError(msg)
    _require(
        torch.isfinite(scalars).all().item()
        and torch.allclose(scalars, torch.full_like(scalars, 0.1)),
        f"Amp must remain fixed at 0.1: {path}",
    )
    for key, value in payload.items():
        _require(
            key in {"data", "constant_scalars"} or value is None,
            f"Unexpected nonempty payload field {key}: {path}",
        )
    hashes = []
    for trajectory in range(count):
        _require(
            torch.isfinite(fields[trajectory]).all().item(),
            f"Nonfinite fields: {path}, trajectory {trajectory}",
        )
        # Retained frame zero is t=40, not the unsaved initial condition at t=0.
        digest = hashlib.sha256(
            fields[trajectory, 0].contiguous().numpy().tobytes()
        ).hexdigest()
        identity = f"{path}:{trajectory}"
        _require(
            digest not in seen_starts,
            f"Duplicate retained start state: {identity} and {seen_starts.get(digest)}",
        )
        seen_starts[digest] = identity
        hashes.append(digest)
    return hashes


def _validate_sources(
    baseline: Path, additional: Path, output: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    baseline, additional, output = _validate_paths(baseline, additional, output)
    first, second = _validate_configs(baseline, additional, contract)
    source_files = {
        "baseline": [baseline / name for name in contract["sha256"]],
        "additional": [
            additional / "resolved_config.yaml",
            additional / "train/data.pt",
        ],
    }
    for split in ("valid", "test"):
        path = additional / split / "data.pt"
        if path.exists():
            source_files["additional"].append(path)
    all_paths = [path for paths in source_files.values() for path in paths]
    _require(
        len({path.resolve() for path in all_paths}) == len(all_paths),
        "Source files alias one another",
    )
    hashes = {str(path): _sha256(path) for path in all_paths}
    for name, expected_hash in contract["sha256"].items():
        _require(
            hashes[str(baseline / name)] == expected_hash,
            f"Baseline hash mismatch: {name}",
        )
    # Reuse the existing SWE field-order, normalization, shape and amp checks.
    _CHECKS["check_dataset"](baseline, contract, verify_hashes=False)
    seen_starts: dict[str, str] = {}
    retained_start_hashes = {}
    for split, count in contract["splits"].items():
        retained_start_hashes[f"baseline/{split}"] = _check_payload(
            baseline / split / "data.pt", count, contract, seen_starts
        )
    retained_start_hashes["additional/train"] = _check_payload(
        additional / "train/data.pt", contract["splits"]["train"], contract, seen_starts
    )
    for split in ("valid", "test"):
        path = additional / split / "data.pt"
        if path.exists():
            _check_payload(path, 0, contract, seen_starts)
    return {
        "baseline": str(baseline),
        "additional": str(additional),
        "output": str(output),
        "baseline_config": first,
        "additional_config": second,
        "input_sha256": hashes,
        "retained_start_state_sha256": retained_start_hashes,
    }


def validate_sources(baseline: Path, additional: Path, output: Path) -> dict[str, Any]:
    """Read-only production preflight; reject changed physics, seeds or source data."""
    return _validate_sources(baseline, additional, output, _contract())


def _write_view(
    checked: dict[str, Any], contract: dict[str, Any], keep: int
) -> dict[str, Any]:
    baseline, additional, output = (
        Path(checked[key]) for key in ("baseline", "additional", "output")
    )
    indices = selected_frame_indices(contract["frames"], keep)
    count = contract["splits"]["train"]
    shape = (2 * count, keep, *contract["resolution"], 3)
    # Validation is complete before the first write. mkdir never reuses a destination.
    output.mkdir()
    (output / "train").mkdir()
    buffer_path = output / "train/.field-buffer.bin"
    buffer = torch.from_file(
        str(buffer_path), shared=True, size=math.prod(shape), dtype=torch.float32
    )
    fields = buffer.view(shape)
    scalar_parts = []
    mapping = []
    for pool, source in enumerate((baseline, additional)):
        payload = _load_payload(source / "train/data.pt")
        scalar_parts.append(payload["constant_scalars"].clone())
        for trajectory in range(count):
            destination = pool * count + trajectory
            fields[destination].copy_(
                payload["data"][trajectory].index_select(0, indices)
            )
            mapping.append(
                {
                    "output_trajectory": destination,
                    "source_dataset": str(source),
                    "source_split": "train",
                    "source_trajectory": trajectory,
                }
            )
    output_payload = {
        "data": fields,
        "constant_scalars": torch.cat(scalar_parts),
        "constant_fields": None,
    }
    train_path = output / "train/data.pt"
    with train_path.open("xb") as stream:
        torch.save(output_payload, stream)
    del output_payload, fields, buffer
    buffer_path.unlink()
    for split in ("valid", "test"):
        (output / split).symlink_to(baseline / split, target_is_directory=True)
    (output / "stats.yml").symlink_to(baseline / "stats.yml")

    # Check every copied trajectory against its source, not just tensor dimensions.
    saved = _load_payload(train_path)
    for pool, source in enumerate((baseline, additional)):
        payload = _load_payload(source / "train/data.pt")
        for trajectory in range(count):
            destination = pool * count + trajectory
            _require(
                torch.equal(
                    saved["data"][destination],
                    payload["data"][trajectory].index_select(0, indices),
                ),
                "Saved selected fields differ from the source",
            )
            _require(
                torch.equal(
                    saved["constant_scalars"][destination],
                    payload["constant_scalars"][trajectory],
                ),
                "Saved amp differs from the source",
            )
    for source, digest in checked["input_sha256"].items():
        _require(
            _sha256(Path(source)) == digest,
            f"Source changed during view preparation: {source}",
        )
    start, interval = (
        contract["saved_times"]["start"],
        contract["saved_times"]["interval"],
    )
    manifest = checked | {
        "ae_only": True,
        "scope": "Materialized AE view; no simulator generation or training",
        "shape": list(shape),
        "field_names": list(contract["field_names"]),
        "source_frame_indices": indices.tolist(),
        "source_saved_times": [start + interval * i for i in indices.tolist()],
        "trajectory_mapping": mapping,
        "baseline_frame_count": count * contract["frames"],
        "view_frame_count": 2 * count * keep,
        "normalization": "Unchanged baseline training stats; never recomputed",
        "valid_test": "Unchanged baseline splits referenced by directory symlinks",
        "temporal_warning": (
            "Selected times have nonuniform gaps; single-frame reconstruction only, "
            "not forecasting"
        ),
        "start_hash_warning": (
            "Duplicate checks use the first retained state, "
            "not an unsaved t=0 initial condition"
        ),
        "inputs_unchanged_after_write": True,
        "output_sha256": {
            "train/data.pt": _sha256(train_path),
            "stats.yml": _sha256(output / "stats.yml"),
        },
        "builder_sha256": _sha256(Path(__file__)),
    }
    view_config = {
        "ae_only": True,
        "simulator": checked["baseline_config"]["simulator"],
        "dataset": {
            "n_train": 2 * count,
            "n_valid": contract["splits"]["valid"],
            "n_test": contract["splits"]["test"],
            "output_dir": str(output),
        },
        "view": {
            "source_frame_indices": indices.tolist(),
            "source_saved_times": manifest["source_saved_times"],
            "normalization_path": str(baseline / "stats.yml"),
            "provenance": "ae_diversity_manifest.json",
        },
    }
    OmegaConf.save(OmegaConf.create(view_config), output / "resolved_config.yaml")
    manifest["output_sha256"]["resolved_config.yaml"] = _sha256(
        output / "resolved_config.yaml"
    )
    with (output / "ae_diversity_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def _validate_data_view(
    output: Path, contract: dict[str, Any], keep: int
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    manifest = json.loads((output / "ae_diversity_manifest.json").read_text())
    _require(isinstance(manifest, dict), "Expected a view manifest mapping")
    baseline, additional = (Path(manifest[key]) for key in ("baseline", "additional"))
    _require(str(output) == manifest["output"], "View location differs from manifest")
    _require(manifest.get("ae_only") is True, "Not an AE-only data view")
    _require(
        manifest.get("inputs_unchanged_after_write") is True,
        "View did not record immutable source verification",
    )
    count = contract["splits"]["train"]
    shape = [2 * count, keep, *contract["resolution"], 3]
    _require(manifest.get("shape") == shape, "Unexpected view shape metadata")
    _require(
        manifest.get("field_names") == contract["field_names"],
        "Unexpected view field order",
    )
    indices = selected_frame_indices(contract["frames"], keep).tolist()
    times = [
        contract["saved_times"]["start"] + contract["saved_times"]["interval"] * i
        for i in indices
    ]
    _require(
        manifest.get("source_frame_indices") == indices
        and manifest.get("source_saved_times") == times,
        "Unexpected selected frames or saved times",
    )
    _require(
        manifest.get("baseline_frame_count") == count * contract["frames"]
        and manifest.get("view_frame_count") == 2 * count * keep,
        "Unexpected frame counts",
    )
    expected_mapping = [
        {
            "output_trajectory": pool * count + trajectory,
            "source_dataset": str(source),
            "source_split": "train",
            "source_trajectory": trajectory,
        }
        for pool, source in enumerate((baseline, additional))
        for trajectory in range(count)
    ]
    _require(
        manifest.get("trajectory_mapping") == expected_mapping,
        "Unexpected source trajectory identities or ordering",
    )
    retained_starts = manifest.get("retained_start_state_sha256", {})
    expected_counts = {
        f"baseline/{split}": size for split, size in contract["splits"].items()
    } | {"additional/train": count}
    _require(set(retained_starts) == set(expected_counts), "Missing start-state hashes")
    all_starts = []
    for name, expected_count in expected_counts.items():
        digests = retained_starts[name]
        _require(
            len(digests) == expected_count
            and all(
                isinstance(digest, str) and len(digest) == 64 for digest in digests
            ),
            f"Unexpected start-state hashes: {name}",
        )
        all_starts.extend(digests)
    _require(len(set(all_starts)) == len(all_starts), "Duplicate source start states")

    input_hashes = manifest["input_sha256"]
    for name, expected_hash in contract["sha256"].items():
        _require(
            input_hashes.get(str(baseline / name)) == expected_hash,
            f"Manifest baseline hash differs from the pinned source: {name}",
        )
        # Original training tensors were checked before/after materialization.
        # Do not reread another 3 GB at every training preflight.
        if name != "train/data.pt":
            _require(
                _sha256(baseline / name) == expected_hash,
                f"Baseline hash mismatch: {name}",
            )
    for split in ("valid", "test"):
        link = output / split
        _require(
            link.is_symlink() and link.resolve() == (baseline / split).resolve(),
            f"View {split} must reference the unchanged baseline split",
        )
    stats = output / "stats.yml"
    _require(
        stats.is_symlink() and stats.resolve() == (baseline / "stats.yml").resolve(),
        "View stats must reference unchanged baseline normalization",
    )
    first, second = _validate_configs(baseline, additional, contract)
    _require(
        first == manifest["baseline_config"]
        and second == manifest["additional_config"],
        "Source configuration differs from view provenance",
    )
    _require(
        _sha256(additional / "resolved_config.yaml")
        == input_hashes[str(additional / "resolved_config.yaml")],
        "Additional config hash mismatch",
    )
    output_hashes = manifest.get("output_sha256", {})
    _require(
        set(output_hashes) == {"train/data.pt", "stats.yml", "resolved_config.yaml"},
        "Expected all three output hashes",
    )
    for name, digest in output_hashes.items():
        _require(_sha256(output / name) == digest, f"View hash mismatch: {name}")
    payload = _load_payload(output / "train/data.pt")
    _require(
        list(payload["data"].shape) == shape and payload["data"].dtype == torch.float32,
        "Unexpected materialized fields",
    )
    scalars = payload["constant_scalars"]
    _require(
        tuple(scalars.shape) == (2 * count, 1)
        and torch.isfinite(scalars).all().item()
        and torch.allclose(scalars, torch.full_like(scalars, 0.1)),
        "Unexpected materialized amp metadata",
    )
    config = _load_config(output / "resolved_config.yaml")
    _require(
        config.get("ae_only") is True
        and config.get("simulator") == first["simulator"]
        and config.get("dataset")
        == {
            "n_train": 2 * count,
            "n_valid": contract["splits"]["valid"],
            "n_test": contract["splits"]["test"],
            "output_dir": str(output),
        }
        and config.get("view", {}).get("source_frame_indices") == indices
        and config["view"].get("source_saved_times") == times
        and config["view"].get("normalization_path") == str(baseline / "stats.yml"),
        "View configuration does not match approved data selection",
    )
    return manifest


def validate_data_view(output: Path) -> dict[str, Any]:
    """Read-only fit preflight: verify view, provenance and immutable linked splits.

    Hashes protect the materialized tensors and their recorded prior finite checks.
    This avoids repeating the source training scans performed by the builder.
    """
    return _validate_data_view(output, _contract(), KEPT_FRAMES)


def prepare_data_view(baseline: Path, additional: Path, output: Path) -> dict[str, Any]:
    """Create the approved 400-by-160 AE-only view after strict source validation."""
    contract = _contract()
    checked = _validate_sources(baseline, additional, output, contract)
    return _write_view(checked, contract, KEPT_FRAMES)


def main() -> None:
    """Validate or build a new local data view; never launch generation or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--additional", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    operation = validate_sources if args.check_only else prepare_data_view
    record = operation(args.baseline, args.additional, args.output)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
