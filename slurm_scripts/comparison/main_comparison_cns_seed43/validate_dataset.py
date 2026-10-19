"""Validate the seed-43 CNS dataset against the published dataset schema."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import yaml

EXPECTED_SPLIT_SIZES = {"train": 200, "valid": 20, "test": 20}
EXPECTED_KEYS = {"data", "constant_scalars", "constant_fields"}
EXPECTED_TAIL_SHAPE = (321, 64, 64, 3)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}")
    return payload


def _validate_split(root: Path, split: str, size: int) -> dict[str, Any]:
    payload_path = root / split / "data.pt"
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = _load_payload(payload_path)
    if set(payload) != EXPECTED_KEYS:
        raise ValueError(f"Unexpected keys for {split}: {sorted(payload)}")

    data = payload["data"]
    scalars = payload["constant_scalars"]
    if not isinstance(data, torch.Tensor) or not isinstance(scalars, torch.Tensor):
        raise TypeError(f"Tensor fields are invalid for {split}")
    if tuple(data.shape) != (size, *EXPECTED_TAIL_SHAPE):
        raise ValueError(f"Unexpected data shape for {split}: {tuple(data.shape)}")
    if tuple(scalars.shape) != (size, 4):
        raise ValueError(
            f"Unexpected constant_scalars shape for {split}: {tuple(scalars.shape)}"
        )
    if data.dtype != torch.float32 or scalars.dtype != torch.float32:
        raise TypeError(f"Unexpected dtype for {split}: {data.dtype}, {scalars.dtype}")
    if payload["constant_fields"] is not None:
        raise ValueError(f"constant_fields must be None for {split}")
    if not torch.isfinite(scalars).all():
        raise ValueError(f"Non-finite constant scalars in {split}")
    return payload


def _validate_stats(root: Path) -> None:
    with (root / "stats.yml").open(encoding="utf-8") as handle:
        stats = yaml.safe_load(handle)
    if stats.get("core_field_names") != ["smoke", "u", "v"]:
        msg = "Unexpected core_field_names in stats.yml"
        raise ValueError(msg)
    if stats.get("constant_field_names") != []:
        msg = "Unexpected constant_field_names in stats.yml"
        raise ValueError(msg)
    for statistic in ("mean", "std", "mean_delta", "std_delta"):
        values = stats.get("stats", {}).get(statistic, {})
        if set(values) != {"smoke", "u", "v"}:
            raise ValueError(f"Unexpected fields for stats.{statistic}")
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"Non-finite value in stats.{statistic}")


def _validate_config(root: Path, seed: int) -> None:
    with (root / "resolved_config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    simulator = config["simulator"]
    dataset = config["dataset"]
    expected_simulator = {
        "n": 64,
        "L": 32.0,
        "T": 85.869,
        "dt": 0.26,
        "snapshot_dt": 0.261,
        "nu": 0.01,
        "cfl": 0.35,
        "bc_mode": "neumann",
        "buoyancy_mode": "raw",
        "skip_nt": 8,
    }
    for key, expected in expected_simulator.items():
        if simulator.get(key) != expected:
            msg = f"simulator.{key}={simulator.get(key)!r}, expected {expected!r}"
            raise ValueError(msg)
    if config.get("seed") != seed or config.get("overwrite") is not False:
        msg = "Dataset seed/overwrite guard does not match the plan"
        raise ValueError(msg)
    if [dataset.get(f"n_{split}") for split in ("train", "valid", "test")] != [
        200,
        20,
        20,
    ]:
        msg = "Dataset split sizes do not match 200/20/20"
        raise ValueError(msg)


def main() -> None:
    """Validate the new dataset and print its accepted schema."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--published-dataset-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    root = args.dataset_dir.resolve()
    published_root = args.published_dataset_dir.resolve()
    payloads = {
        split: _validate_split(root, split, size)
        for split, size in EXPECTED_SPLIT_SIZES.items()
    }
    _validate_stats(root)
    _validate_config(root, args.seed)

    published_train = _load_payload(published_root / "train" / "data.pt")
    if torch.equal(payloads["train"]["data"][0], published_train["data"][0]):
        msg = "New and published first training trajectories are identical"
        raise ValueError(msg)

    print(f"Validated dataset: {root}")
    for split, payload in payloads.items():
        print(
            f"  {split}: data={tuple(payload['data'].shape)}, "
            f"constant_scalars={tuple(payload['constant_scalars'].shape)}"
        )
    print(f"  seed: {args.seed}")
    print("  first training trajectory differs from the published dataset")


if __name__ == "__main__":
    main()
