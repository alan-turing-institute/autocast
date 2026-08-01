"""Validate completion and schema of the CNS seed-43 latent cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

EXPECTED_SPLIT_SIZES = {"train": 200, "valid": 20, "test": 20}
EXPECTED_ENCODED_SHAPE = [321, 16, 16, 8]
EXPECTED_GLOBAL_COND_SHAPE = [4]


def _load_metadata(cache_dir: Path) -> dict[str, Any]:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        msg = "metadata.json must contain a mapping"
        raise TypeError(msg)
    return metadata


def _validate_trajectory(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}")
    if set(payload) != {"encoded_fields", "global_cond"}:
        raise ValueError(f"Unexpected keys in {path}: {sorted(payload)}")

    encoded = payload["encoded_fields"]
    global_cond = payload["global_cond"]
    if not isinstance(encoded, torch.Tensor) or not isinstance(
        global_cond, torch.Tensor
    ):
        raise TypeError(f"Expected tensor fields in {path}")
    if list(encoded.shape) != EXPECTED_ENCODED_SHAPE:
        raise ValueError(f"Unexpected encoded shape in {path}: {list(encoded.shape)}")
    if list(global_cond.shape) != EXPECTED_GLOBAL_COND_SHAPE:
        raise ValueError(
            f"Unexpected global conditioning shape in {path}: {list(global_cond.shape)}"
        )
    if not torch.isfinite(encoded).all() or not torch.isfinite(global_cond).all():
        raise ValueError(f"Non-finite values in {path}")


def _validate_split(cache_dir: Path, split: str, expected_size: int) -> None:
    split_dir = cache_dir / split
    files = sorted(split_dir.glob("traj_*.pt"))
    expected_names = [f"traj_{index:06d}.pt" for index in range(expected_size)]
    if [path.name for path in files] != expected_names:
        raise ValueError(
            f"Unexpected trajectory files in {split_dir}: found {len(files)}, "
            f"expected {expected_size}"
        )
    _validate_trajectory(files[0])
    _validate_trajectory(files[-1])


def main() -> None:
    """Validate cache metadata, file counts, and representative tensors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    if not (cache_dir / "autoencoder_config.yaml").is_file():
        raise FileNotFoundError(cache_dir / "autoencoder_config.yaml")

    metadata = _load_metadata(cache_dir)
    if metadata.get("encoder_class") != "DCEncoder":
        msg = "Expected DCEncoder cache metadata"
        raise ValueError(msg)
    if metadata.get("latent_channels") != 8:
        msg = "Expected eight latent channels"
        raise ValueError(msg)

    split_metadata = metadata.get("splits")
    if not isinstance(split_metadata, dict):
        msg = "metadata.splits must contain a mapping"
        raise TypeError(msg)

    for split, expected_size in EXPECTED_SPLIT_SIZES.items():
        details = split_metadata.get(split)
        if not isinstance(details, dict):
            raise TypeError(f"metadata.splits.{split} must contain a mapping")
        if details.get("num_trajectories") != expected_size:
            raise ValueError(f"Unexpected metadata count for {split}")
        sample_shape = details.get("sample_shape")
        if not isinstance(sample_shape, dict):
            raise TypeError(f"metadata.splits.{split}.sample_shape is invalid")
        if sample_shape.get("encoded_fields") != EXPECTED_ENCODED_SHAPE:
            raise ValueError(f"Unexpected metadata encoded shape for {split}")
        if sample_shape.get("global_cond") != EXPECTED_GLOBAL_COND_SHAPE:
            raise ValueError(f"Unexpected metadata global_cond shape for {split}")
        _validate_split(cache_dir, split, expected_size)

    print(f"Validated cached latents: {cache_dir}")
    print("  splits: train=200, valid=20, test=20")
    print("  encoded trajectory shape: (321, 16, 16, 8)")
    print("  global conditioning shape: (4,)")


if __name__ == "__main__":
    main()
