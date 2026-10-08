"""Derive a constant-LR resume checkpoint from a trusted local AE checkpoint.

Weights, optimizer moments/preconditioners and loop counters are preserved.
Never load an untrusted checkpoint: full-state restoration requires pickle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    """Hash the binary input or output for the run's provenance record."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retarget_checkpoint(checkpoint: dict[str, Any], learning_rate: float) -> list[str]:
    """Change only LR metadata and remove source-run diagnostic callback state."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        msg = "The refinement learning rate must be finite and positive"
        raise ValueError(msg)
    optimizers = checkpoint.get("optimizer_states", [])
    schedulers = checkpoint.get("lr_schedulers", [])
    if len(optimizers) != 1 or len(schedulers) != 1 or not optimizers[0]["state"]:
        msg = "A full-state checkpoint with one optimizer and scheduler is required"
        raise ValueError(msg)
    config = checkpoint.get("hyper_parameters", {}).get("optimizer_config")
    if config is None or config.get("scheduler") != "cosine":
        msg = "Expected optimizer_config for the AE's cosine scheduler"
        raise ValueError(msg)
    groups = optimizers[0]["param_groups"]
    scheduler = schedulers[0]
    if len(scheduler.get("base_lrs", [])) != len(groups):
        msg = "Scheduler and optimizer parameter groups do not match"
        raise ValueError(msg)
    for group in groups:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate
    scheduler["base_lrs"] = [learning_rate] * len(groups)
    scheduler["_last_lr"] = [learning_rate] * len(groups)
    config["learning_rate"] = learning_rate
    config["min_lr_ratio"] = 1.0

    removed = []
    for key in list(checkpoint.get("callbacks", {})):
        if any(
            name in key
            for name in (
                "ModelCheckpoint",
                "ValidationMetricPlot",
                "TrainingTimerCallback",
            )
        ):
            removed.append(key)
            del checkpoint["callbacks"][key]
    return removed


def prepare_checkpoint(
    source: Path,
    target: Path,
    *,
    learning_rate: float = 3e-6,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a new checkpoint and hash manifest; refuse existing destinations."""
    source, target = source.resolve(), target.resolve()
    manifest = target.with_suffix(".provenance.json")
    if source == target or target.exists() or manifest.exists():
        msg = "Refusing to overwrite a checkpoint or its provenance manifest"
        raise FileExistsError(msg)
    source_hash = sha256(source)
    if expected_sha256 is not None and source_hash != expected_sha256:
        msg = "Source checkpoint SHA256 does not match the selected input"
        raise ValueError(msg)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    old_lrs = [
        group["lr"]
        for optimizer in checkpoint.get("optimizer_states", [])
        for group in optimizer["param_groups"]
    ]
    removed = retarget_checkpoint(checkpoint, learning_rate)
    with target.open("xb") as stream:
        torch.save(checkpoint, stream)
    if sha256(source) != source_hash:
        msg = "Source checkpoint changed during preparation"
        raise RuntimeError(msg)
    record = {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "derived_checkpoint": str(target),
        "derived_checkpoint_sha256": sha256(target),
        "source_epoch_zero_based": checkpoint["epoch"],
        "source_global_step": checkpoint["global_step"],
        "source_optimizer_lrs": old_lrs,
        "learning_rate": learning_rate,
        "weights_optimizer_state_and_loop_counters_preserved": True,
        "diagnostic_callbacks_reset": removed,
        "scheduler": "restored epoch cosine with min_lr_ratio=1.0 (constant LR)",
        "timer": "launch with reset_resume_time_budget=true",
    }
    with manifest.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def main() -> None:
    """Prepare the selected full-state checkpoint without running training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    record = prepare_checkpoint(
        args.source,
        args.target,
        learning_rate=args.learning_rate,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
