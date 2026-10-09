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
from typing import Any, cast

import torch
from omegaconf import OmegaConf


def sha256(path: Path) -> str:
    """Hash the binary input or output for the run's provenance record."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retarget_checkpoint(
    checkpoint: dict[str, Any],
    learning_rate: float,
    *,
    source_optimizer_config: dict[str, Any] | None = None,
) -> list[str]:
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
    # AE does not save this hyperparameter. Use only an explicitly supplied
    # original training config, not the refinement preset or guessed defaults.
    source_config = (
        source_optimizer_config if source_optimizer_config is not None else config
    )
    if source_config is None:
        msg = "Missing optimizer metadata; supply the original --source-config"
        raise ValueError(msg)
    if (
        source_config.get("scheduler") != "cosine"
        or source_config.get("scheduler_interval", "epoch") != "epoch"
    ):
        msg = "Expected optimizer_config for the AE's cosine scheduler"
        raise ValueError(msg)
    groups = optimizers[0]["param_groups"]
    scheduler = schedulers[0]
    if len(scheduler.get("base_lrs", [])) != len(groups):
        msg = "Scheduler and optimizer parameter groups do not match"
        raise ValueError(msg)
    base_lr = source_config.get("learning_rate")
    if (
        not isinstance(base_lr, (int, float))
        or not math.isfinite(base_lr)
        or base_lr <= 0
    ):
        msg = "Source optimizer config requires a finite positive base learning rate"
        raise ValueError(msg)
    if any(
        not math.isclose(lr, base_lr, rel_tol=1e-9) for lr in scheduler["base_lrs"]
    ) or any(
        not math.isclose(group.get("initial_lr", base_lr), base_lr, rel_tol=1e-9)
        for group in groups
    ):
        msg = "Source config base learning rate does not match the checkpoint"
        raise ValueError(msg)
    for group in groups:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate
    scheduler["base_lrs"] = [learning_rate] * len(groups)
    scheduler["_last_lr"] = [learning_rate] * len(groups)
    if config is not None:
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
    source_config: Path | None = None,
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
    config_hash = None
    optimizer_config: dict[str, Any] | None = None
    if source_config is not None:
        source_config = source_config.expanduser().resolve()
        config_hash = sha256(source_config)
        resolved_optimizer = OmegaConf.to_container(
            OmegaConf.load(source_config).optimizer, resolve=True
        )
        if not isinstance(resolved_optimizer, dict):
            msg = "Source config must contain an optimizer mapping"
            raise ValueError(msg)
        optimizer_config = cast(dict[str, Any], resolved_optimizer)
    old_lrs = [
        group["lr"]
        for optimizer in checkpoint.get("optimizer_states", [])
        for group in optimizer["param_groups"]
    ]
    removed = retarget_checkpoint(
        checkpoint, learning_rate, source_optimizer_config=optimizer_config
    )
    with target.open("xb") as stream:
        torch.save(checkpoint, stream)
    if sha256(source) != source_hash:
        msg = "Source checkpoint changed during preparation"
        raise RuntimeError(msg)
    record = {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "source_config": str(source_config) if source_config is not None else None,
        "source_config_sha256": config_hash,
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
    parser.add_argument(
        "--source-config",
        type=Path,
        help="Original resolved AE config, required if optimizer metadata is absent.",
    )
    args = parser.parse_args()
    record = prepare_checkpoint(
        args.source,
        args.target,
        learning_rate=args.learning_rate,
        expected_sha256=args.expected_sha256,
        source_config=args.source_config,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
