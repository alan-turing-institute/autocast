"""Validation helpers for workflow launch configuration."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

from autocast.scripts.workflow.overrides import (
    expand_sweep_overrides,
    extract_override_value,
    normalized_override,
    strip_hydra_sweep_controls,
)

_DISTRIBUTED_DEFAULT_KEYS = {"/distributed", "distributed"}


def _parse_override_scalar(value: str) -> int | str | bool:
    stripped = value.strip().strip('"').strip("'")
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return int(stripped) if stripped.isdigit() else stripped


def _nested_set(target: dict, dotted_key: str, value: int | str | bool) -> None:
    parts = dotted_key.split(".")
    current = target
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value


def _extract_local_experiment_name(overrides: list[str]) -> str | None:
    for override in overrides:
        norm = normalized_override(override)
        if norm.startswith("local_experiment="):
            return norm.split("=", 1)[1]
    return None


def _extract_distributed_preset_name(cfg: dict) -> str | None:
    """Return the distributed preset referenced in a config defaults list, if any."""
    defaults = cfg.get("defaults")
    if not isinstance(defaults, list):
        return None
    for item in defaults:
        if not isinstance(item, dict):
            continue
        for key in _DISTRIBUTED_DEFAULT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _load_distributed_cfg(name: str, *, config_root: Path) -> dict:
    path = config_root / "distributed" / f"{name}.yaml"
    if not path.exists():
        return {}
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return loaded if isinstance(loaded, dict) else {}


def _load_preset_trainer_cfg(overrides: list[str]) -> dict:
    """Load ``trainer`` mapping from selected experiment preset.

    Supports both ``local_experiment=<name>`` and ``experiment=<name>``.
    ``local_experiment`` has precedence if both are provided.
    """
    config_root = Path(__file__).resolve().parents[2] / "configs"

    def _load_trainer_from_file(path: Path) -> dict:
        if not path.exists():
            return {}
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(raw, dict):
            return {}
        distributed = _extract_distributed_preset_name(raw)
        merged = (
            OmegaConf.merge(
                _load_distributed_cfg(distributed, config_root=config_root), raw
            )
            if distributed
            else raw
        )
        cfg = OmegaConf.to_container(merged, resolve=True)
        if not isinstance(cfg, dict):
            return {}
        trainer_cfg = cfg.get("trainer")
        return trainer_cfg if isinstance(trainer_cfg, dict) else {}

    local_experiment = _extract_local_experiment_name(overrides)
    if local_experiment:
        local_cfg_path = (
            Path.cwd() / "local_hydra" / "local_experiment" / f"{local_experiment}.yaml"
        )
        local_trainer_cfg = _load_trainer_from_file(local_cfg_path)
        if local_trainer_cfg:
            return local_trainer_cfg

    experiment_name = extract_override_value(overrides, "experiment")
    if isinstance(experiment_name, str) and experiment_name:
        experiment_cfg_path = config_root / "experiment" / f"{experiment_name}.yaml"
        experiment_trainer_cfg = _load_trainer_from_file(experiment_cfg_path)
        if experiment_trainer_cfg:
            return experiment_trainer_cfg

        external_config_root = os.environ.get("AUTOCAST_CONFIG_PATH")
        if external_config_root:
            external_experiment_cfg_path = (
                Path(external_config_root) / "experiment" / f"{experiment_name}.yaml"
            )
            external_experiment_trainer_cfg = _load_trainer_from_file(
                external_experiment_cfg_path
            )
            if external_experiment_trainer_cfg:
                return external_experiment_trainer_cfg

    return {}


def _extract_trainer_override_cfg(overrides: list[str]) -> dict:
    trainer_specific: dict = {}
    for override in overrides:
        norm = normalized_override(override)
        if not norm.startswith("trainer.") or "=" not in norm:
            continue
        key, raw_value = norm.split("=", 1)
        trainer_key = key.removeprefix("trainer.")
        _nested_set(trainer_specific, trainer_key, _parse_override_scalar(raw_value))
    return trainer_specific


def _as_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _validate_ddp_slurm_alignment(
    *,
    job_overrides: list[str],
    original_overrides: list[str],
    merged_launcher_cfg: dict,
) -> None:
    trainer_cfg = OmegaConf.to_container(
        OmegaConf.merge(
            _load_preset_trainer_cfg(original_overrides),
            _extract_trainer_override_cfg(job_overrides),
        ),
        resolve=True,
    )
    if not isinstance(trainer_cfg, dict):
        return

    devices = _as_positive_int(trainer_cfg.get("devices"))
    if devices is None or devices <= 1:
        return

    tasks_per_node = _as_positive_int(merged_launcher_cfg.get("tasks_per_node"))
    gpus_per_node = _as_positive_int(merged_launcher_cfg.get("gpus_per_node"))

    if tasks_per_node is not None and tasks_per_node != devices:
        msg = (
            "DDP configuration mismatch before submission: "
            "trainer.devices="
            f"{devices} but hydra.launcher.tasks_per_node={tasks_per_node}. "
            "Set tasks_per_node to match devices (or set trainer.devices=1)."
        )
        raise ValueError(msg)

    if gpus_per_node is not None and gpus_per_node < devices:
        msg = (
            "Insufficient GPUs requested for DDP before submission: "
            "trainer.devices="
            f"{devices} but hydra.launcher.gpus_per_node={gpus_per_node}. "
            "Request at least as many GPUs as trainer.devices."
        )
        raise ValueError(msg)


def validate_alignment_for_submission(
    *,
    module_overrides: list[str],
    original_overrides: list[str],
    merged_launcher_cfg: dict,
) -> list[list[str]]:
    """Validate DDP config alignment, returns list of overrides for each submit job."""
    run_dir_override = extract_override_value(original_overrides, "hydra.run.dir")
    module_run_overrides = list(module_overrides)
    if run_dir_override is not None:
        module_run_overrides = [
            *(o for o in module_run_overrides if not o.startswith("hydra.run.dir=")),
            f"hydra.run.dir={run_dir_override}",
        ]
    module_run_overrides = strip_hydra_sweep_controls(module_run_overrides)
    expanded_jobs = expand_sweep_overrides(module_run_overrides)
    for job_overrides in expanded_jobs:
        _validate_ddp_slurm_alignment(
            job_overrides=job_overrides,
            original_overrides=original_overrides,
            merged_launcher_cfg=merged_launcher_cfg,
        )
    return expanded_jobs
