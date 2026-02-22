"""Auto-naming logic for workflow run directories."""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

from omegaconf import OmegaConf

from autocast.scripts.workflow.constants import DATASET_NAME_TOKENS, NAMING_DEFAULT_KEYS
from autocast.scripts.workflow.overrides import extract_override_value


def sanitize_name_part(value: str) -> str:
    """Sanitize a token to filesystem-friendly characters."""
    stripped = value.strip().strip('"').strip("'")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", stripped)
    return sanitized.strip("-")


def _git_hash() -> str:
    """Return short git hash, or fallback token when unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _short_uuid() -> str:
    return uuid.uuid4().hex[:7]


# ---------------------------------------------------------------------------
# Naming hints from preset YAMLs
# ---------------------------------------------------------------------------


def _naming_hints_from_defaults(defaults: object) -> list[str]:
    if not isinstance(defaults, list):
        return []

    hints: list[str] = []
    for item in defaults:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if not isinstance(value, str):
                continue
            normalized_key = key.removeprefix("override ").lstrip("/")
            if normalized_key in NAMING_DEFAULT_KEYS:
                hints.append(f"{normalized_key}={value}")
    return hints


def _naming_hints_from_model(model_cfg: object) -> list[str]:
    if not isinstance(model_cfg, dict):
        return []

    hints: list[str] = []
    processor_cfg = model_cfg.get("processor")
    if isinstance(processor_cfg, dict):
        target = processor_cfg.get("_target_")
        if isinstance(target, str):
            hints.append(f"model.processor._target_={target}")

    loss_cfg = model_cfg.get("loss_func")
    if isinstance(loss_cfg, dict):
        target = loss_cfg.get("_target_")
        if isinstance(target, str):
            hints.append(f"model.loss_func._target_={target}")

    return hints


def _extract_naming_hints_from_preset(path: Path) -> list[str]:
    if not path.exists():
        return []

    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        return []

    return [
        *_naming_hints_from_defaults(loaded.get("defaults", [])),
        *_naming_hints_from_model(loaded.get("model")),
    ]


def _preset_overrides_for_naming(overrides: list[str]) -> list[str]:
    """Collect naming-relevant hints from ``experiment=`` / ``local_experiment=``."""
    local_experiment = extract_override_value(overrides, "local_experiment")
    experiment = extract_override_value(overrides, "experiment")

    hints: list[str] = []
    if experiment:
        hints.extend(
            _extract_naming_hints_from_preset(
                Path(__file__).resolve().parents[2]
                / "configs"
                / "experiment"
                / f"{experiment}.yaml"
            )
        )
    if local_experiment:
        hints.extend(
            _extract_naming_hints_from_preset(
                Path.cwd()
                / "local_hydra"
                / "local_experiment"
                / f"{local_experiment}.yaml"
            )
        )
    return hints


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dataset_name_token(dataset: str, overrides: list[str]) -> str:
    """Short token for *dataset* used in auto-generated run names."""
    datamodule_cfg = extract_override_value(overrides, "datamodule") or dataset
    return sanitize_name_part(DATASET_NAME_TOKENS.get(datamodule_cfg, datamodule_cfg))


def auto_run_name(kind: str, dataset: str, overrides: list[str]) -> str:
    """Build a legacy-style run name from *kind*, *dataset* and overrides.

    Pattern: ``<prefix>_<dataset>_<model>[_<noise>][_<hidden>]_<git>_<uuid>``
    """
    naming_overrides = [*overrides, *_preset_overrides_for_naming(overrides)]
    dataset_part = dataset_name_token(dataset, naming_overrides)

    if kind == "ae":
        prefix = "ae"
    else:
        loss_target = (
            extract_override_value(naming_overrides, "model.loss_func._target_") or ""
        ).lower()
        processor_ref = (
            extract_override_value(naming_overrides, "processor@model.processor") or ""
        ).lower()
        processor_target = (
            extract_override_value(naming_overrides, "model.processor._target_") or ""
        ).lower()
        processor_text = processor_ref or processor_target

        if "crps" in loss_target:
            prefix = "crps"
        elif "flow_matching" in processor_text or "diffusion" in processor_text:
            prefix = "diff"
        else:
            prefix = "epd"

    model_name = extract_override_value(naming_overrides, "processor@model.processor")
    if model_name is None:
        proc_target = extract_override_value(
            naming_overrides, "model.processor._target_"
        )
        if proc_target:
            model_name = proc_target.split(".")[-2]

    noise_name = extract_override_value(
        naming_overrides, "input_noise_injector@model.input_noise_injector"
    )
    hidden = (
        extract_override_value(naming_overrides, "model.processor.hidden_dim")
        or extract_override_value(naming_overrides, "model.processor.hidden_channels")
        or extract_override_value(
            naming_overrides, "model.processor.backbone.hid_channels"
        )
    )

    parts = [prefix, dataset_part]
    if model_name:
        parts.append(sanitize_name_part(model_name))
    if noise_name:
        parts.append(sanitize_name_part(noise_name))
    if hidden:
        parts.append(sanitize_name_part(str(hidden)))
    parts.append(_git_hash())
    parts.append(_short_uuid())

    return "_".join(part for part in parts if part)
