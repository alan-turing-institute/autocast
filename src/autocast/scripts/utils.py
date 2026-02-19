"""Utility functions for AutoCast scripts."""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

# Short aliases for dataset names, matching slurm_templates/isambard/dataset_aliases.sh
DATASET_ALIASES: dict[str, str] = {
    "advection_diffusion_multichannel_64_64": "adm64",
    "advection_diffusion_multichannel": "adm32",
}


def determine_epd_run_prefix(cfg: DictConfig) -> str:
    """Determine the run-name prefix for encoder-processor-decoder training.

    Prefix rules (kept intentionally simple):
    - ``crps`` when the configured loss target contains ``crps``.
    - ``diff`` when the processor target contains ``flow_matching`` or ``diffusion``.
    - ``epd`` otherwise.
    """
    model_cfg = cfg.get("model", {})

    loss_target = ""
    loss_cfg = model_cfg.get("loss_func", {})
    if isinstance(loss_cfg, Mapping):
        loss_target = str(loss_cfg.get("_target_", "")).lower()

    processor_target = ""
    processor_cfg = model_cfg.get("processor", {})
    if isinstance(processor_cfg, Mapping):
        processor_target = str(processor_cfg.get("_target_", "")).lower()

    if "crps" in loss_target:
        return "crps"
    if "flow_matching" in processor_target or "diffusion" in processor_target:
        return "diff"
    return "epd"


def generate_run_name(cfg: DictConfig, prefix: str) -> str:
    """Build a descriptive run name from a resolved Hydra config.

    The name follows the pattern used in the SLURM template scripts:
      ``{prefix}_{dataset}_{processor}_{extras}_{git}_{uuid}``

    Parameters
    ----------
    cfg : DictConfig
        The resolved Hydra config.
    prefix : str
        Run type prefix, e.g. ``"ae"``, ``"crps"``, ``"diff"``.
    """
    parts: list[str] = [prefix]

    # ---- dataset info ----
    # see if data_path is in the config and use it to determine dataset name
    datamodule_cfg = cfg.get("datamodule", {})
    data_path = ""
    if isinstance(datamodule_cfg, Mapping):
        data_path = datamodule_cfg.get("data_path", "")
    data_path = str(data_path)
    if data_path:
        # get name of the dataset file from the path
        dataset_key = Path(data_path).name
        # use dataset alias if it exists, otherwise use the full name (may be long)
        if dataset_key in DATASET_ALIASES:
            parts.append(DATASET_ALIASES[dataset_key])
        else:
            parts.append(dataset_key)

    # ---- processor info (not relevant for autoencoders) ----
    if prefix != "ae":
        model_cfg = cfg.get("model", {})
        processor_cfg = model_cfg.get("processor", {})
        processor_target = ""
        if isinstance(processor_cfg, Mapping):
            processor_target = processor_cfg.get("_target_", "")

        # processor short name
        if processor_target:
            # autocast.processors.fno.FNOProcessor → fno
            # autocast.processors.diffusion.DiffusionProcessor → diffusion
            # autocast.processors.flow_matching.FlowMatchingProcessor → flow_matching
            module_path = processor_target.rsplit(".", 1)[0]
            proc_short = module_path.rsplit(".", 1)[-1]
            parts.append(proc_short)

        # hidden dim
        if isinstance(processor_cfg, Mapping):
            # FNOs have hidden_channels, ViT processor has hidden_dim
            hid = processor_cfg.get("hidden_channels") or processor_cfg.get(
                "hidden_dim"
            )
            # diffusion models have hid_channels in the backbone (e.g., ViT) config
            if hid is None:
                backbone = processor_cfg.get("backbone", {})
                if isinstance(backbone, Mapping):
                    hid = backbone.get("hid_channels")
            if hid is not None:
                parts.append(str(hid))

    # ---- git hash ----
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        parts.append(git_hash)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # ---- short uuid ----
    parts.append(uuid.uuid4().hex[:7])

    return "_".join(parts)


def default_run_name(prefix: str = "run") -> str:
    """Generate a short default run name when none is provided."""
    return f"{prefix}_{uuid.uuid4().hex[:7]}"


def resolve_work_dir(
    *,
    output_base: str | Path = "outputs",
    date_str: str | None = None,
    run_name: str | None = None,
    work_dir: str | Path | None = None,
    prefix: str = "run",
) -> tuple[Path, str]:
    """Resolve final work directory and run name.

    Priority:
    1. If ``work_dir`` is provided, use it directly.
    2. Otherwise build ``<output_base>/<date>/<run_name>``.
    3. If ``run_name`` is missing, generate a short default.
    """
    if work_dir is not None:
        resolved = Path(work_dir).expanduser().resolve()
        return resolved, (run_name or resolved.name)

    date_value = date_str or datetime.now().strftime("%Y-%m-%d")
    resolved_name = run_name or default_run_name(prefix=prefix)
    resolved = (Path(output_base) / date_value / resolved_name).expanduser().resolve()
    return resolved, resolved_name


def get_default_config_path() -> str:
    """Find the configs directory by searching upward for project root.

    Searches upward from this file for pyproject.toml (project root marker),
    then returns the path to the configs directory.

    Returns
    -------
    str
        Absolute path to the configs directory.

    Raises
    ------
    FileNotFoundError
        If project root (pyproject.toml) cannot be found.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:  # Stop at filesystem root
        if (current / "pyproject.toml").exists():
            config_dir = current / "configs"
            if not config_dir.exists():
                msg = f"Project root found at {current}, but configs directory missing"
                raise FileNotFoundError(msg)
            return str(config_dir)
        current = current.parent

    msg = "Could not find project root (pyproject.toml)"
    raise FileNotFoundError(msg)
