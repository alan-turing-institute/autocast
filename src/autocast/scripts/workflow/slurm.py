"""SLURM sbatch job submission and launcher configuration."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from autocast.scripts.workflow.helpers import (
    ensure_group_writable_parents,
    format_command,
    resolve_umask_from_overrides,
    run_module_command,
    temporary_umask,
)
from autocast.scripts.workflow.naming import sanitize_name_part
from autocast.scripts.workflow.overrides import (
    expand_sweep_overrides,
    extract_override_value,
    normalized_override,
    set_override,
    strip_hydra_sweep_controls,
)

# ---------------------------------------------------------------------------
# Launcher config helpers
# ---------------------------------------------------------------------------


def _parse_override_scalar(value: str) -> int | str:
    stripped = value.strip().strip('"').strip("'")
    return int(stripped) if stripped.isdigit() else stripped


def _nested_set(target: dict, dotted_key: str, value: int | str) -> None:
    parts = dotted_key.split(".")
    current = target
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value


def load_launcher_defaults(launcher_name: str) -> dict:
    """Load launcher YAML defaults by *launcher_name*."""
    candidate_paths = [
        Path(__file__).resolve().parents[2]
        / "configs"
        / "hydra"
        / "launcher"
        / f"{launcher_name}.yaml",
        Path.cwd() / "local_hydra" / "hydra" / "launcher" / f"{launcher_name}.yaml",
    ]
    for path in candidate_paths:
        if path.exists():
            cfg = OmegaConf.load(path)
            loaded = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(loaded, dict):
                loaded.pop("defaults", None)
                return loaded
    if launcher_name != "slurm":
        raise ValueError(f"Unable to resolve hydra launcher preset '{launcher_name}'.")
    return {}


def extract_launcher_overrides(
    overrides: list[str],
) -> tuple[str, dict, list[str]]:
    """Separate ``hydra/launcher`` and ``hydra.launcher.*`` from *overrides*.

    Returns ``(launcher_name, launcher_specific_cfg, remaining_overrides)``.
    """
    launcher_name = "slurm"
    launcher_specific: dict = {}
    remaining: list[str] = []

    for override in overrides:
        norm = normalized_override(override)
        if norm.startswith("hydra/launcher="):
            launcher_name = norm.split("=", 1)[1]
            continue
        if norm.startswith("hydra.launcher."):
            key, raw_value = norm.split("=", 1)
            launcher_key = key.removeprefix("hydra.launcher.")
            _nested_set(
                launcher_specific, launcher_key, _parse_override_scalar(raw_value)
            )
            continue
        remaining.append(override)

    return launcher_name, launcher_specific, remaining


# ---------------------------------------------------------------------------
# sbatch script generation and submission
# ---------------------------------------------------------------------------


def derive_sbatch_job_name(module: str, output_dir: Path, overrides: list[str]) -> str:
    """Build a sanitised SLURM job name from *module* and *overrides*."""
    module_suffix = module.rsplit(".", maxsplit=1)[-1]
    run_name = (
        extract_override_value(overrides, "logging.wandb.name")
        or extract_override_value(overrides, "hydra.job.name")
        or output_dir.name
    )
    raw = f"{module_suffix}_{run_name}"
    sanitized = sanitize_name_part(raw)
    return (sanitized or f"autocast_{module_suffix}")[:120]


def _format_sbatch_time(timeout_min: int | str | None) -> str | None:
    if timeout_min is None:
        return None
    if isinstance(timeout_min, str):
        if timeout_min.isdigit():
            timeout_min = int(timeout_min)
        else:
            return timeout_min
    if timeout_min <= 0:
        return None
    hours, minutes = divmod(timeout_min, 60)
    return f"{hours:02d}:{minutes:02d}:00"


def _submit_one_sbatch_job(
    *,
    module: str,
    output_dir: Path,
    job_overrides: list[str],
    setup_commands: list[str],
    merged_launcher_cfg: dict,
    batch_script_path: Path,
    job_name_suffix: str | None = None,
    umask_value: int,
) -> tuple[str, str]:
    job_name = derive_sbatch_job_name(module, output_dir, job_overrides)
    if job_name_suffix:
        suffix = sanitize_name_part(job_name_suffix)
        if suffix:
            job_name = f"{job_name}_{suffix}"[:120]

    command_text = format_command(run_module_command(module, job_overrides))
    script_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(Path.cwd().resolve()))}",
    ]
    script_lines.extend(str(line) for line in setup_commands)
    script_lines.append(f"exec {command_text}")

    with temporary_umask(umask_value):
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_script_path.parent.mkdir(parents=True, exist_ok=True)
        batch_script_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")

    script_mode = 0o777 & (~umask_value)
    os.chmod(batch_script_path, script_mode)

    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--output={output_dir / 'slurm-%j.out'}",
        f"--error={output_dir / 'slurm-%j.err'}",
    ]

    formatted_time = _format_sbatch_time(merged_launcher_cfg.get("timeout_min"))
    if formatted_time is not None:
        sbatch_cmd.append(f"--time={formatted_time}")

    for cfg_key, sbatch_flag in [
        ("cpus_per_task", "cpus-per-task"),
        ("gpus_per_node", "gpus-per-node"),
        ("tasks_per_node", "ntasks-per-node"),
        ("partition", "partition"),
    ]:
        val = merged_launcher_cfg.get(cfg_key)
        if val is not None:
            sbatch_cmd.append(f"--{sbatch_flag}={val}")

    additional_parameters = merged_launcher_cfg.get("additional_parameters", {})
    if isinstance(additional_parameters, dict):
        for key, value in additional_parameters.items():
            sbatch_cmd.append(f"--{key}={value}")

    sbatch_cmd.append(str(batch_script_path))

    result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
    raw_job_id = result.stdout.strip().splitlines()[-1] if result.stdout else ""
    job_id = raw_job_id.split(";", 1)[0]
    return job_id, job_name


def submit_via_sbatch(
    module: str,
    overrides: list[str],
    dry_run: bool = False,
) -> None:
    """Submit *module* as one or more SLURM jobs via ``sbatch``."""
    if dry_run:
        print(f"DRY-RUN: {format_command(run_module_command(module, overrides))}")
        return

    umask_value = resolve_umask_from_overrides(overrides)

    sweep_dir = extract_override_value(overrides, "hydra.sweep.dir")
    run_dir_override = extract_override_value(overrides, "hydra.run.dir")
    output_dir_raw = sweep_dir or run_dir_override
    output_dir = (
        Path(output_dir_raw).expanduser().resolve()
        if output_dir_raw is not None
        else Path.cwd().resolve()
    )
    with temporary_umask(umask_value):
        output_dir.mkdir(parents=True, exist_ok=True)
    ensure_group_writable_parents(output_dir)

    launcher_name, launcher_override_cfg, module_overrides = extract_launcher_overrides(
        overrides
    )
    launcher_cfg = load_launcher_defaults(launcher_name)
    merged_launcher_cfg = OmegaConf.to_container(
        OmegaConf.merge(launcher_cfg, launcher_override_cfg), resolve=True
    )
    if not isinstance(merged_launcher_cfg, dict):
        merged_launcher_cfg = {}

    module_run_overrides = [
        *(o for o in module_overrides if not o.startswith("hydra.run.dir=")),
        f"hydra.run.dir={output_dir}",
    ]
    module_run_overrides = strip_hydra_sweep_controls(module_run_overrides)

    setup_commands = merged_launcher_cfg.get("setup", [])
    if not isinstance(setup_commands, list):
        setup_commands = []

    expanded_jobs = expand_sweep_overrides(module_run_overrides)

    if len(expanded_jobs) == 1:
        batch_script_path = (output_dir / "submit_job.sh").resolve()
        job_id, job_name = _submit_one_sbatch_job(
            module=module,
            output_dir=output_dir,
            job_overrides=expanded_jobs[0],
            setup_commands=setup_commands,
            merged_launcher_cfg=merged_launcher_cfg,
            batch_script_path=batch_script_path,
            umask_value=umask_value,
        )
        print(f"Submitted SLURM job {job_id} via {batch_script_path}")
        print(f"SLURM job name: {job_name}")
        print(f"SLURM logs: {output_dir / 'slurm-%j.out'}")
        return

    submitted: list[tuple[str, Path, str]] = []
    for index, base_job_overrides in enumerate(expanded_jobs):
        case_dir = (output_dir / f"sweep_{index:04d}").resolve()
        ensure_group_writable_parents(case_dir.parent)
        case_overrides = set_override(
            base_job_overrides, "hydra.run.dir", str(case_dir)
        )

        wandb_name = extract_override_value(case_overrides, "logging.wandb.name")
        if wandb_name:
            case_overrides = set_override(
                case_overrides,
                "logging.wandb.name",
                f"{wandb_name}_s{index:04d}",
            )

        batch_script_path = (
            output_dir / ".sbatch" / f"submit_{index:04d}.sh"
        ).resolve()
        job_id, job_name = _submit_one_sbatch_job(
            module=module,
            output_dir=case_dir,
            job_overrides=case_overrides,
            setup_commands=setup_commands,
            merged_launcher_cfg=merged_launcher_cfg,
            batch_script_path=batch_script_path,
            umask_value=umask_value,
        )
        submitted.append((job_id, batch_script_path, job_name))

    print(f"Submitted {len(submitted)} SLURM jobs for sweep")
    preview = submitted[:5]
    for job_id, script_path, job_name in preview:
        print(f"  - {job_id}: {job_name} via {script_path}")
    if len(submitted) > len(preview):
        print(f"  ... and {len(submitted) - len(preview)} more")
