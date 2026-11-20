"""SLURM sbatch job submission and launcher configuration."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

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


def _parse_override_scalar(value: str) -> int | str | bool:
    stripped = value.strip().strip('"').strip("'")
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return int(stripped) if stripped.isdigit() else stripped


def _as_positive_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _launcher_positive_int(launcher_cfg: dict, key: str) -> int:
    value = launcher_cfg.get(key)
    if value is None:
        additional_parameters = launcher_cfg.get("additional_parameters", {})
        if isinstance(additional_parameters, dict):
            value = additional_parameters.get(key)
    return _as_positive_int(value)


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
        if not path.exists():
            continue
        cfg = OmegaConf.load(path)
        loaded = OmegaConf.to_container(cfg, resolve=False)
        if isinstance(loaded, dict):
            loaded.pop("defaults", None)
            return loaded
    if launcher_name != "slurm":
        raise ValueError(f"Unable to resolve hydra launcher preset '{launcher_name}'.")
    return {}


def _extract_local_experiment_name(overrides: list[str]) -> str | None:
    """Return selected local_experiment name from CLI overrides, if any."""
    for override in overrides:
        norm = normalized_override(override)
        if norm.startswith("local_experiment="):
            return norm.split("=", 1)[1]
    return None


def _resolve_local_experiment_parent(item: object) -> Path | None:
    # Map a defaults-list entry to its YAML file so we can follow the
    # inheritance chain when looking up ``/distributed``.
    # Handles both absolute (``/local_experiment/foo``) and relative
    # (``foo/bar``) references within the local_experiment config group.
    if not isinstance(item, str):
        return None
    if item.startswith("/local_experiment/"):
        rel = item.removeprefix("/local_experiment/")
        return Path.cwd() / "local_hydra" / "local_experiment" / f"{rel}.yaml"
    if item.startswith(("_", "/")):
        return None
    return Path.cwd() / "local_hydra" / "local_experiment" / f"{item}.yaml"


def _extract_distributed_preset_name(
    cfg: dict, seen: set[Path] | None = None
) -> str | None:
    if seen is None:
        seen = set()
    defaults = cfg.get("defaults")
    if not isinstance(defaults, list):
        return None
    for item in defaults:
        if not isinstance(item, dict):
            continue
        val = item.get("/distributed") or item.get("override /distributed")
        if isinstance(val, str) and val:
            return val
    for item in defaults:
        parent_path = _resolve_local_experiment_parent(item)
        if parent_path is None or parent_path in seen or not parent_path.exists():
            continue
        seen.add(parent_path)
        parent_raw = OmegaConf.to_container(OmegaConf.load(parent_path), resolve=False)
        if not isinstance(parent_raw, dict):
            continue
        result = _extract_distributed_preset_name(parent_raw, seen)
        if result:
            return result
    return None


def _extract_distributed_override_name(overrides: list[str]) -> str | None:
    for override in reversed(overrides):
        norm = normalized_override(override)
        if norm.startswith(("distributed=", "/distributed=")):
            value = norm.split("=", 1)[1].strip()
            return value or None
    return None


def _load_distributed_cfg(name: str) -> dict:
    cfg_root = Path(__file__).resolve().parents[2] / "configs"
    candidate_paths = [
        cfg_root / "distributed" / f"{name}.yaml",
        Path.cwd() / "local_hydra" / "distributed" / f"{name}.yaml",
    ]
    external_config_root = os.environ.get("AUTOCAST_CONFIG_PATH")
    if external_config_root:
        candidate_paths.append(
            Path(external_config_root) / "distributed" / f"{name}.yaml"
        )
    for path in candidate_paths:
        if not path.exists():
            continue
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _extract_launcher_cfg(cfg: dict) -> dict:
    hydra_cfg = cfg.get("hydra")
    if not isinstance(hydra_cfg, dict):
        return {}
    launcher_cfg = hydra_cfg.get("launcher")
    return launcher_cfg if isinstance(launcher_cfg, dict) else {}


def _load_launcher_from_file(path: Path) -> dict:
    if not path.exists():
        return {}
    # Do not resolve the whole experiment config here; it may contain
    # interpolations that are valid only during full Hydra composition
    # (e.g. model.processor.n_steps_input -> datamodule.n_steps_input).
    # We only need the hydra.launcher subtree for submission defaults.
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(raw, dict):
        return {}
    distributed = _extract_distributed_preset_name(raw)
    merged = (
        OmegaConf.merge(_load_distributed_cfg(distributed), OmegaConf.create(raw))
        if distributed
        else OmegaConf.create(raw)
    )
    cfg = OmegaConf.to_container(merged, resolve=False)
    if not isinstance(cfg, dict):
        return {}
    return _extract_launcher_cfg(cfg)


def _load_direct_distributed_launcher_cfg(overrides: list[str]) -> dict:
    distributed = _extract_distributed_override_name(overrides)
    if distributed is None:
        return {}
    return _extract_launcher_cfg(_load_distributed_cfg(distributed))


def _load_preset_launcher_cfg(overrides: list[str]) -> dict:
    """Load ``hydra.launcher`` mapping from selected experiment preset.

    Supports both ``local_experiment=<name>`` and ``experiment=<name>``.
    ``local_experiment`` has precedence if both are provided.
    """
    local_experiment = _extract_local_experiment_name(overrides)
    if local_experiment:
        local_cfg_path = (
            Path.cwd() / "local_hydra" / "local_experiment" / f"{local_experiment}.yaml"
        )
        local_launcher_cfg = _load_launcher_from_file(local_cfg_path)
        if local_launcher_cfg:
            return local_launcher_cfg

    experiment_name = extract_override_value(overrides, "experiment")
    if isinstance(experiment_name, str) and experiment_name:
        experiment_cfg_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "experiment"
            / f"{experiment_name}.yaml"
        )
        experiment_launcher_cfg = _load_launcher_from_file(experiment_cfg_path)
        if experiment_launcher_cfg:
            return experiment_launcher_cfg

        external_config_root = os.environ.get("AUTOCAST_CONFIG_PATH")
        if external_config_root:
            external_experiment_cfg_path = (
                Path(external_config_root) / "experiment" / f"{experiment_name}.yaml"
            )
            external_experiment_launcher_cfg = _load_launcher_from_file(
                external_experiment_cfg_path
            )
            if external_experiment_launcher_cfg:
                return external_experiment_launcher_cfg

    return {}


def _resolve_launcher_submission_context(
    overrides: list[str],
) -> tuple[dict, list[str]]:
    launcher_name, launcher_override_cfg, module_overrides = extract_launcher_overrides(
        overrides
    )
    preset_launcher_cfg = _load_preset_launcher_cfg(overrides)
    direct_distributed_launcher_cfg = _load_direct_distributed_launcher_cfg(overrides)
    launcher_cfg = load_launcher_defaults(launcher_name)
    merged_launcher_cfg = OmegaConf.to_container(
        OmegaConf.merge(
            launcher_cfg,
            preset_launcher_cfg,
            direct_distributed_launcher_cfg,
            launcher_override_cfg,
        ),
        resolve=True,
    )
    if not isinstance(merged_launcher_cfg, dict):
        merged_launcher_cfg = {}
    return merged_launcher_cfg, module_overrides


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
        norm = override.lstrip("+")
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


def _submission_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _should_use_srun(launcher_cfg: dict) -> bool:
    explicit = launcher_cfg.get("use_srun")
    if isinstance(explicit, bool):
        return explicit
    if isinstance(explicit, str):
        lowered = explicit.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False

    nodes = _launcher_positive_int(launcher_cfg, "nodes")
    tasks_per_node = _launcher_positive_int(launcher_cfg, "tasks_per_node")
    gpus_per_node = _launcher_positive_int(launcher_cfg, "gpus_per_node")
    ntasks = _launcher_positive_int(launcher_cfg, "ntasks")
    return nodes > 1 or tasks_per_node > 1 or gpus_per_node > 1 or ntasks > 1


def _build_sbatch_command(
    *,
    job_name: str,
    log_dir: Path,
    launcher_cfg: dict,
    batch_script_path: Path,
) -> list[str]:
    """Build the sbatch command for a generated batch script."""
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--output={log_dir / 'slurm-%j.out'}",
        f"--error={log_dir / 'slurm-%j.err'}",
    ]

    formatted_time = _format_sbatch_time(launcher_cfg.get("timeout_min"))
    if formatted_time is not None:
        sbatch_cmd.append(f"--time={formatted_time}")

    mapped_sbatch_flags = [
        ("nodes", "nodes"),
        ("cpus_per_task", "cpus-per-task"),
        ("gpus_per_node", "gpus-per-node"),
        ("tasks_per_node", "ntasks-per-node"),
        ("partition", "partition"),
    ]
    handled_additional_keys: set[str] = set()
    for cfg_key, sbatch_flag in mapped_sbatch_flags:
        val = launcher_cfg.get(cfg_key)
        if val is not None:
            sbatch_cmd.append(f"--{sbatch_flag}={val}")
            handled_additional_keys.add(cfg_key)

    additional_parameters = launcher_cfg.get("additional_parameters", {})
    if isinstance(additional_parameters, dict):
        for key, value in additional_parameters.items():
            if key in handled_additional_keys:
                continue
            sbatch_cmd.append(f"--{key}={value}")

    sbatch_cmd.append(str(batch_script_path))
    return sbatch_cmd


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
    runtime_typechecking: bool,
    chain_options: tuple[str, ...] = (),
    dry_run: bool = False,
) -> tuple[str, str]:
    job_name = derive_sbatch_job_name(module, output_dir, job_overrides)
    if job_name_suffix:
        suffix = sanitize_name_part(job_name_suffix)
        if suffix:
            job_name = f"{job_name[: 119 - len(suffix)]}_{suffix}"

    command = run_module_command(module, job_overrides)
    if _should_use_srun(merged_launcher_cfg):
        srun_args = ["--kill-on-bad-exit=1"] if chain_options else []
        command = ["srun", *srun_args, *command]
    script_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(Path.cwd().resolve()))}",
        f"export RUNTIME_TYPECHECKING={str(runtime_typechecking).lower()}",
        *(str(line) for line in setup_commands),
        f"exec {format_command(command)}",
    ]
    script = "\n".join(script_lines) + "\n"
    sbatch_cmd = _build_sbatch_command(
        job_name=job_name,
        log_dir=output_dir,
        launcher_cfg=merged_launcher_cfg,
        batch_script_path=batch_script_path,
    )
    sbatch_cmd[-1:-1] = chain_options
    if dry_run:
        print(f"DRY-RUN: {format_command(sbatch_cmd)}\n{script}")
        return f"<{batch_script_path.stem}-job-id>", job_name

    with temporary_umask(umask_value):
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_script_path.parent.mkdir(parents=True, exist_ok=True)
        batch_script_path.write_text(script, encoding="utf-8")
    os.chmod(batch_script_path, 0o777 & (~umask_value))
    result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
    raw_job_id = result.stdout.strip().splitlines()[-1] if result.stdout else ""
    job_id = raw_job_id.split(";", 1)[0]
    return job_id, job_name


def _submit_training_chain(
    module: str,
    output_dir: Path,
    overrides: list[str],
    launcher: dict,
    *,
    segments: int,
    umask: int,
    runtime_typechecking: bool,
    dry_run: bool,
) -> None:
    if segments < 2 or len(expand_sweep_overrides(overrides)) != 1:
        msg = "--chain requires at least two allocations and one configuration."
        raise ValueError(msg)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"A training chain requires a new run directory: {output_dir}")
    minutes = _as_positive_int(launcher.get("timeout_min"))
    extra = launcher.get("additional_parameters")
    extra = extra if isinstance(extra, dict) else {}
    if minutes < 2 or "time" in extra:
        msg = "Set hydra.launcher.timeout_min to an integer of at least two minutes."
        raise ValueError(msg)
    launcher.setdefault("nodes", extra.get("nodes", 1))
    budget = minutes - min(30, max(1, minutes // 10))
    run_id = uuid4().hex[:12]
    chain_dir = output_dir / ".chain"
    finished = shlex.quote(str(chain_dir / "finished"))
    setup = launcher.get("setup", [])
    if not dry_run:
        with temporary_umask(umask):
            output_dir.mkdir(parents=True, exist_ok=False)

    previous = None
    for index in range(1, segments + 1):
        segment = chain_dir / f"segment-{index:04d}"
        prologue = [
            ': "${SLURM_JOB_ID:?Run inside a Slurm allocation}"',
            "allocation_start=$(date +%s)",
        ]
        job_overrides = overrides.copy()
        options = ("--no-requeue",)
        if previous is not None:
            options += (f"--dependency=afterok:{previous}", "--kill-on-invalid-dep=yes")
            prologue += [
                f"finished={finished}",
                'if [[ -e "$finished" || -L "$finished" ]]; then',
                '  [[ -L "$finished" && -f "$finished" ]] || '
                "{ echo 'Chain completion checkpoint is missing' >&2; exit 1; }",
                "  exit 0",
                "fi",
            ]
            checkpoint = chain_dir / f"segment-{index - 1:04d}.ckpt"
            resume = json.dumps(str(checkpoint), ensure_ascii=False)
            job_overrides.append(f"++resume_from_checkpoint={resume}")
        prologue += [
            f"mkdir {shlex.quote(str(segment.with_suffix('.started')))}",
            f'export AUTOCAST_SLURM_CHAIN="{index}:'
            f'$((allocation_start + {budget * 60})):{run_id}"',
            f"export AUTOCAST_SLURM_CHAIN_FINAL={int(index == segments)}",
        ]
        previous, _ = _submit_one_sbatch_job(
            module=module,
            output_dir=output_dir,
            job_overrides=job_overrides,
            setup_commands=[*prologue, *(setup if isinstance(setup, list) else [])],
            merged_launcher_cfg=launcher,
            batch_script_path=segment.with_suffix(".sh"),
            job_name_suffix=segment.name,
            umask_value=umask,
            runtime_typechecking=runtime_typechecking,
            chain_options=options,
            dry_run=dry_run,
        )
        if not dry_run:
            if not previous.isdecimal():
                raise ValueError(f"Unexpected sbatch job ID: {previous!r}")
            segment.with_suffix(".job-id").write_text(previous + "\n", encoding="utf-8")
            print(f"Submitted Slurm job {previous} for {segment.name}", flush=True)


def submit_via_sbatch(
    module: str,
    overrides: list[str],
    runtime_typechecking: bool = False,
    dry_run: bool = False,
    *,
    chain: int | None = None,
) -> None:
    """Submit *module* as one or more SLURM jobs via ``sbatch``."""
    merged_launcher_cfg, module_overrides = _resolve_launcher_submission_context(
        overrides
    )

    output_dir_raw = extract_override_value(
        overrides, "hydra.sweep.dir"
    ) or extract_override_value(overrides, "hydra.run.dir")
    output_dir = Path(output_dir_raw or ".").expanduser().resolve()
    module_run_overrides = set_override(
        strip_hydra_sweep_controls(module_overrides),
        "hydra.run.dir",
        json.dumps(str(output_dir), ensure_ascii=False) if chain else str(output_dir),
    )

    if chain is not None:
        _submit_training_chain(
            module,
            output_dir,
            module_run_overrides,
            merged_launcher_cfg,
            segments=chain,
            umask=resolve_umask_from_overrides(overrides),
            runtime_typechecking=runtime_typechecking,
            dry_run=dry_run,
        )
        return

    expanded_jobs = expand_sweep_overrides(module_run_overrides)
    if dry_run:
        if len(expanded_jobs) == 1:
            print(
                "DRY-RUN: "
                f"{format_command(run_module_command(module, expanded_jobs[0]))}"
            )
            return

        print(f"DRY-RUN: would submit {len(expanded_jobs)} SLURM jobs for sweep")
        for index, base_job_overrides in enumerate(expanded_jobs):
            case_dir = (output_dir / f"sweep_{index:04d}").resolve()
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

            print(
                f"  - [{index:04d}] "
                f"{format_command(run_module_command(module, case_overrides))}"
            )
        return

    umask_value = resolve_umask_from_overrides(overrides)

    with temporary_umask(umask_value):
        output_dir.mkdir(parents=True, exist_ok=True)
    ensure_group_writable_parents(output_dir)

    setup_commands = merged_launcher_cfg.get("setup", [])
    if not isinstance(setup_commands, list):
        setup_commands = []

    submission_ts = _submission_timestamp()

    if len(expanded_jobs) == 1:
        batch_script_path = (output_dir / f"submit_job_{submission_ts}.sh").resolve()
        job_id, job_name = _submit_one_sbatch_job(
            module=module,
            output_dir=output_dir,
            job_overrides=expanded_jobs[0],
            setup_commands=setup_commands,
            merged_launcher_cfg=merged_launcher_cfg,
            batch_script_path=batch_script_path,
            umask_value=umask_value,
            runtime_typechecking=runtime_typechecking,
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
            output_dir / ".sbatch" / f"submit_{submission_ts}_{index:04d}.sh"
        ).resolve()
        job_id, job_name = _submit_one_sbatch_job(
            module=module,
            output_dir=case_dir,
            job_overrides=case_overrides,
            setup_commands=setup_commands,
            merged_launcher_cfg=merged_launcher_cfg,
            batch_script_path=batch_script_path,
            umask_value=umask_value,
            runtime_typechecking=runtime_typechecking,
        )
        submitted.append((job_id, batch_script_path, job_name))

    print(f"Submitted {len(submitted)} SLURM jobs for sweep")
    preview = submitted[:5]
    for job_id, script_path, job_name in preview:
        print(f"  - {job_id}: {job_name} via {script_path}")
    if len(submitted) > len(preview):
        print(f"  ... and {len(submitted) - len(preview)} more")


def submit_manifest_via_sbatch(
    manifest: Path,
    lines: list[str],
    work_dirs: list[str],
    overrides: list[str],
    runtime_typechecking: bool = False,
    dry_run: bool = False,
) -> None:
    r"""Submit all *lines* from *manifest* as a **single** SLURM job.

    Runs are executed sequentially inside one SLURM allocation.  A combine
    step is appended that concatenates per-run ``eval/benchmark_metrics.csv``
    files into ``<manifest_stem>_combined.csv`` next to the manifest.

    SLURM allocation is configured via ``hydra.launcher.*`` overrides, e.g.::

        autocast benchmark --manifest benchmarks.txt \\
            hydra.launcher.partition=gpu \\
            hydra.launcher.timeout_min=60 \\
            hydra.launcher.gpus_per_node=1
    """
    umask_value = resolve_umask_from_overrides(overrides)

    launcher_name, launcher_override_cfg, _ = extract_launcher_overrides(overrides)
    preset_launcher_cfg = _load_preset_launcher_cfg(overrides)
    direct_distributed_launcher_cfg = _load_direct_distributed_launcher_cfg(overrides)
    launcher_cfg = load_launcher_defaults(launcher_name)
    merged_launcher_cfg = OmegaConf.to_container(
        OmegaConf.merge(
            launcher_cfg,
            preset_launcher_cfg,
            direct_distributed_launcher_cfg,
            launcher_override_cfg,
        ),
        resolve=True,
    )
    if not isinstance(merged_launcher_cfg, dict):
        merged_launcher_cfg = {}

    setup_commands: list[str] = merged_launcher_cfg.get("setup", [])  # type: ignore[assignment]
    if not isinstance(setup_commands, list):
        setup_commands = []

    cwd = Path.cwd().resolve()
    submission_ts = _submission_timestamp()
    log_dir = cwd / "slurm_logs" / f"benchmark_manifest_{submission_ts}"
    job_name = f"autocast_benchmark_manifest_{manifest.stem}"[:120]

    # Build batch script body
    script_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(cwd))}",
        f"export RUNTIME_TYPECHECKING={str(runtime_typechecking).lower()}",
        f"echo '=== autocast benchmark manifest: {manifest.name} ==='",
        f"echo 'Runs     : {len(lines)}'",
        "echo 'Node     : '$(hostname)",
        "echo 'Started  : '$(date)",
        "echo '=============================='",
    ]
    for cmd in setup_commands:
        script_lines.append(str(cmd))
    for line in lines:
        # Each manifest line is a full `benchmark --workdir X ...` invocation.
        # Run it locally (no --mode slurm) inside the allocated node.
        script_lines.append(f"uv run autocast {line}")
    # Combine step: concatenate per-run eval/benchmark_metrics.csv files.
    csv_paths_repr = repr(
        [str(Path(wd) / "eval" / "benchmark_metrics.csv") for wd in work_dirs]
    )
    combined_path_repr = repr(
        str(manifest.parent.resolve() / f"{manifest.stem}_combined.csv")
    )
    combine_snippet = (
        "import pandas as pd, pathlib; "
        f"_ps=[p for p in {csv_paths_repr} if pathlib.Path(p).exists()]; "
        f"pd.concat([pd.read_csv(p) for p in _ps], ignore_index=True)"
        f".to_csv({combined_path_repr}, index=False) if _ps else None; "
        f"print('Combined benchmark CSV:', {combined_path_repr})"
    )
    script_lines.append(f"uv run python -c {shlex.quote(combine_snippet)}")
    script_lines.append("echo '=== Completed '$(date) '==='")

    script_text = "\n".join(script_lines) + "\n"

    if dry_run:
        print(f"DRY-RUN: single SLURM job for manifest {manifest} ({len(lines)} runs)")
        print("--- sbatch script ---")
        print(script_text)
        return

    with temporary_umask(umask_value):
        log_dir.mkdir(parents=True, exist_ok=True)

    batch_script_path = log_dir / f"submit_{submission_ts}.sh"
    script_mode = 0o777 & (~umask_value)
    batch_script_path.write_text(script_text, encoding="utf-8")
    os.chmod(batch_script_path, script_mode)

    sbatch_cmd = _build_sbatch_command(
        job_name=job_name,
        log_dir=log_dir,
        launcher_cfg=merged_launcher_cfg,
        batch_script_path=batch_script_path,
    )

    result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
    raw_job_id = result.stdout.strip().splitlines()[-1] if result.stdout else ""
    job_id = raw_job_id.split(";", 1)[0]

    print(f"Submitted SLURM job {job_id} ({len(lines)} benchmarks, sequential)")
    print(f"Manifest : {manifest}")
    print(f"Script   : {batch_script_path}")
    print(f"Logs     : {log_dir / 'slurm-%j.out'}")
