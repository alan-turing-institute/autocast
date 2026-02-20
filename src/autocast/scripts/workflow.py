"""Unified workflow CLI for local and SLURM AutoCast runs."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import re
import shlex
import stat
import subprocess
import sys
import uuid
from pathlib import Path

from omegaconf import OmegaConf

from autocast.scripts.utils import resolve_work_dir

TRAIN_MODULES = {
    "ae": "autocast.scripts.train.autoencoder",
    "epd": "autocast.scripts.train.encoder_processor_decoder",
    "processor": "autocast.scripts.train.processor",
}
EVAL_MODULE = "autocast.scripts.eval.encoder_processor_decoder"
TRAIN_EVAL_MODULE = "autocast.scripts.train_eval.encoder_processor_decoder"


def _sanitize_name_part(value: str) -> str:
    """Sanitize a run-name token to filesystem-friendly characters."""
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
    """Return short random suffix used for unique run names."""
    return uuid.uuid4().hex[:7]


def _auto_run_name(kind: str, dataset: str, overrides: list[str]) -> str:
    """Build legacy-style run name without requiring manual --run-name.

    Pattern:
      <prefix>_<dataset>_<model>[_<noise>][_<hidden>]_<git>_<uuid>
    """
    dataset_part = _sanitize_name_part(dataset)

    if kind == "ae":
        prefix = "ae"
    else:
        loss_target = (
            _extract_override_value(overrides, "model.loss_func._target_") or ""
        ).lower()
        processor_ref = (
            _extract_override_value(overrides, "processor@model.processor") or ""
        ).lower()
        processor_target = (
            _extract_override_value(overrides, "model.processor._target_") or ""
        ).lower()
        processor_text = processor_ref or processor_target

        if "crps" in loss_target:
            prefix = "crps"
        elif "flow_matching" in processor_text or "diffusion" in processor_text:
            prefix = "diff"
        else:
            prefix = "epd"

    model_name = _extract_override_value(overrides, "processor@model.processor")
    if model_name is None:
        processor_target = _extract_override_value(
            overrides, "model.processor._target_"
        )
        if processor_target:
            model_name = processor_target.split(".")[-2]

    noise_name = _extract_override_value(
        overrides, "input_noise_injector@model.input_noise_injector"
    )
    hidden = (
        _extract_override_value(overrides, "model.processor.hidden_dim")
        or _extract_override_value(overrides, "model.processor.hidden_channels")
        or _extract_override_value(overrides, "model.processor.backbone.hid_channels")
    )

    parts = [prefix, dataset_part]
    if model_name:
        parts.append(_sanitize_name_part(model_name))
    if noise_name:
        parts.append(_sanitize_name_part(noise_name))
    if hidden:
        parts.append(_sanitize_name_part(str(hidden)))
    parts.append(_git_hash())
    parts.append(_short_uuid())

    return "_".join(part for part in parts if part)


def _build_common_launch_overrides(mode: str, work_dir: Path) -> list[str]:
    if mode == "slurm":
        return [
            "hydra.mode=MULTIRUN",
            "hydra/launcher=slurm",
            f"hydra.sweep.dir={work_dir}",
            "hydra.sweep.subdir=.",
        ]
    return [f"hydra.run.dir={work_dir}"]


def _dataset_overrides(dataset: str, datasets_root: Path) -> list[str]:
    return [
        f"datamodule={dataset}",
        f"datamodule.data_path={datasets_root / dataset}",
    ]


def _datasets_root() -> Path:
    return Path(os.environ.get("AUTOCAST_DATASETS", Path.cwd() / "datasets"))


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _hydra_string_list_literal(values: list[str]) -> str:
    escaped_values = [
        value.replace("\\", "\\\\").replace('"', '\\"') for value in values
    ]
    quoted = [f'"{value}"' for value in escaped_values]
    return f"[{','.join(quoted)}]"


def _load_launcher_defaults(launcher_name: str) -> dict:
    candidate_paths = [
        Path(__file__).resolve().parents[1]
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


def _parse_override_scalar(value: str) -> int | str:
    stripped = value.strip().strip('"').strip("'")
    if stripped.isdigit():
        return int(stripped)
    return stripped


def _nested_set(target: dict, dotted_key: str, value: int | str) -> None:
    parts = dotted_key.split(".")
    current = target
    for part in parts[:-1]:
        next_level = current.get(part)
        if not isinstance(next_level, dict):
            next_level = {}
            current[part] = next_level
        current = next_level
    current[parts[-1]] = value


def _extract_launcher_overrides(overrides: list[str]) -> tuple[str, dict, list[str]]:
    launcher_name = "slurm"
    launcher_specific: dict = {}
    remaining: list[str] = []

    for override in overrides:
        normalized = override[1:] if override.startswith("+") else override
        if normalized.startswith("hydra/launcher="):
            launcher_name = normalized.split("=", 1)[1]
            continue
        if normalized.startswith("hydra.launcher."):
            key, raw_value = normalized.split("=", 1)
            launcher_key = key.removeprefix("hydra.launcher.")
            _nested_set(
                launcher_specific,
                launcher_key,
                _parse_override_scalar(raw_value),
            )
            continue
        remaining.append(override)

    return launcher_name, launcher_specific, remaining


def _derive_sbatch_job_name(module: str, output_dir: Path, overrides: list[str]) -> str:
    module_suffix = module.rsplit(".", maxsplit=1)[-1]
    run_name = (
        _extract_override_value(overrides, "logging.wandb.name")
        or _extract_override_value(overrides, "hydra.job.name")
        or output_dir.name
    )
    raw_name = f"{module_suffix}_{run_name}"
    sanitized = _sanitize_name_part(raw_name)
    if not sanitized:
        sanitized = f"autocast_{module_suffix}"
    return sanitized[:120]


def _resolve_umask_from_overrides(
    overrides: list[str],
    default: str = "0002",
) -> int:
    umask_value = _extract_override_value(overrides, "umask") or default
    normalized = str(umask_value).strip().strip('"').strip("'")
    try:
        return int(normalized, 8)
    except ValueError:
        return int(default, 8)


def _ensure_group_writable_parents(target_dir: Path) -> None:
    """Best-effort ensure g+w on target dir and its parents.

    Walks from the nearest existing ancestor down to target_dir, adding group
    write and setgid bits where permissions allow. Permission errors are ignored
    so this can run safely on shared filesystem roots.
    """
    resolved = target_dir.resolve()
    for directory in [resolved, *resolved.parents]:
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            mode = stat.S_IMODE(directory.stat().st_mode)
            desired_mode = mode | stat.S_IWGRP | stat.S_ISGID
            if desired_mode != mode:
                os.chmod(directory, desired_mode)
        except PermissionError:
            continue


@contextlib.contextmanager
def _temporary_umask(umask_value: int):
    previous = os.umask(umask_value)
    try:
        yield
    finally:
        os.umask(previous)


def _strip_hydra_sweep_controls(overrides: list[str]) -> list[str]:
    filtered: list[str] = []
    for override in overrides:
        normalized = override[1:] if override.startswith("+") else override
        if normalized.startswith("hydra.mode="):
            continue
        if normalized.startswith("hydra.sweep."):
            continue
        filtered.append(override)
    return filtered


def _split_top_level_csv(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_quote: str | None = None
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    for char in value:
        if in_quote is not None:
            current.append(char)
            if char == in_quote:
                in_quote = None
            continue

        if char in {'"', "'"}:
            in_quote = char
            current.append(char)
            continue

        if char == "(":
            paren_depth += 1
            current.append(char)
            continue
        if char == ")":
            paren_depth = max(0, paren_depth - 1)
            current.append(char)
            continue
        if char == "[":
            bracket_depth += 1
            current.append(char)
            continue
        if char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            current.append(char)
            continue
        if char == "{":
            brace_depth += 1
            current.append(char)
            continue
        if char == "}":
            brace_depth = max(0, brace_depth - 1)
            current.append(char)
            continue

        if char == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            token = "".join(current).strip()
            parts.append(token)
            current = []
            continue

        current.append(char)

    token = "".join(current).strip()
    parts.append(token)
    return parts


def _expand_sweep_overrides(overrides: list[str]) -> list[list[str]]:
    choice_groups: list[list[str]] = []

    for override in overrides:
        normalized = override[1:] if override.startswith("+") else override
        if "=" not in normalized:
            choice_groups.append([override])
            continue

        key, value = override.split("=", 1)
        values = _split_top_level_csv(value)
        if len(values) <= 1:
            choice_groups.append([override])
            continue

        choice_groups.append([f"{key}={value_item}" for value_item in values])

    product_size = 1
    for group in choice_groups:
        product_size *= len(group)

    if product_size > 512:
        raise ValueError(
            f"Refusing to submit {product_size} sweep jobs (>512). "
            "Reduce sweep cardinality."
        )

    return [list(combo) for combo in itertools.product(*choice_groups)]


def _set_override(overrides: list[str], key: str, value: str) -> list[str]:
    updated = []
    key_prefix = f"{key}="
    for override in overrides:
        normalized = override[1:] if override.startswith("+") else override
        if normalized.startswith(key_prefix):
            continue
        updated.append(override)
    updated.append(f"{key}={value}")
    return updated


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
    job_name = _derive_sbatch_job_name(module, output_dir, job_overrides)
    if job_name_suffix:
        suffix = _sanitize_name_part(job_name_suffix)
        if suffix:
            job_name = f"{job_name}_{suffix}"[:120]

    command_text = _format_command(_run_module_command(module, job_overrides))
    script_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(Path.cwd().resolve()))}",
    ]
    script_lines.extend(str(line) for line in setup_commands)
    script_lines.append(f"exec {command_text}")
    with _temporary_umask(umask_value):
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

    timeout_min = merged_launcher_cfg.get("timeout_min")
    formatted_time = _format_sbatch_time(timeout_min)
    if formatted_time is not None:
        sbatch_cmd.append(f"--time={formatted_time}")

    cpus_per_task = merged_launcher_cfg.get("cpus_per_task")
    if cpus_per_task is not None:
        sbatch_cmd.append(f"--cpus-per-task={cpus_per_task}")

    gpus_per_node = merged_launcher_cfg.get("gpus_per_node")
    if gpus_per_node is not None:
        sbatch_cmd.append(f"--gpus-per-node={gpus_per_node}")

    tasks_per_node = merged_launcher_cfg.get("tasks_per_node")
    if tasks_per_node is not None:
        sbatch_cmd.append(f"--ntasks-per-node={tasks_per_node}")

    additional_parameters = merged_launcher_cfg.get("additional_parameters", {})
    if isinstance(additional_parameters, dict):
        for key, value in additional_parameters.items():
            sbatch_cmd.append(f"--{key}={value}")

    sbatch_cmd.append(str(batch_script_path))

    result = subprocess.run(
        sbatch_cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    raw_job_id = result.stdout.strip().splitlines()[-1] if result.stdout else ""
    job_id = raw_job_id.split(";", 1)[0]
    return job_id, job_name


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


def _submit_via_sbatch(
    module: str,
    overrides: list[str],
    dry_run: bool = False,
) -> None:
    if dry_run:
        print(f"DRY-RUN: {_format_command(_run_module_command(module, overrides))}")
        return

    umask_value = _resolve_umask_from_overrides(overrides)

    sweep_dir = _extract_override_value(overrides, "hydra.sweep.dir")
    run_dir_override = _extract_override_value(overrides, "hydra.run.dir")
    output_dir_raw = sweep_dir or run_dir_override
    output_dir = (
        Path(output_dir_raw).expanduser().resolve()
        if output_dir_raw is not None
        else Path.cwd().resolve()
    )
    with _temporary_umask(umask_value):
        output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_group_writable_parents(output_dir)

    (
        launcher_name,
        launcher_override_cfg,
        module_overrides,
    ) = _extract_launcher_overrides(overrides)
    launcher_cfg = _load_launcher_defaults(launcher_name)
    merged_launcher_cfg = OmegaConf.to_container(
        OmegaConf.merge(launcher_cfg, launcher_override_cfg),
        resolve=True,
    )
    if not isinstance(merged_launcher_cfg, dict):
        merged_launcher_cfg = {}

    module_run_overrides = [
        *(
            override
            for override in module_overrides
            if not override.startswith("hydra.run.dir=")
        ),
        f"hydra.run.dir={output_dir}",
    ]
    module_run_overrides = _strip_hydra_sweep_controls(module_run_overrides)

    setup_commands = merged_launcher_cfg.get("setup", [])
    if not isinstance(setup_commands, list):
        setup_commands = []

    expanded_jobs = _expand_sweep_overrides(module_run_overrides)

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
        _ensure_group_writable_parents(case_dir.parent)
        case_overrides = _set_override(
            base_job_overrides,
            "hydra.run.dir",
            str(case_dir),
        )

        wandb_name = _extract_override_value(case_overrides, "logging.wandb.name")
        if wandb_name:
            case_overrides = _set_override(
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


def _run_module(
    module: str,
    overrides: list[str],
    dry_run: bool = False,
    mode: str = "local",
) -> None:
    if mode == "slurm":
        _submit_via_sbatch(module, overrides, dry_run=dry_run)
        return

    cmd = [sys.executable, "-m", module, *overrides]
    if dry_run:
        print(f"DRY-RUN: {_format_command(cmd)}")
        return
    subprocess.run(cmd, check=True)


def _run_module_command(module: str, overrides: list[str]) -> list[str]:
    """Build the Python module execution command with overrides."""
    return [sys.executable, "-m", module, *overrides]


def _extract_override_value(overrides: list[str], key: str) -> str | None:
    """Extract latest override value for key from Hydra-style overrides list."""
    for override in reversed(overrides):
        normalized = override[1:] if override.startswith("+") else override
        if normalized.startswith(f"{key}="):
            return normalized.split("=", 1)[1]
    return None


def _contains_override(overrides: list[str], key_prefix: str) -> bool:
    return any(override.startswith(key_prefix) for override in overrides)


def _build_effective_eval_overrides(
    train_overrides: list[str], eval_overrides: list[str]
) -> list[str]:
    """Forward model/datamodule-related train overrides to eval.

    This keeps eval model construction aligned with training architecture while
    allowing explicit eval overrides to take precedence.
    """
    train_only_prefixes = (
        "trainer.",
        "+trainer.",
        "optimizer.",
        "+optimizer.",
        "logging.",
        "+logging.",
        "hydra.",
        "+hydra.",
        "resume_from_checkpoint=",
        "+resume_from_checkpoint=",
        "eval.",
        "+eval.",
    )

    forwarded = [
        override
        for override in train_overrides
        if not override.startswith(train_only_prefixes)
    ]
    return [*forwarded, *eval_overrides]


def _build_train_overrides(
    *,
    kind: str,
    mode: str,
    dataset: str,
    output_base: str,
    date_str: str | None,
    run_name: str | None,
    work_dir: str | None,
    wandb_name: str | None,
    resume_from: str | None,
    overrides: list[str],
) -> tuple[Path, str, list[str]]:
    """Resolve workdir/name and build final overrides for training commands."""
    effective_run_name = run_name
    if effective_run_name is None and work_dir is None:
        effective_run_name = _auto_run_name(
            kind=kind, dataset=dataset, overrides=overrides
        )

    final_work_dir, resolved_run_name = resolve_work_dir(
        output_base=output_base,
        date_str=date_str,
        run_name=effective_run_name,
        work_dir=work_dir,
        prefix=kind,
    )

    command_overrides = [
        *_build_common_launch_overrides(mode=mode, work_dir=final_work_dir),
        *_dataset_overrides(dataset=dataset, datasets_root=_datasets_root()),
    ]

    if resume_from is not None and not _contains_override(
        overrides, "resume_from_checkpoint="
    ):
        command_overrides.append(f"+resume_from_checkpoint={resume_from}")

    if wandb_name is not None:
        command_overrides.append(f"logging.wandb.name={wandb_name}")
    elif not _contains_override(overrides, "logging.wandb.name="):
        command_overrides.append(f"logging.wandb.name={resolved_run_name}")

    command_overrides.extend(overrides)
    return final_work_dir, resolved_run_name, command_overrides


def _build_eval_overrides(
    *,
    mode: str,
    dataset: str,
    work_dir: str,
    checkpoint: str | None,
    eval_subdir: str,
    video_dir: str | None,
    batch_indices: str,
    overrides: list[str],
) -> tuple[Path, list[str]]:
    """Resolve eval workdir/checkpoint and build final eval overrides."""
    base_work_dir = Path(work_dir).expanduser().resolve()
    eval_dir = (base_work_dir / eval_subdir).resolve()

    ckpt = _resolve_eval_checkpoint(work_dir=base_work_dir, checkpoint=checkpoint)
    resolved_video_dir = (
        Path(video_dir).expanduser().resolve() if video_dir else (eval_dir / "videos")
    )

    command_overrides = [
        *_build_common_launch_overrides(mode=mode, work_dir=eval_dir),
        "eval=encoder_processor_decoder",
        *_dataset_overrides(dataset=dataset, datasets_root=_datasets_root()),
        f"eval.checkpoint={ckpt}",
        f"eval.batch_indices={batch_indices}",
        f"eval.video_dir={resolved_video_dir}",
        *overrides,
    ]

    return eval_dir, command_overrides


def _train_command(
    *,
    kind: str,
    mode: str,
    dataset: str,
    output_base: str,
    date_str: str | None,
    run_name: str | None,
    work_dir: str | None,
    wandb_name: str | None,
    resume_from: str | None,
    overrides: list[str],
    dry_run: bool = False,
) -> tuple[Path, str]:
    final_work_dir, resolved_run_name, command_overrides = _build_train_overrides(
        kind=kind,
        mode=mode,
        dataset=dataset,
        output_base=output_base,
        date_str=date_str,
        run_name=run_name,
        work_dir=work_dir,
        wandb_name=wandb_name,
        resume_from=resume_from,
        overrides=overrides,
    )

    _run_module(
        TRAIN_MODULES[kind],
        command_overrides,
        dry_run=dry_run,
        mode=mode,
    )
    return final_work_dir, resolved_run_name


def _resolve_eval_checkpoint(work_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        return Path(checkpoint).expanduser().resolve()
    candidates = [
        work_dir / "encoder_processor_decoder.ckpt",
        work_dir / "run" / "encoder_processor_decoder.ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _eval_command(
    *,
    mode: str,
    dataset: str,
    work_dir: str,
    checkpoint: str | None,
    eval_subdir: str,
    video_dir: str | None,
    batch_indices: str,
    overrides: list[str],
    dry_run: bool = False,
) -> None:
    _eval_dir, command_overrides = _build_eval_overrides(
        mode=mode,
        dataset=dataset,
        work_dir=work_dir,
        checkpoint=checkpoint,
        eval_subdir=eval_subdir,
        video_dir=video_dir,
        batch_indices=batch_indices,
        overrides=overrides,
    )

    _run_module(
        EVAL_MODULE,
        command_overrides,
        dry_run=dry_run,
        mode=mode,
    )


def _train_eval_single_job_command(
    *,
    mode: str,
    dataset: str,
    output_base: str,
    date_str: str | None,
    run_name: str | None,
    work_dir: str | None,
    wandb_name: str | None,
    resume_from: str | None,
    checkpoint: str | None,
    eval_subdir: str,
    video_dir: str | None,
    batch_indices: str,
    train_overrides: list[str],
    eval_overrides: list[str],
    dry_run: bool = False,
) -> tuple[Path, str]:
    final_work_dir, resolved_run_name, command_overrides = _build_train_overrides(
        kind="epd",
        mode=mode,
        dataset=dataset,
        output_base=output_base,
        date_str=date_str,
        run_name=run_name,
        work_dir=work_dir,
        wandb_name=wandb_name,
        resume_from=resume_from,
        overrides=train_overrides,
    )

    command_overrides.append(f"train_eval.eval_subdir={eval_subdir}")
    command_overrides.append(f"train_eval.batch_indices={batch_indices}")
    if checkpoint is not None:
        command_overrides.append(f"train_eval.checkpoint={checkpoint}")
    if video_dir is not None:
        command_overrides.append(f"train_eval.video_dir={video_dir}")
    if eval_overrides:
        command_overrides.append(
            f"train_eval.eval_overrides={_hydra_string_list_literal(eval_overrides)}"
        )

    _run_module(
        TRAIN_EVAL_MODULE,
        command_overrides,
        dry_run=dry_run,
        mode=mode,
    )
    return final_work_dir, resolved_run_name


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the unified workflow CLI."""
    parser = argparse.ArgumentParser(
        prog="autocast",
        description="Unified AutoCast workflow CLI for local and SLURM runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_train_parser(name: str) -> argparse.ArgumentParser:
        train_parser = subparsers.add_parser(name)
        train_parser.add_argument("--dataset", required=True)
        train_parser.add_argument("--mode", choices=["local", "slurm"], default="local")
        train_parser.add_argument("--output-base", default="outputs")
        train_parser.add_argument(
            "--run-label",
            "--date",
            dest="date_str",
            help=(
                "Top-level output folder label (defaults to current date). "
                "--date is kept as a backward-compatible alias."
            ),
        )
        train_parser.add_argument("--run-name")
        train_parser.add_argument("--workdir")
        train_parser.add_argument("--wandb-name")
        train_parser.add_argument("--resume-from")
        train_parser.add_argument("--dry-run", action="store_true")
        train_parser.add_argument(
            "--override",
            action="append",
            default=[],
            help="Additional Hydra override; can be passed multiple times.",
        )
        train_parser.add_argument(
            "overrides",
            nargs="*",
            help=(
                "Additional Hydra overrides passed directly, e.g. trainer.max_epochs=5"
            ),
        )
        return train_parser

    add_train_parser("ae")
    add_train_parser("epd")
    add_train_parser("processor")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--mode", choices=["local", "slurm"], default="local")
    eval_parser.add_argument("--workdir", required=True)
    eval_parser.add_argument("--checkpoint")
    eval_parser.add_argument("--eval-subdir", default="eval")
    eval_parser.add_argument("--video-dir")
    eval_parser.add_argument("--batch-indices", default="[0,1,2,3]")
    eval_parser.add_argument("--dry-run", action="store_true")
    eval_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; can be passed multiple times.",
    )
    eval_parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Additional Hydra overrides passed directly, e.g. eval.batch_indices=[0,1]"
        ),
    )

    train_eval_parser = subparsers.add_parser("train-eval")
    train_eval_parser.add_argument("--dataset", required=True)
    train_eval_parser.add_argument(
        "--mode", choices=["local", "slurm"], default="local"
    )
    train_eval_parser.add_argument("--output-base", default="outputs")
    train_eval_parser.add_argument(
        "--run-label",
        "--date",
        dest="date_str",
        help=(
            "Top-level output folder label (defaults to current date). "
            "--date is kept as a backward-compatible alias."
        ),
    )
    train_eval_parser.add_argument("--run-name")
    train_eval_parser.add_argument("--workdir")
    train_eval_parser.add_argument("--wandb-name")
    train_eval_parser.add_argument("--resume-from")
    train_eval_parser.add_argument("--checkpoint")
    train_eval_parser.add_argument("--eval-subdir", default="eval")
    train_eval_parser.add_argument("--video-dir")
    train_eval_parser.add_argument("--batch-indices", default="[0,1,2,3]")
    train_eval_parser.add_argument("--dry-run", action="store_true")
    train_eval_parser.add_argument(
        "--eval-overrides",
        nargs="+",
        default=[],
        help=(
            "Hydra overrides for the eval step passed in one group, e.g. "
            "--eval-overrides eval.batch_indices=[0,1] +model.n_members=10"
        ),
    )
    train_eval_parser.add_argument(
        "overrides",
        nargs="*",
        help=("Direct Hydra overrides for training, e.g. trainer.max_epochs=1"),
    )

    return parser


def main() -> None:
    """Parse command-line args and execute the selected workflow command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"ae", "epd", "processor"}:
        combined_overrides = [*args.override, *args.overrides]
        _train_command(
            kind=args.command,
            mode=args.mode,
            dataset=args.dataset,
            output_base=args.output_base,
            date_str=args.date_str,
            run_name=args.run_name,
            work_dir=args.workdir,
            wandb_name=args.wandb_name,
            resume_from=args.resume_from,
            overrides=combined_overrides,
            dry_run=args.dry_run,
        )
        return

    if args.command == "eval":
        combined_overrides = [*args.override, *args.overrides]
        _eval_command(
            mode=args.mode,
            dataset=args.dataset,
            work_dir=args.workdir,
            checkpoint=args.checkpoint,
            eval_subdir=args.eval_subdir,
            video_dir=args.video_dir,
            batch_indices=args.batch_indices,
            overrides=combined_overrides,
            dry_run=args.dry_run,
        )
        return

    if args.command == "train-eval":
        train_overrides = [*args.overrides]
        eval_overrides_cli = [*args.eval_overrides]

        _final_work_dir, _run_name = _train_eval_single_job_command(
            mode=args.mode,
            dataset=args.dataset,
            output_base=args.output_base,
            date_str=args.date_str,
            run_name=args.run_name,
            work_dir=args.workdir,
            wandb_name=args.wandb_name,
            resume_from=args.resume_from,
            checkpoint=args.checkpoint,
            eval_subdir=args.eval_subdir,
            video_dir=args.video_dir,
            batch_indices=args.batch_indices,
            train_overrides=train_overrides,
            eval_overrides=eval_overrides_cli,
            dry_run=args.dry_run,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
