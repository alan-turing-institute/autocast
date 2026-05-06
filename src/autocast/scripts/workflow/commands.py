"""Train, eval, train-eval, and time-epochs command implementations."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from autocast.scripts.utils import get_default_config_path, resolve_work_dir
from autocast.scripts.workflow.constants import (
    BENCHMARK_MODULE,
    CACHE_LATENTS_MODULE,
    EVAL_MODULE,
    TRAIN_EVAL_MODULE,
    TRAIN_MODULES,
)
from autocast.scripts.workflow.helpers import format_command, run_module_command
from autocast.scripts.workflow.naming import auto_run_name
from autocast.scripts.workflow.overrides import (
    contains_override,
    hydra_string_list_literal,
)
from autocast.scripts.workflow.slurm import (
    submit_manifest_via_sbatch,
    submit_via_sbatch,
)

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

_RESOLVED_CONFIG_STEMS = (
    "resolved_config",
    "resolved_autoencoder_config",
    "resolved_eval_config",
)


def _hydra_quote_string(value: str) -> str:
    # CLI overrides are passed as argv items (not through a shell), so we only
    # escape embedded double quotes and keep native path separators untouched.
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def build_common_launch_overrides(mode: str, work_dir: Path) -> list[str]:
    """Return Hydra overrides for directory routing in *mode*."""
    if mode == "slurm":
        return [
            "hydra.mode=MULTIRUN",
            "hydra/launcher=slurm",
            f"hydra.sweep.dir={work_dir}",
            "hydra.sweep.subdir=.",
        ]
    return [f"hydra.run.dir={work_dir}"]


def dataset_overrides(dataset: str) -> list[str]:
    """Return Hydra overrides selecting *dataset*."""
    return [
        f"datamodule={dataset}",
    ]


def datasets_root() -> Path:
    """Return the root datasets directory (honouring ``AUTOCAST_DATASETS``)."""
    return Path(os.environ.get("AUTOCAST_DATASETS", Path.cwd() / "datasets"))


def _resolved_config_candidates(base: Path) -> list[Path]:
    return [
        *(base / f"{stem}.yaml" for stem in _RESOLVED_CONFIG_STEMS),
        *(base / "run" / f"{stem}.yaml" for stem in _RESOLVED_CONFIG_STEMS),
    ]


def _flatten_overrides(prefix: str, value: object) -> list[str]:
    """Flatten nested mappings/lists into Hydra dot-path overrides."""
    if isinstance(value, dict):
        overrides: list[str] = []
        for key, nested in value.items():
            if not isinstance(key, str):
                continue
            if key == "defaults":
                continue
            overrides.extend(_flatten_overrides(f"{prefix}.{key}", nested))
        return overrides
    return [f"{prefix}={json.dumps(value)}"]


def _path_exists_in_mapping(mapping: object, path: str) -> bool:
    """Return whether dot-path exists in a nested dict-like mapping."""
    current = mapping
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _struct_safe_overrides(
    overrides: list[str], resolved_cfg: dict[str, object] | None
) -> list[str]:
    """Prefix with '+' when override key is missing under Hydra struct configs."""
    if not isinstance(resolved_cfg, dict):
        return overrides

    adjusted: list[str] = []
    for override in overrides:
        if override.startswith("+") or "=" not in override:
            adjusted.append(override)
            continue

        key, value = override.split("=", 1)
        if _path_exists_in_mapping(resolved_cfg, key):
            adjusted.append(override)
        else:
            adjusted.append(f"+{key}={value}")

    return adjusted


def _resolved_eval_default_overrides() -> list[str]:
    """Return eval.* overrides from the live eval config for stale resolved configs."""
    eval_config_dir = Path(get_default_config_path()) / "eval"
    default_cfg_path = eval_config_dir / "default.yaml"
    epd_cfg_path = eval_config_dir / "encoder_processor_decoder.yaml"
    if not epd_cfg_path.exists():
        return []

    if default_cfg_path.exists():
        cfg = OmegaConf.merge(
            OmegaConf.load(default_cfg_path),
            OmegaConf.load(epd_cfg_path),
        )
    else:
        cfg = OmegaConf.load(epd_cfg_path)

    loaded = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(loaded, dict):
        return []

    return _flatten_overrides("eval", loaded)


def _load_resolved_config_from_workdir(work_dir: str | Path) -> dict | None:
    base = Path(work_dir).expanduser().resolve()
    candidates = _resolved_config_candidates(base)

    for candidate in candidates:
        if not candidate.exists():
            continue
        loaded = OmegaConf.to_container(OmegaConf.load(candidate), resolve=True)
        if isinstance(loaded, dict):
            return loaded

    return None


def infer_hydra_config_from_workdir(work_dir: str | Path) -> tuple[str, str] | None:
    """Infer ``(--config-path, --config-name)`` from a work directory.

    Prefers ``resolved_config.yaml`` and falls back to other resolved config
    variants, including those under ``run/``.
    """
    base = Path(work_dir).expanduser().resolve()
    candidates = _resolved_config_candidates(base)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.parent), candidate.stem

    return None


def _has_cli_flag(overrides: list[str], flag: str) -> bool:
    """Return whether a passthrough CLI flag is already present."""
    return any(item == flag or item.startswith(f"{flag}=") for item in overrides)


def _extract_cli_flag_value(overrides: list[str], flag: str) -> str | None:
    for index, item in enumerate(overrides):
        if item == flag and index + 1 < len(overrides):
            return overrides[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _uses_resolved_config(overrides: list[str]) -> bool:
    config_name = _extract_cli_flag_value(overrides, "--config-name")
    return config_name in _RESOLVED_CONFIG_STEMS


def _with_inferred_resolved_config(
    work_dir: str | Path,
    overrides: list[str],
) -> tuple[list[str], bool]:
    """Add inferred resolved-config flags when no explicit config flags are set."""
    effective_overrides = list(overrides)
    has_config_name = _has_cli_flag(effective_overrides, "--config-name")
    has_config_path = _has_cli_flag(effective_overrides, "--config-path")

    using_resolved_config = _uses_resolved_config(effective_overrides)
    if not (has_config_name or has_config_path):
        inferred_config = infer_hydra_config_from_workdir(work_dir)
        if inferred_config is not None:
            config_path, config_name = inferred_config
            effective_overrides = [
                "--config-name",
                config_name,
                "--config-path",
                config_path,
                *effective_overrides,
            ]
            using_resolved_config = True

    return effective_overrides, using_resolved_config


def _append_inferred_eval_checkpoint(
    work_dir: str | Path,
    overrides: list[str],
) -> list[str]:
    """Append inferred eval.checkpoint override if one is not already provided."""
    effective_overrides = list(overrides)
    if contains_override(effective_overrides, "eval.checkpoint="):
        return effective_overrides

    inferred_eval_checkpoint = infer_eval_checkpoint(work_dir)
    if inferred_eval_checkpoint is None:
        return effective_overrides

    checkpoint_value = _hydra_quote_string(str(inferred_eval_checkpoint))
    effective_overrides.append(f"eval.checkpoint={checkpoint_value}")
    return effective_overrides


def _rewrite_resume_override_for_resolved_config(
    overrides: list[str],
    resolved_cfg: dict[str, object] | None,
) -> list[str]:
    """Adjust resume_from_checkpoint override for resolved-config struct rules."""
    if not isinstance(resolved_cfg, dict):
        return overrides

    cfg_resume = resolved_cfg.get("resume_from_checkpoint")
    if not isinstance(cfg_resume, os.PathLike | str):
        # Key absent in resolved struct: keep +resume_from_checkpoint= so Hydra can add
        return overrides

    resolved_cfg_resume = str(Path(cfg_resume).expanduser().resolve())
    normalized: list[str] = []
    for override in overrides:
        candidate = (
            f"resume_from_checkpoint={override.split('=', 1)[1]}"
            if override.startswith("+resume_from_checkpoint=")
            else override
        )
        if candidate.startswith("resume_from_checkpoint=") and (
            candidate.split("=", 1)[1] == resolved_cfg_resume
        ):
            continue
        normalized.append(candidate)
    return normalized


def _apply_dataset_override_for_resolved_config(
    overrides: list[str],
    dataset: str | None,
) -> list[str]:
    """Map dataset selection to dot-path override for resolved configs."""
    if dataset is None:
        return overrides

    stripped = [
        override for override in overrides if not override.startswith("datamodule=")
    ]
    if not contains_override(stripped, "datamodule.data_path="):
        stripped.append(f"datamodule.data_path={datasets_root() / dataset}")
    return stripped


def _normalize_train_eval_overrides_for_resolved_config(
    work_dir: str | Path,
    overrides: list[str],
    dataset: str | None,
) -> list[str]:
    """Apply resolved-config compatibility fixes for train-eval overrides."""
    resolved_cfg = _load_resolved_config_from_workdir(work_dir)
    normalized = _rewrite_resume_override_for_resolved_config(overrides, resolved_cfg)
    return _apply_dataset_override_for_resolved_config(normalized, dataset)


def infer_dataset_from_workdir(work_dir: str | Path) -> str | None:  # noqa: PLR0911
    """Infer dataset name from a run work directory.

        Reads resolved config YAML if available and infers dataset from:
    - ``datamodule`` when it is a string
        - ``datamodule.data_path`` relative path under ``datasets_root()`` when
            possible (fallback: basename)
    - ``dataset`` top-level key as a fallback
    """
    cfg = _load_resolved_config_from_workdir(work_dir)
    if not isinstance(cfg, dict):
        return None

    datamodule_cfg = cfg.get("datamodule")
    if isinstance(datamodule_cfg, str):
        return datamodule_cfg

    if isinstance(datamodule_cfg, dict):
        dataset_name = datamodule_cfg.get("dataset")
        if isinstance(dataset_name, str) and dataset_name:
            return dataset_name

        data_path = datamodule_cfg.get("data_path")
        if isinstance(data_path, os.PathLike | str):
            data_path_obj = Path(data_path).expanduser()
            datasets_root_obj = datasets_root().expanduser()
            try:
                rel = data_path_obj.resolve().relative_to(datasets_root_obj.resolve())
                return rel.as_posix()
            except (RuntimeError, ValueError):
                return data_path_obj.name

    top_level_dataset = cfg.get("dataset")
    if isinstance(top_level_dataset, str) and top_level_dataset:
        return top_level_dataset

    return None


def infer_resume_checkpoint(kind: str, work_dir: str | Path) -> Path | None:
    """Infer a restart checkpoint path from *work_dir* for a training kind."""
    base = Path(work_dir).expanduser().resolve()

    candidates_by_kind = {
        "ae": ["autoencoder.ckpt", "model.ckpt"],
        "epd": ["encoder_processor_decoder.ckpt", "model.ckpt"],
        "processor": ["processor.ckpt", "model.ckpt"],
    }
    names = candidates_by_kind.get(kind, ["model.ckpt"])

    candidates = [
        *(base / name for name in names),
        *(base / "run" / name for name in names),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def infer_eval_checkpoint(work_dir: str | Path) -> Path | None:
    """Infer an evaluation checkpoint path from *work_dir*."""
    base = Path(work_dir).expanduser().resolve()

    resolved_cfg = _load_resolved_config_from_workdir(base)
    candidate_names: list[str] = []

    if isinstance(resolved_cfg, dict):
        output_cfg = resolved_cfg.get("output")
        if isinstance(output_cfg, dict):
            checkpoint_name = output_cfg.get("checkpoint_name")
            if isinstance(checkpoint_name, str) and checkpoint_name:
                candidate_names.append(checkpoint_name)

        eval_cfg = resolved_cfg.get("eval")
        if isinstance(eval_cfg, dict):
            configured_checkpoint = eval_cfg.get("checkpoint")
            if isinstance(configured_checkpoint, str) and configured_checkpoint:
                candidate_names.append(configured_checkpoint)

    candidate_names.extend(
        ["encoder_processor_decoder.ckpt", "processor.ckpt", "model.ckpt"]
    )

    for name in dict.fromkeys(candidate_names):
        as_path = Path(name).expanduser()
        if as_path.is_absolute() and as_path.exists():
            return as_path.resolve()

        for candidate in (base / as_path, base / "run" / as_path):
            if candidate.exists():
                return candidate.resolve()

    return None


def run_module(
    module: str,
    overrides: list[str],
    dry_run: bool = False,
    mode: str = "local",
    runtime_typechecking: bool = False,
) -> None:
    """Execute *module* locally or via SLURM depending on *mode*."""
    if mode == "slurm":
        submit_via_sbatch(
            module,
            overrides,
            dry_run=dry_run,
            runtime_typechecking=runtime_typechecking,
        )
        return

    cmd = run_module_command(module, overrides)
    if dry_run:
        print(f"DRY-RUN: {format_command(cmd)}")
        return
    subprocess.run(
        cmd,
        check=True,
        env={
            **os.environ,
            "RUNTIME_TYPECHECKING": str(runtime_typechecking).lower(),
        },
    )


def build_effective_eval_overrides(
    train_overrides: list[str], eval_overrides: list[str]
) -> list[str]:
    """Forward model/datamodule overrides from training to eval.

    Training-only prefixes (``trainer.``, ``optimizer.``, etc.) are excluded so
    that evaluation model construction stays aligned with training architecture
    while eval-specific overrides take precedence.
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
    forwarded = [o for o in train_overrides if not o.startswith(train_only_prefixes)]
    return [*forwarded, *eval_overrides]


# ---------------------------------------------------------------------------
# Build override lists
# ---------------------------------------------------------------------------


def build_train_overrides(
    *,
    kind: str,
    mode: str,
    dataset: str | None,
    output_base: str,
    work_dir: str | None,
    resume_from: str | None,
    overrides: list[str],
    run_group: str | None = None,
    run_id: str | None = None,
) -> tuple[Path, str, list[str]]:
    """Resolve workdir/name and build final overrides for a training command."""
    effective_run_id = run_id
    if effective_run_id is None and work_dir is None:
        dataset_for_name = dataset or "default"
        effective_run_id = auto_run_name(
            kind=kind, dataset=dataset_for_name, overrides=overrides
        )

    final_work_dir, resolved_run_id = resolve_work_dir(
        output_base=output_base,
        run_group=run_group,
        run_id=effective_run_id,
        work_dir=work_dir,
        prefix=kind,
    )

    command_overrides = [
        *build_common_launch_overrides(mode=mode, work_dir=final_work_dir),
    ]
    if dataset is not None and not (
        contains_override(overrides, "datamodule=")
        or contains_override(overrides, "datamodule.data_path=")
    ):
        command_overrides.extend(dataset_overrides(dataset=dataset))

    if resume_from is not None and not contains_override(
        overrides, "resume_from_checkpoint="
    ):
        resolved_resume_from = Path(resume_from).expanduser().resolve()
        command_overrides.append(f"+resume_from_checkpoint={resolved_resume_from}")

    if not contains_override(overrides, "logging.wandb.name="):
        command_overrides.append(f"logging.wandb.name={resolved_run_id}")

    command_overrides.extend(overrides)
    return final_work_dir, resolved_run_id, command_overrides


def build_eval_overrides(
    *,
    mode: str,
    dataset: str | None,
    work_dir: str,
    overrides: list[str],
    using_resolved_config: bool = False,
    output_subdir: str = "eval",
) -> tuple[Path, list[str]]:
    """Build evaluation overrides from CLI arguments."""
    base_work_dir = Path(work_dir).expanduser().resolve()
    eval_dir = (base_work_dir / output_subdir).resolve()

    command_overrides = [
        *build_common_launch_overrides(mode=mode, work_dir=eval_dir),
    ]

    # Defaults-group overrides (eval=..., datamodule=...) are only valid when
    # using the default Hydra config which contains a ``defaults`` list. A
    # resolved config already has these sections fully inlined, so we must
    # skip group selectors and only emit dot-path value overrides.
    if not using_resolved_config:
        command_overrides.append("eval=encoder_processor_decoder")
        if dataset is not None:
            command_overrides.extend(dataset_overrides(dataset=dataset))
    else:
        # The resolved config may be stale (saved before current eval defaults
        # were added). Re-inject key EPD eval defaults from the source eval config.
        # These are placed before `overrides` so callers can still override them.
        resolved_cfg = _load_resolved_config_from_workdir(base_work_dir)
        command_overrides.extend(
            _struct_safe_overrides(_resolved_eval_default_overrides(), resolved_cfg)
        )
        if dataset is not None:
            command_overrides.append(
                f"datamodule.data_path={datasets_root() / dataset}"
            )

    command_overrides.extend(overrides)
    return eval_dir, command_overrides


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


def train_command(
    *,
    kind: str,
    mode: str,
    dataset: str | None,
    output_base: str,
    work_dir: str | None,
    resume_from: str | None,
    overrides: list[str],
    run_group: str | None = None,
    run_id: str | None = None,
    runtime_typechecking: bool = False,
    dry_run: bool = False,
) -> tuple[Path, str]:
    """Run a training command."""
    final_work_dir, resolved_run_id, command_overrides = build_train_overrides(
        kind=kind,
        mode=mode,
        dataset=dataset,
        output_base=output_base,
        run_group=run_group,
        run_id=run_id,
        work_dir=work_dir,
        resume_from=resume_from,
        overrides=overrides,
    )

    run_module(
        TRAIN_MODULES[kind],
        command_overrides,
        dry_run=dry_run,
        mode=mode,
        runtime_typechecking=runtime_typechecking,
    )
    return final_work_dir, resolved_run_id


def eval_command(
    *,
    mode: str,
    dataset: str | None,
    work_dir: str,
    overrides: list[str],
    output_subdir: str = "eval",
    runtime_typechecking: bool = False,
    dry_run: bool = False,
) -> None:
    """Run an evaluation command."""
    effective_overrides, using_resolved_config = _with_inferred_resolved_config(
        work_dir, overrides
    )
    effective_overrides = _append_inferred_eval_checkpoint(
        work_dir, effective_overrides
    )
    if not contains_override(effective_overrides, "eval.devices="):
        effective_overrides.append("eval.devices=1")

    _eval_dir, command_overrides = build_eval_overrides(
        mode=mode,
        dataset=dataset,
        work_dir=work_dir,
        overrides=effective_overrides,
        using_resolved_config=using_resolved_config,
        output_subdir=output_subdir,
    )

    run_module(
        EVAL_MODULE,
        command_overrides,
        dry_run=dry_run,
        mode=mode,
        runtime_typechecking=runtime_typechecking,
    )


def benchmark_command(
    *,
    mode: str,
    dataset: str | None,
    work_dir: str,
    overrides: list[str],
    runtime_typechecking: bool = False,
    dry_run: bool = False,
) -> None:
    """Run a benchmark command."""
    effective_overrides, using_resolved_config = _with_inferred_resolved_config(
        work_dir, overrides
    )
    effective_overrides = _append_inferred_eval_checkpoint(
        work_dir, effective_overrides
    )
    if not contains_override(effective_overrides, "eval.devices="):
        effective_overrides.append("eval.devices=1")

    if not contains_override(effective_overrides, "eval.benchmark.enabled="):
        effective_overrides.append("eval.benchmark.enabled=true")

    # Mirror eval_command behaviour: when running from an inferred resolved config,
    # re-inject current eval defaults (including benchmark sections) so older runs
    # created before these defaults existed can still be benchmarked reliably.
    _eval_dir, command_overrides = build_eval_overrides(
        mode=mode,
        dataset=dataset,
        work_dir=work_dir,
        overrides=effective_overrides,
        using_resolved_config=using_resolved_config,
    )

    run_module(
        BENCHMARK_MODULE,
        command_overrides,
        dry_run=dry_run,
        mode=mode,
        runtime_typechecking=runtime_typechecking,
    )


def _read_manifest_lines(manifest: Path) -> list[str]:
    """Return non-blank, non-comment lines from *manifest*."""
    lines: list[str] = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _write_combined_csv(
    work_dirs: list[str],
    manifest: Path,
    *,
    input_name: str,
    output_name: str,
) -> Path | None:
    """Concatenate per-run CSVs into a single combined CSV next to manifest."""
    frames: list[pd.DataFrame] = []
    for wd in work_dirs:
        csv_path = Path(wd) / input_name
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))
        else:
            print(f"WARNING: no {input_name} at {csv_path} — skipping")

    if not frames:
        print(f"WARNING: no {input_name} files found; combined CSV not written.")
        return None

    combined_path = manifest.parent / output_name
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(combined_path, index=False)
    return combined_path


def _write_combined_benchmark_csv(work_dirs: list[str], manifest: Path) -> Path | None:
    """Concatenate per-run ``eval/benchmark_metrics.csv`` files into one CSV."""
    return _write_combined_csv(
        work_dirs,
        manifest,
        input_name="eval/benchmark_metrics.csv",
        output_name=f"{manifest.stem}_combined.csv",
    )


def _parse_benchmark_manifest_line(line: str) -> tuple[str, list[str]]:
    """Parse a manifest line into ``(work_dir, extra_overrides)``.

    Accepts lines with or without the leading ``benchmark`` token, e.g.::

        benchmark --workdir outputs/run_a eval.benchmark_rollout.enabled=true
        --workdir outputs/run_a eval.benchmark_rollout.enabled=true
    """
    # On Windows, POSIX tokenization treats backslashes as escapes and can
    # corrupt paths like C:\Users\... -> C:Users....
    tokens = shlex.split(line, posix=(os.name != "nt"))
    if tokens and tokens[0] == "benchmark":
        tokens = tokens[1:]

    work_dir: str | None = None
    remaining: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "--workdir" and i + 1 < len(tokens):
            work_dir = tokens[i + 1]
            i += 2
        elif tokens[i].startswith("--workdir="):
            work_dir = tokens[i].split("=", 1)[1]
            i += 1
        else:
            remaining.append(tokens[i])
            i += 1

    if work_dir is None:
        msg = f"No --workdir found in manifest line: {line!r}"
        raise ValueError(msg)

    return work_dir, remaining


def benchmark_manifest_command(
    *,
    mode: str,
    manifest: Path,
    overrides: list[str],
    runtime_typechecking: bool = False,
    dry_run: bool = False,
) -> None:
    """Run benchmarks for every entry in *manifest*.

    With ``mode='local'``, runs are executed sequentially in this process and
    a combined ``<manifest_stem>_combined.csv`` is written once all runs
    finish.
    With ``mode='slurm'``, all runs are submitted as a **single** SLURM job
    (sequential on one allocated node) and the same combine step is appended
    to the batch script. Pass ``hydra.launcher.*`` overrides to configure the
    allocation (partition, timeout, GPUs, etc.).
    """
    if not manifest.exists():
        msg = f"Manifest file not found: {manifest}"
        raise FileNotFoundError(msg)

    lines = _read_manifest_lines(manifest)
    if not lines:
        msg = f"No runnable lines found in manifest: {manifest}"
        raise ValueError(msg)

    # Parse upfront so we have work_dirs for both modes.
    parsed = [_parse_benchmark_manifest_line(line) for line in lines]
    work_dirs = [wd for wd, _ in parsed]

    if mode == "slurm":
        submit_manifest_via_sbatch(
            manifest=manifest,
            lines=lines,
            work_dirs=work_dirs,
            overrides=overrides,
            runtime_typechecking=runtime_typechecking,
            dry_run=dry_run,
        )
        return

    # Local: run each line sequentially, then write a combined CSV.
    for work_dir, extra_overrides in parsed:
        dataset = infer_dataset_from_workdir(work_dir)
        benchmark_command(
            mode="local",
            dataset=dataset,
            work_dir=work_dir,
            overrides=[*overrides, *extra_overrides],
            runtime_typechecking=runtime_typechecking,
            dry_run=dry_run,
        )

    if not dry_run:
        combined_path = _write_combined_benchmark_csv(work_dirs, manifest)
        if combined_path is not None:
            print(f"Combined benchmark CSV: {combined_path}")


def train_eval_single_job_command(
    *,
    mode: str,
    dataset: str | None,
    output_base: str,
    work_dir: str | None,
    resume_from: str | None,
    train_overrides: list[str],
    eval_overrides: list[str],
    run_group: str | None = None,
    run_id: str | None = None,
    runtime_typechecking: bool = False,
    dry_run: bool = False,
) -> tuple[Path, str]:
    """Run train→eval in a single Hydra job."""
    final_work_dir, resolved_run_id, command_overrides = build_train_overrides(
        kind="epd",
        mode=mode,
        dataset=dataset,
        output_base=output_base,
        run_group=run_group,
        run_id=run_id,
        work_dir=work_dir,
        resume_from=resume_from,
        overrides=train_overrides,
    )

    # Keep train-eval consistent with eval/benchmark: when a prior workdir is
    # provided and contains a resolved config, reuse it so checkpoint resumes
    # load against the original architecture.
    if work_dir is not None:
        command_overrides, using_resolved_config = _with_inferred_resolved_config(
            work_dir, command_overrides
        )
        if using_resolved_config:
            command_overrides = _normalize_train_eval_overrides_for_resolved_config(
                work_dir=work_dir,
                overrides=command_overrides,
                dataset=dataset,
            )

    if eval_overrides:
        command_overrides.append(
            f"train_eval.eval_overrides={hydra_string_list_literal(eval_overrides)}"
        )

    run_module(
        TRAIN_EVAL_MODULE,
        command_overrides,
        dry_run=dry_run,
        mode=mode,
        runtime_typechecking=runtime_typechecking,
    )
    return final_work_dir, resolved_run_id


def cache_latents_command(
    *,
    mode: str,
    work_dir: str,
    output_dir: str | None,
    overrides: list[str],
    runtime_typechecking: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the cache-latents command to encode data with a trained autoencoder."""
    effective_overrides, _using_resolved_config = _with_inferred_resolved_config(
        work_dir, overrides
    )

    base_work_dir = Path(work_dir).expanduser().resolve()
    cache_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else base_work_dir / "cached"
    )

    command_overrides = [
        *build_common_launch_overrides(mode=mode, work_dir=base_work_dir),
        f"+cache_latents.output_dir={cache_dir}",
        *effective_overrides,
    ]

    run_module(
        CACHE_LATENTS_MODULE,
        command_overrides,
        dry_run=dry_run,
        mode=mode,
        runtime_typechecking=runtime_typechecking,
    )


# ---------------------------------------------------------------------------
# time-epochs
# ---------------------------------------------------------------------------


def _extract_epoch_times_from_checkpoint(ckpt_path: Path) -> list[float] | None:
    """Read per-epoch durations saved by ``TrainingTimerCallback``.

    Loads the Lightning checkpoint at *ckpt_path* and looks for the
    ``epoch_times_s`` list persisted by
    :class:`~autocast.scripts.training.TrainingTimerCallback`.  Returns
    ``None`` when the checkpoint is missing or does not contain timing data.
    """
    if not ckpt_path.exists():
        return None
    try:
        import torch  # noqa: PLC0415 - deferred to avoid import at CLI startup

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        callbacks = ckpt.get("callbacks", {})
        for key, state in callbacks.items():
            if "TrainingTimerCallback" in key and "epoch_times_s" in state:
                times = state["epoch_times_s"]
                if times:
                    return times
    except Exception:
        pass
    return None


def _compute_max_epochs(
    seconds_per_epoch: float,
    budget_hours: float,
    margin: float = 0.02,
) -> dict:
    """Compute ``max_epochs`` that fits within *budget_hours*.

    Subtracts *margin* (fraction) from the budget and rounds down to a
    whole epoch via ``floor()``, guaranteeing that:

    1. All ``max_epochs`` epochs complete before the wall-clock budget.
    2. A cosine half-period schedule (``cosine_epochs = max_epochs``)
       reaches exactly zero and never starts increasing again.
    """
    if seconds_per_epoch <= 0:
        msg = "seconds_per_epoch must be positive"
        raise ValueError(msg)
    if budget_hours <= 0:
        msg = "budget_hours must be positive"
        raise ValueError(msg)
    if not (0.0 <= margin < 1.0):
        msg = "margin must be in [0, 1)"
        raise ValueError(msg)

    budget_seconds = budget_hours * 3600
    usable_seconds = budget_seconds * (1.0 - margin)
    max_epochs = math.floor(usable_seconds / seconds_per_epoch)
    expected_hours = (max_epochs * seconds_per_epoch) / 3600
    headroom_hours = budget_hours - expected_hours
    return {
        "max_epochs": max_epochs,
        "seconds_per_epoch": round(seconds_per_epoch, 2),
        "expected_hours": round(expected_hours, 2),
        "headroom_hours": round(headroom_hours, 2),
        "budget_hours": budget_hours,
        "margin": margin,
    }


def _format_max_time(budget_hours: float) -> str:
    """Format *budget_hours* as a ``DD:HH:MM:SS`` string for Lightning."""
    if budget_hours <= 0:
        msg = "budget_hours must be positive"
        raise ValueError(msg)

    total_seconds = round(budget_hours * 3600)
    days, rem = divmod(total_seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"


def _print_timing_results(
    epoch_times: list[float],
    budget_hours: float,
    margin: float,
) -> dict | None:
    """Compute and print the ``max_epochs`` recommendation from epoch timings."""
    seconds_per_epoch = sum(epoch_times) / len(epoch_times)
    print(
        "\nPer-epoch times (from TrainingTimerCallback): "
        + ", ".join(f"{t:.1f}s" for t in epoch_times)
    )

    try:
        result = _compute_max_epochs(seconds_per_epoch, budget_hours, margin)
        max_time_str = _format_max_time(budget_hours)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        return None

    if result["max_epochs"] < 1:
        print(
            "\nERROR: Computed max_epochs < 1. Increase the budget, reduce the "
            "margin, or re-check the epoch timing estimate."
        )
        return None

    print(f"\n{'=' * 60}")
    print(f"  Seconds/epoch:  {result['seconds_per_epoch']:.1f}s")
    print(f"  Budget:         {budget_hours}h (margin: {margin:.0%})")
    print(f"  max_epochs:     {result['max_epochs']}")
    print(f"  Expected time:  {result['expected_hours']:.1f}h")
    print(f"  Headroom:       {result['headroom_hours']:.1f}h")
    print(f"{'=' * 60}")
    print("\nRecommended overrides:")
    print(
        f"  trainer.max_epochs={result['max_epochs']} "
        f"trainer.max_time={max_time_str} "
        f"optimizer=adamw_half"
    )
    return result


def _validate_time_epochs_args(
    *, num_epochs: int, budget_hours: float, margin: float
) -> None:
    if num_epochs < 1:
        msg = "--num-epochs must be >= 1"
        raise ValueError(msg)
    if budget_hours <= 0:
        msg = "--budget must be > 0"
        raise ValueError(msg)
    if not (0.0 <= margin < 1.0):
        msg = "--margin must be in [0, 1)"
        raise ValueError(msg)


def _run_time_epochs_training(
    *,
    kind: str,
    mode: str,
    dataset: str | None,
    output_base: str,
    overrides: list[str],
    num_epochs: int,
    budget_hours: float,
    margin: float,
    run_group: str | None,
    run_id: str | None,
    work_dir: str | None,
    runtime_typechecking: bool,
    dry_run: bool,
) -> tuple[list[float] | None, bool]:
    """Run timing training job and return (epoch_times, exit_early)."""
    timing_run_id = run_id or "timing"

    use_tempdir = mode == "local" and work_dir is None
    tmpdir_ctx = (
        tempfile.TemporaryDirectory(prefix="autocast_timing_") if use_tempdir else None
    )
    tmpdir = tmpdir_ctx.__enter__() if tmpdir_ctx is not None else None

    try:
        effective_work_dir = tmpdir if use_tempdir else work_dir

        timing_overrides = [
            f"++trainer.max_epochs={num_epochs}",
            "++trainer.max_time=null",
            "logging.wandb.enabled=false",
            "output.skip_test=true",
            "output.save_config=false",
            "output.checkpoint_path=timing.ckpt",
        ]

        final_work_dir, _resolved_run_id, command_overrides = build_train_overrides(
            kind=kind,
            mode=mode,
            dataset=dataset,
            output_base=output_base,
            run_group=run_group,
            run_id=timing_run_id,
            work_dir=str(effective_work_dir)
            if effective_work_dir is not None
            else None,
            resume_from=None,
            overrides=[*timing_overrides, *overrides],
        )

        ckpt_path = final_work_dir / "timing.ckpt"

        if dry_run:
            cmd = run_module_command(TRAIN_MODULES[kind], command_overrides)
            print(f"DRY-RUN: {format_command(cmd)}")
            print(f"\nWould time {num_epochs} epochs, then compute max_epochs")
            print(f"for a {budget_hours}h budget with {margin:.0%} margin.")
            return None, True

        if mode == "slurm":
            run_module(
                TRAIN_MODULES[kind],
                command_overrides,
                dry_run=False,
                mode="slurm",
                runtime_typechecking=runtime_typechecking,
            )
            retrieve_cmd = (
                f"uv run autocast time-epochs "
                f"--from-checkpoint {ckpt_path} "
                f"-b {budget_hours} -m {margin}"
            )
            final_work_dir.mkdir(parents=True, exist_ok=True)
            (final_work_dir / "retrieve.sh").write_text(
                f"#!/usr/bin/env bash\n{retrieve_cmd}\n"
            )
            print(f"\nSLURM job submitted. Workdir: {final_work_dir}")
            print("Once complete, compute results with:")
            print(f"  {retrieve_cmd}")
            print(
                "\nOr collect all timing results at once:\n"
                '  for f in outputs/timing/*/retrieve.sh; do bash "$f"; done'
            )
            return None, True

        print(f"Timing {num_epochs} epoch(s) to estimate per-epoch duration...")
        run_module(
            TRAIN_MODULES[kind],
            command_overrides,
            dry_run=False,
            mode="local",
            runtime_typechecking=runtime_typechecking,
        )
        return _extract_epoch_times_from_checkpoint(ckpt_path), False
    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.__exit__(None, None, None)


def time_epochs_command(
    *,
    kind: str = "epd",
    mode: str,
    dataset: str | None,
    output_base: str,
    overrides: list[str],
    num_epochs: int = 3,
    budget_hours: float = 24.0,
    margin: float = 0.02,
    run_group: str | None = None,
    run_id: str | None = None,
    work_dir: str | None = None,
    from_checkpoint: str | None = None,
    runtime_typechecking: bool = False,
    dry_run: bool = False,
) -> dict | None:
    """Run a short training to time per-epoch duration and recommend ``max_epochs``.

    Executes *num_epochs* epochs of training (ae, epd, or processor) with
    W&B logging and testing disabled, saves a checkpoint so that per-epoch
    wall-clock times can be extracted from ``TrainingTimerCallback``, and
    prints the recommended ``trainer.max_epochs`` for a cosine half-period
    schedule (``optimizer=adamw_half``) that completes within *budget_hours*.

    The calculation is conservative: a *margin* fraction is subtracted
    from the budget **and** the result is rounded down to a whole epoch,
    so the schedule will always reach zero before the wall-clock limit.
    ``trainer.max_time`` is emitted as a hard safety stop equal to the
    full (un-margined) budget.

    With ``--mode slurm`` the timing run is submitted via sbatch and the
    command exits immediately, printing a ``--from-checkpoint`` command to
    retrieve results once the job completes.

    Parameters
    ----------
    kind:
        Training kind: ``"ae"``, ``"epd"``, or ``"processor"``.
    dataset:
        Hydra datamodule group name (e.g. ``"advection_diffusion_multichannel"``).
    output_base:
        Root output directory (forwarded to ``build_train_overrides``).
    overrides:
        Additional Hydra overrides forwarded to the timing run.
    num_epochs:
        How many epochs to run for the timing measurement.
    budget_hours:
        Target wall-clock budget in hours.
    margin:
        Fraction of *budget_hours* held back as safety headroom (default 2 %).
    from_checkpoint:
        Path to an existing checkpoint; skips training and computes the
        recommendation directly.
    """
    try:
        _validate_time_epochs_args(
            num_epochs=num_epochs,
            budget_hours=budget_hours,
            margin=margin,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return None

    if from_checkpoint is not None:
        ckpt = Path(from_checkpoint)
        epoch_times = _extract_epoch_times_from_checkpoint(ckpt)
        if not epoch_times:
            print(
                f"ERROR: Could not extract per-epoch times from {ckpt}. "
                "Check that the checkpoint was produced by a timing run with "
                "TrainingTimerCallback."
            )
            return None
        return _print_timing_results(epoch_times, budget_hours, margin)

    epoch_times, exit_early = _run_time_epochs_training(
        kind=kind,
        mode=mode,
        dataset=dataset,
        output_base=output_base,
        overrides=overrides,
        num_epochs=num_epochs,
        budget_hours=budget_hours,
        margin=margin,
        run_group=run_group,
        run_id=run_id,
        work_dir=work_dir,
        runtime_typechecking=runtime_typechecking,
        dry_run=dry_run,
    )
    if exit_early:
        return None

    if not epoch_times:
        print(
            "\nWARNING: Could not extract per-epoch times from checkpoint. "
            "No timing data available."
        )
        return None

    return _print_timing_results(epoch_times, budget_hours, margin)
