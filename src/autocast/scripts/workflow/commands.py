"""Train, eval, and train-eval command implementations."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from autocast.scripts.utils import resolve_work_dir
from autocast.scripts.workflow.constants import (
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
from autocast.scripts.workflow.slurm import submit_via_sbatch

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


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


def dataset_overrides(dataset: str, datasets_root: Path) -> list[str]:
    """Return Hydra overrides selecting *dataset*."""
    return [
        f"datamodule={dataset}",
        f"datamodule.data_path={datasets_root / dataset}",
    ]


def datasets_root() -> Path:
    """Return the root datasets directory (honouring ``AUTOCAST_DATASETS``)."""
    return Path(os.environ.get("AUTOCAST_DATASETS", Path.cwd() / "datasets"))


def _load_resolved_config_from_workdir(work_dir: str | Path) -> dict | None:
    base = Path(work_dir).expanduser().resolve()
    candidates = [
        base / "resolved_config.yaml",
        base / "resolved_autoencoder_config.yaml",
        base / "resolved_eval_config.yaml",
        base / "run" / "resolved_config.yaml",
        base / "run" / "resolved_autoencoder_config.yaml",
        base / "run" / "resolved_eval_config.yaml",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        loaded = OmegaConf.to_container(OmegaConf.load(candidate), resolve=True)
        if isinstance(loaded, dict):
            return loaded

    return None


def infer_dataset_from_workdir(work_dir: str | Path) -> str | None:
    """Infer dataset name from a run work directory.

    Reads resolved config YAML if available and infers dataset from:
    - ``datamodule`` when it is a string
    - ``datamodule.data_path`` basename
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
            return Path(data_path).name

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


def resolve_eval_checkpoint(work_dir: Path, checkpoint: str | None) -> Path:
    """Resolve the evaluation checkpoint path.

    Searches for common checkpoint filenames when *checkpoint* is ``None``.
    """
    if checkpoint is not None:
        return Path(checkpoint).expanduser().resolve()
    candidates = [
        work_dir / "encoder_processor_decoder.ckpt",
        work_dir / "run" / "encoder_processor_decoder.ckpt",
        work_dir / "autoencoder.ckpt",
        work_dir / "model.ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_module(
    module: str,
    overrides: list[str],
    dry_run: bool = False,
    mode: str = "local",
) -> None:
    """Execute *module* locally or via SLURM depending on *mode*."""
    if mode == "slurm":
        submit_via_sbatch(module, overrides, dry_run=dry_run)
        return

    cmd = run_module_command(module, overrides)
    if dry_run:
        print(f"DRY-RUN: {format_command(cmd)}")
        return
    subprocess.run(cmd, check=True)


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
    dataset: str,
    output_base: str,
    date_str: str | None,
    run_name: str | None,
    work_dir: str | None,
    wandb_name: str | None,
    resume_from: str | None,
    overrides: list[str],
) -> tuple[Path, str, list[str]]:
    """Resolve workdir/name and build final overrides for a training command."""
    effective_run_name = run_name
    if effective_run_name is None and work_dir is None:
        effective_run_name = auto_run_name(
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
        *build_common_launch_overrides(mode=mode, work_dir=final_work_dir),
        *dataset_overrides(dataset=dataset, datasets_root=datasets_root()),
    ]

    if resume_from is not None and not contains_override(
        overrides, "resume_from_checkpoint="
    ):
        command_overrides.append(f"+resume_from_checkpoint={resume_from}")

    if wandb_name is not None:
        command_overrides.append(f"logging.wandb.name={wandb_name}")
    elif not contains_override(overrides, "logging.wandb.name="):
        command_overrides.append(f"logging.wandb.name={resolved_run_name}")

    command_overrides.extend(overrides)
    return final_work_dir, resolved_run_name, command_overrides


def build_eval_overrides(
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
    """Build evaluation overrides from CLI arguments."""
    base_work_dir = Path(work_dir).expanduser().resolve()
    eval_dir = (base_work_dir / eval_subdir).resolve()

    ckpt = resolve_eval_checkpoint(work_dir=base_work_dir, checkpoint=checkpoint)
    resolved_video_dir = (
        Path(video_dir).expanduser().resolve() if video_dir else (eval_dir / "videos")
    )

    command_overrides = [
        *build_common_launch_overrides(mode=mode, work_dir=eval_dir),
        "eval=encoder_processor_decoder",
        *dataset_overrides(dataset=dataset, datasets_root=datasets_root()),
        f"eval.checkpoint={ckpt}",
        f"eval.batch_indices={batch_indices}",
        f"eval.video_dir={resolved_video_dir}",
        *overrides,
    ]
    return eval_dir, command_overrides


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


def train_command(
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
    """Run a training command."""
    final_work_dir, resolved_run_name, command_overrides = build_train_overrides(
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

    run_module(TRAIN_MODULES[kind], command_overrides, dry_run=dry_run, mode=mode)
    return final_work_dir, resolved_run_name


def eval_command(
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
    """Run an evaluation command."""
    _eval_dir, command_overrides = build_eval_overrides(
        mode=mode,
        dataset=dataset,
        work_dir=work_dir,
        checkpoint=checkpoint,
        eval_subdir=eval_subdir,
        video_dir=video_dir,
        batch_indices=batch_indices,
        overrides=overrides,
    )

    run_module(EVAL_MODULE, command_overrides, dry_run=dry_run, mode=mode)


def train_eval_single_job_command(
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
    """Run train→eval in a single Hydra job."""
    final_work_dir, resolved_run_name, command_overrides = build_train_overrides(
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
            f"train_eval.eval_overrides={hydra_string_list_literal(eval_overrides)}"
        )

    run_module(TRAIN_EVAL_MODULE, command_overrides, dry_run=dry_run, mode=mode)
    return final_work_dir, resolved_run_name
