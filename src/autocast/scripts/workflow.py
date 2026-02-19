"""Unified workflow CLI for local and SLURM AutoCast runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from autocast.scripts.utils import resolve_work_dir

TRAIN_MODULES = {
    "ae": "autocast.scripts.train.autoencoder",
    "epd": "autocast.scripts.train.encoder_processor_decoder",
    "processor": "autocast.scripts.train.processor",
}
EVAL_MODULE = "autocast.scripts.eval.encoder_processor_decoder"


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


def _run_module(module: str, overrides: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *overrides]
    subprocess.run(cmd, check=True)


def _contains_override(overrides: list[str], key_prefix: str) -> bool:
    return any(override.startswith(key_prefix) for override in overrides)


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
) -> tuple[Path, str]:
    final_work_dir, resolved_run_name = resolve_work_dir(
        output_base=output_base,
        date_str=date_str,
        run_name=run_name,
        work_dir=work_dir,
        prefix=kind,
    )

    datasets_root = Path(os.environ.get("AUTOCAST_DATASETS", Path.cwd() / "datasets"))

    launch = _build_common_launch_overrides(mode=mode, work_dir=final_work_dir)
    command_overrides = [
        *launch,
        *_dataset_overrides(dataset=dataset, datasets_root=datasets_root),
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

    _run_module(TRAIN_MODULES[kind], command_overrides)
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
) -> None:
    base_work_dir = Path(work_dir).expanduser().resolve()
    eval_dir = (base_work_dir / eval_subdir).resolve()
    datasets_root = Path(os.environ.get("AUTOCAST_DATASETS", Path.cwd() / "datasets"))

    ckpt = _resolve_eval_checkpoint(work_dir=base_work_dir, checkpoint=checkpoint)
    resolved_video_dir = (
        Path(video_dir).expanduser().resolve() if video_dir else (eval_dir / "videos")
    )

    launch = _build_common_launch_overrides(mode=mode, work_dir=eval_dir)
    command_overrides = [
        *launch,
        "eval=encoder_processor_decoder",
        *_dataset_overrides(dataset=dataset, datasets_root=datasets_root),
        f"eval.checkpoint={ckpt}",
        f"eval.batch_indices={batch_indices}",
        f"eval.video_dir={resolved_video_dir}",
        *overrides,
    ]

    _run_module(EVAL_MODULE, command_overrides)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the unified workflow CLI."""
    parser = argparse.ArgumentParser(
        prog="autocast_run",
        description="Unified AutoCast workflow CLI for local and SLURM runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_train_parser(name: str) -> argparse.ArgumentParser:
        train_parser = subparsers.add_parser(name)
        train_parser.add_argument("--dataset", required=True)
        train_parser.add_argument("--mode", choices=["local", "slurm"], default="local")
        train_parser.add_argument("--output-base", default="outputs")
        train_parser.add_argument("--date", dest="date_str")
        train_parser.add_argument("--run-name")
        train_parser.add_argument("--workdir")
        train_parser.add_argument("--wandb-name")
        train_parser.add_argument("--resume-from")
        train_parser.add_argument(
            "--override",
            action="append",
            default=[],
            help="Additional Hydra override; can be passed multiple times.",
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
    eval_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; can be passed multiple times.",
    )

    train_eval_parser = subparsers.add_parser("train-eval")
    train_eval_parser.add_argument("--dataset", required=True)
    train_eval_parser.add_argument(
        "--mode", choices=["local", "slurm"], default="local"
    )
    train_eval_parser.add_argument("--output-base", default="outputs")
    train_eval_parser.add_argument("--date", dest="date_str")
    train_eval_parser.add_argument("--run-name")
    train_eval_parser.add_argument("--workdir")
    train_eval_parser.add_argument("--wandb-name")
    train_eval_parser.add_argument("--resume-from")
    train_eval_parser.add_argument("--checkpoint")
    train_eval_parser.add_argument("--eval-subdir", default="eval")
    train_eval_parser.add_argument("--video-dir")
    train_eval_parser.add_argument("--batch-indices", default="[0,1,2,3]")
    train_eval_parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Hydra override for the training step; can be passed multiple times.",
    )
    train_eval_parser.add_argument(
        "--eval-override",
        action="append",
        default=[],
        help="Hydra override for the eval step; can be passed multiple times.",
    )

    return parser


def main() -> None:
    """Parse command-line args and execute the selected workflow command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"ae", "epd", "processor"}:
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
            overrides=args.override,
        )
        return

    if args.command == "eval":
        _eval_command(
            mode=args.mode,
            dataset=args.dataset,
            work_dir=args.workdir,
            checkpoint=args.checkpoint,
            eval_subdir=args.eval_subdir,
            video_dir=args.video_dir,
            batch_indices=args.batch_indices,
            overrides=args.override,
        )
        return

    if args.command == "train-eval":
        final_work_dir, _run_name = _train_command(
            kind="epd",
            mode=args.mode,
            dataset=args.dataset,
            output_base=args.output_base,
            date_str=args.date_str,
            run_name=args.run_name,
            work_dir=args.workdir,
            wandb_name=args.wandb_name,
            resume_from=args.resume_from,
            overrides=args.train_override,
        )
        _eval_command(
            mode=args.mode,
            dataset=args.dataset,
            work_dir=str(final_work_dir),
            checkpoint=args.checkpoint,
            eval_subdir=args.eval_subdir,
            video_dir=args.video_dir,
            batch_indices=args.batch_indices,
            overrides=args.eval_override,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
