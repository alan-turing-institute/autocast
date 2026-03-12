"""Argument parser and main entry point for the workflow CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from autocast.scripts.workflow.commands import (
    eval_command,
    infer_dataset_from_workdir,
    infer_resume_checkpoint,
    train_command,
    train_eval_single_job_command,
)
from autocast.scripts.workflow.overrides import extract_override_value

# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every subcommand."""
    parser.add_argument("--mode", choices=["local", "slurm"], default="local")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--config-name",
        help="Hydra top-level config name passthrough.",
    )
    parser.add_argument(
        "--config-path",
        help="Hydra config path passthrough.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. trainer.max_epochs=5",
    )


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by training subcommands (ae, epd, processor, train-eval)."""
    parser.add_argument("--output-base", default="outputs")
    parser.add_argument(
        "--run-group",
        "--run-label",
        dest="run_group",
        help=(
            "Top-level output folder grouping label only (defaults to current date)."
        ),
    )
    parser.add_argument(
        "--run-id",
        "--run-name",
        dest="run_id",
        help=(
            "Run folder identifier; also default W&B name when not "
            "set explicitly via Hydra overrides."
        ),
    )
    parser.add_argument("--workdir")
    parser.add_argument("--resume-from")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the unified workflow CLI."""
    parser = argparse.ArgumentParser(
        prog="autocast",
        description="Unified AutoCast workflow CLI for local and SLURM runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- train subcommands (ae, epd, processor) ----------------------------
    for name in ("ae", "epd", "processor"):
        sub = subparsers.add_parser(name)
        _add_train_args(sub)
        _add_common_args(sub)

    # -- eval --------------------------------------------------------------
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--workdir", required=True)
    _add_common_args(eval_parser)

    # -- train-eval --------------------------------------------------------
    te_parser = subparsers.add_parser("train-eval")
    _add_train_args(te_parser)
    te_parser.add_argument(
        "--eval-overrides",
        nargs="+",
        default=[],
        help=(
            "Hydra overrides for the eval step, e.g. "
            "--eval-overrides eval.batch_indices=[0,1] eval.n_members=10. "
            "Acts as a separator: put train overrides before this flag and "
            "eval overrides after it."
        ),
    )
    _add_common_args(te_parser)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_dataset(
    *,
    work_dir: str | None,
    overrides: list[str],
    dataset: str | None = None,
) -> str | None:
    resolved_dataset = dataset

    if resolved_dataset is None:
        resolved_dataset = extract_override_value(overrides, "datamodule")

    if resolved_dataset is None:
        data_path = extract_override_value(overrides, "datamodule.data_path")
        if data_path:
            resolved_dataset = Path(data_path).name

    if resolved_dataset is None:
        resolved_dataset = extract_override_value(overrides, "dataset")

    if resolved_dataset is None and work_dir is not None:
        resolved_dataset = infer_dataset_from_workdir(work_dir)

    return resolved_dataset


def _resolve_resume_from(
    *,
    kind: str,
    work_dir: str | None,
    resume_from: str | None,
) -> str | None:
    if resume_from is not None or work_dir is None:
        return resume_from

    inferred_resume = infer_resume_checkpoint(kind, work_dir)
    return str(inferred_resume) if inferred_resume is not None else None


def main() -> None:
    """Parse command-line args and execute the selected workflow command."""
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    unknown_flags = [token for token in unknown if token.startswith("-")]
    if unknown_flags:
        parser.error(f"unrecognized arguments: {' '.join(unknown_flags)}")

    # Merge passthrough Hydra globals and positional/unknown overrides.
    combined_overrides = []
    if args.config_name is not None:
        combined_overrides.extend(["--config-name", args.config_name])
    if args.config_path is not None:
        combined_overrides.extend(["--config-path", args.config_path])
    combined_overrides.extend([*args.overrides, *unknown])

    if args.command in {"ae", "epd", "processor"}:
        dataset = _resolve_dataset(
            work_dir=args.workdir,
            overrides=combined_overrides,
        )
        resume_from = _resolve_resume_from(
            kind=args.command,
            work_dir=args.workdir,
            resume_from=args.resume_from,
        )

        train_command(
            kind=args.command,
            mode=args.mode,
            dataset=dataset,
            output_base=args.output_base,
            run_group=args.run_group,
            run_id=args.run_id,
            work_dir=args.workdir,
            resume_from=resume_from,
            overrides=combined_overrides,
            dry_run=args.dry_run,
        )
        return

    if args.command == "eval":
        dataset = _resolve_dataset(
            work_dir=args.workdir,
            overrides=combined_overrides,
        )

        eval_command(
            mode=args.mode,
            dataset=dataset,
            work_dir=args.workdir,
            overrides=combined_overrides,
            dry_run=args.dry_run,
        )
        return

    if args.command == "train-eval":
        dataset = _resolve_dataset(
            work_dir=args.workdir,
            overrides=combined_overrides,
        )
        resume_from = _resolve_resume_from(
            kind="epd",
            work_dir=args.workdir,
            resume_from=args.resume_from,
        )

        train_eval_single_job_command(
            mode=args.mode,
            dataset=dataset,
            output_base=args.output_base,
            run_group=args.run_group,
            run_id=args.run_id,
            work_dir=args.workdir,
            resume_from=resume_from,
            train_overrides=combined_overrides,
            eval_overrides=[*args.eval_overrides],
            dry_run=args.dry_run,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
