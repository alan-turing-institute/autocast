"""Argument parser and main entry point for the workflow CLI."""

from __future__ import annotations

import argparse

from autocast.scripts.workflow.commands import (
    eval_command,
    train_command,
    train_eval_single_job_command,
)

# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every subcommand."""
    parser.add_argument("--mode", choices=["local", "slurm"], default="local")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "No-op for SLURM mode (jobs are non-blocking by default). "
            "Reserved for future local background support."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; can be passed multiple times.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. trainer.max_epochs=5",
    )


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by training subcommands (ae, epd, processor, train-eval)."""
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-base", default="outputs")
    parser.add_argument(
        "--run-label",
        "--date",
        dest="date_str",
        help=(
            "Top-level output folder label (defaults to current date). "
            "--date is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--workdir")
    parser.add_argument("--wandb-name")
    parser.add_argument("--resume-from")


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by eval subcommands (eval, train-eval)."""
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-subdir", default="eval")
    parser.add_argument("--video-dir")
    parser.add_argument("--batch-indices", default="[0,1,2,3]")


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
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--workdir", required=True)
    _add_eval_args(eval_parser)
    _add_common_args(eval_parser)

    # -- train-eval --------------------------------------------------------
    te_parser = subparsers.add_parser("train-eval")
    _add_train_args(te_parser)
    _add_eval_args(te_parser)
    te_parser.add_argument(
        "--eval-overrides",
        nargs="+",
        default=[],
        help=(
            "Hydra overrides for the eval step, e.g. "
            "--eval-overrides eval.batch_indices=[0,1] eval.n_members=10"
        ),
    )
    _add_common_args(te_parser)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse command-line args and execute the selected workflow command."""
    parser = build_parser()
    args = parser.parse_args()

    # Merge both override mechanisms consistently across all subcommands.
    combined_overrides = [*args.override, *args.overrides]

    if args.command in {"ae", "epd", "processor"}:
        train_command(
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
        eval_command(
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
        train_eval_single_job_command(
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
            train_overrides=combined_overrides,
            eval_overrides=[*args.eval_overrides],
            dry_run=args.dry_run,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
