"""Unified workflow CLI for local and SLURM AutoCast runs."""

from autocast.scripts.workflow.cli import build_parser, main

__all__ = ["build_parser", "main"]
