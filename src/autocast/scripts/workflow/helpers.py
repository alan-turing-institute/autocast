"""Filesystem, umask, and small shared utilities."""

from __future__ import annotations

import contextlib
import os
import shlex
import stat
import sys
from pathlib import Path

from autocast.scripts.workflow.overrides import extract_override_value


def format_command(command: list[str]) -> str:
    """Format a command list as a shell-safe string."""
    return " ".join(shlex.quote(part) for part in command)


def run_module_command(module: str, overrides: list[str]) -> list[str]:
    """Build the ``python -m`` command list for *module* with *overrides*."""
    return [sys.executable, "-m", module, *overrides]


def resolve_umask_from_overrides(
    overrides: list[str],
    default: str = "0002",
) -> int:
    """Parse the ``umask`` override (octal string) into an int."""
    umask_value = extract_override_value(overrides, "umask") or default
    normalized = str(umask_value).strip().strip('"').strip("'")
    try:
        return int(normalized, 8)
    except ValueError:
        return int(default, 8)


def ensure_group_writable_parents(target_dir: Path) -> None:
    """Best-effort ``g+w`` and setgid on *target_dir* and its ancestors.

    Walks from the nearest existing ancestor down to *target_dir*, adding
    group-write and setgid bits where permissions allow.  Permission errors
    are silently ignored so this runs safely on shared filesystem roots.
    """
    resolved = target_dir.resolve()
    for directory in [resolved, *resolved.parents]:
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            mode = stat.S_IMODE(directory.stat().st_mode)
            desired = mode | stat.S_IWGRP | stat.S_ISGID
            if desired != mode:
                os.chmod(directory, desired)
        except PermissionError:
            continue


@contextlib.contextmanager
def temporary_umask(umask_value: int):
    """Context manager that sets the process umask and restores it on exit."""
    previous = os.umask(umask_value)
    try:
        yield
    finally:
        os.umask(previous)
