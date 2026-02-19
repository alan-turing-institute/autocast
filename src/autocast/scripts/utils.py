"""Utility functions for AutoCast scripts."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path


def default_run_name(prefix: str = "run") -> str:
    """Generate a short default run name when none is provided."""
    return f"{prefix}_{uuid.uuid4().hex[:7]}"


def resolve_work_dir(
    *,
    output_base: str | Path = "outputs",
    date_str: str | None = None,
    run_name: str | None = None,
    work_dir: str | Path | None = None,
    prefix: str = "run",
) -> tuple[Path, str]:
    """Resolve final work directory and run name.

    Priority:
    1. If ``work_dir`` is provided, use it directly.
    2. Otherwise build ``<output_base>/<date>/<run_name>``.
    3. If ``run_name`` is missing, generate a short default.
    """
    if work_dir is not None:
        resolved = Path(work_dir).expanduser().resolve()
        return resolved, (run_name or resolved.name)

    date_value = date_str or datetime.now().strftime("%Y-%m-%d")
    resolved_name = run_name or default_run_name(prefix=prefix)
    resolved = (Path(output_base) / date_value / resolved_name).expanduser().resolve()
    return resolved, resolved_name


def get_default_config_path() -> str:
    """Find the configs directory by searching upward for project root.

    Searches upward from this file for pyproject.toml (project root marker),
    then returns the path to the configs directory.

    Returns
    -------
    str
        Absolute path to the configs directory.

    Raises
    ------
    FileNotFoundError
        If project root (pyproject.toml) cannot be found.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:  # Stop at filesystem root
        if (current / "pyproject.toml").exists():
            config_dir = current / "configs"
            if not config_dir.exists():
                msg = f"Project root found at {current}, but configs directory missing"
                raise FileNotFoundError(msg)
            return str(config_dir)
        current = current.parent

    msg = "Could not find project root (pyproject.toml)"
    raise FileNotFoundError(msg)
