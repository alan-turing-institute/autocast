"""Utility functions for AutoCast scripts."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from importlib import resources
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
    """Resolve default Hydra config path.

    Resolution order:
    1. ``AUTOCAST_CONFIG_PATH`` environment variable (if set and exists).
    2. Repository ``src/autocast/configs/`` directory (detected via ``pyproject.toml``).
    3. Packaged ``autocast/configs`` resources (for wheel/sdist installs).

    This allows local development with repository configs while supporting
    installed-package layouts.

    Returns
    -------
    str
        Absolute path to the configs directory.

    Raises
    ------
    FileNotFoundError
        If no valid config directory can be resolved.
    """
    env_path = os.environ.get("AUTOCAST_CONFIG_PATH")
    if env_path:
        config_dir = Path(env_path).expanduser().resolve()
        if config_dir.exists():
            return str(config_dir)
        msg = f"AUTOCAST_CONFIG_PATH was set but does not exist: {config_dir}"
        raise FileNotFoundError(msg)

    current = Path(__file__).resolve().parent
    while current != current.parent:  # Stop at filesystem root
        if (current / "pyproject.toml").exists():
            config_dir = current / "src" / "autocast" / "configs"
            if not config_dir.exists():
                msg = f"Project root found at {current}, but configs directory missing"
                raise FileNotFoundError(msg)
            return str(config_dir)
        current = current.parent

    try:
        packaged_configs = resources.files("autocast") / "configs"
        if packaged_configs.is_dir():
            return str(Path(str(packaged_configs)).resolve())
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    msg = (
        "Could not resolve configs directory. Set AUTOCAST_CONFIG_PATH or install "
        "package data including autocast/configs."
    )
    raise FileNotFoundError(msg)
