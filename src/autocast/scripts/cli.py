"""Shared CLI argument parsing for AutoCast scripts."""

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse_common_args(description: str, default_config_name: str) -> argparse.Namespace:
    """Parse common CLI arguments for training scripts."""
    parser = argparse.ArgumentParser(description=description)
    repo_root = Path(__file__).resolve().parents[4]

    parser.add_argument(
        "--config-dir",
        "--config-path",
        dest="config_dir",
        type=Path,
        default=repo_root / "configs",
        help="Path to the Hydra config directory.",
    )
    parser.add_argument(
        "--config-name",
        default=default_config_name,
        help=f"Hydra config name (default: '{default_config_name}').",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra config overrides (e.g. trainer.max_epochs=5).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for artifacts and checkpoints (default: CWD).",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint filename override.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip running trainer.test() after training.",
    )

    return parser.parse_args()
