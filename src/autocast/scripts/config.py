"""Shared config serialization and deserialization for AutoCast scripts."""

import argparse
import logging
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def load_config(args: argparse.Namespace) -> DictConfig:
    """Load and resolve the Hydra configuration based on CLI arguments."""
    config_dir = args.config_dir.resolve()
    overrides = args.overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        hydra_cfg = compose(config_name=args.config_name, overrides=list(overrides))
    return hydra_cfg


def save_resolved_config(
    config: DictConfig, work_dir: Path, filename: str = "resolved_config.yaml"
) -> Path:
    """Save a resolved config YAML file and return the path."""
    resolved_cfg = OmegaConf.to_container(config, resolve=True)
    output_path = work_dir / filename
    with open(output_path, "w") as f:
        yaml.dump(resolved_cfg, f)
    log.info("Wrote resolved config to %s", output_path.resolve())
    return output_path
