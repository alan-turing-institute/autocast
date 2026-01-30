"""Shared configuration utilities for Autocast training scripts.

This module handles:
1. CLI Argument Parsing
2. Hydra Config Loading
3. DataModule Instantiation
4. Dynamic Parameter Resolution (handling 'auto' values)
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from autocast.data.datamodule import SpatioTemporalDataModule

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


def load_config(args: argparse.Namespace) -> DictConfig:
    """Load and resolve the Hydra configuration based on CLI arguments."""
    config_dir = args.config_dir.resolve()
    overrides = args.overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        hydra_cfg = compose(config_name=args.config_name, overrides=list(overrides))
    return hydra_cfg


def build_datamodule(config: DictConfig) -> SpatioTemporalDataModule:
    """Build the DataModule from the Hydra configuration."""
    # Configure datamodule
    dm_cfg = config.get("datamodule")
    if dm_cfg is None:
        msg = "Config must contain 'datamodule' key."
        raise ValueError(msg)
    dm_container = OmegaConf.to_container(dm_cfg, resolve=True)
    if not isinstance(dm_container, dict):
        msg = f"datamodule config must be a mapping, got {type(dm_container).__name__}"
        raise TypeError(msg)

    # Configure simulator if provided
    sim_cfg = config.get("simulator")
    if sim_cfg is not None:
        target = dm_container.get("_target_")
        allow_simulator = True
        if target:
            try:
                allow_simulator = issubclass(
                    get_class(target), SpatioTemporalDataModule
                )
            except Exception:
                allow_simulator = True
        if not allow_simulator:
            msg = (
                "Simulator config provided, but datamodule target is not a "
                "SpatioTemporalDataModule."
            )
            raise ValueError(msg)
        sim_container = OmegaConf.to_container(sim_cfg, resolve=True)
        if not isinstance(sim_container, dict):
            msg = (
                "simulator config must be a mapping containing 'simulator' and "
                "optional 'split' keys"
            )
            raise TypeError(msg)
        simulator_cfg = sim_container.get("simulator", sim_container)
        if not isinstance(simulator_cfg, dict):
            msg = "simulator config missing 'simulator' mapping"
            raise ValueError(msg)
        if dm_container.get("data_path") is not None:
            log.warning(
                "Simulator config provided; ignoring datamodule.data_path and "
                "using generated data instead."
            )
        simulator = instantiate(simulator_cfg)

        # Generate data splits from simulator and assign to datamodule config
        dm_container["data"] = _generate_split(
            simulator, sim_container.get("split", {})
        )
    if "dtype" in dm_container:
        dm_container["dtype"] = _as_dtype(dm_container["dtype"])
    return instantiate(dm_container)


def _as_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    # if isinstance(name, type) and hasattr(name, "dtype"):  # handle actual types
    if isinstance(name, torch.dtype):
        return name
    if not isinstance(name, str):
        return name

    if not hasattr(torch, name):
        raise ValueError(f"Unknown torch dtype '{name}'")
    return getattr(torch, name)


def _generate_split(simulator: Any, split_cfg: dict) -> dict[str, Any]:
    n_train = split_cfg.get("n_train", 0)
    n_valid = split_cfg.get("n_valid", 0)
    n_test = split_cfg.get("n_test", 0)
    log.info(
        "Generating synthetic dataset (train=%s, valid=%s, test=%s)",
        n_train,
        n_valid,
        n_test,
    )
    return {
        "train": simulator.forward_samples_spatiotemporal(n_train),
        "valid": simulator.forward_samples_spatiotemporal(n_valid),
        "test": simulator.forward_samples_spatiotemporal(n_test),
    }


def resolve_auto_params(
    config: DictConfig, input_shape: tuple, output_shape: tuple
) -> DictConfig:
    """Resolve 'auto' values in the configuration using inferred data shapes."""
    training_cfg = config.get("training")
    if training_cfg is None:
        training_cfg = config.get("datamodule")
    if training_cfg is None:
        return config

    if training_cfg.get("n_steps_input") == "auto":
        training_cfg["n_steps_input"] = input_shape[1]
    if training_cfg.get("n_steps_output") == "auto":
        training_cfg["n_steps_output"] = output_shape[1]

    if training_cfg.get("stride") == "auto":
        training_cfg["stride"] = training_cfg.get("n_steps_output", output_shape[1])

    if training_cfg.get("rollout_stride") == "auto":
        training_cfg["rollout_stride"] = training_cfg.get("stride")

    return config
