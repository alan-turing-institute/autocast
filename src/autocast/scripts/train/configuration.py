"""Shared configuration utilities for Autocast training scripts.

This module handles:
1. CLI Argument Parsing
2. Hydra Config Loading & Conversion to Pydantic
3. DataModule Instantiation
4. Dynamic Parameter Resolution (handling 'auto' values)
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autocast.config.base import Config
from autocast.data.datamodule import SpatioTemporalDataModule
from autocast.data.dataset import SpatioTemporalDataset

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


def load_config(args: argparse.Namespace) -> Config:
    """Load Hydra config, apply overrides, and convert to Pydantic Config."""
    config_dir = args.config_dir.resolve()
    overrides: Sequence[str] = args.overrides or []

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        hydra_cfg = compose(config_name=args.config_name, overrides=list(overrides))

    # Resolve strings like ${oc.env:DATA_PATH}
    resolved_container = OmegaConf.to_container(hydra_cfg, resolve=True)

    # Validation occurs here
    config = Config(**resolved_container)  # type: ignore TODO: confirm handling

    # Apply CLI args that override config values
    if hasattr(args, "n_steps_input") and args.n_steps_input is not None:
        config.training.n_steps_input = args.n_steps_input
    if hasattr(args, "n_steps_output") and args.n_steps_output is not None:
        config.training.n_steps_output = args.n_steps_output
    if hasattr(args, "stride") and args.stride is not None:
        config.training.stride = args.stride
    if (
        hasattr(args, "autoencoder_checkpoint")
        and args.autoencoder_checkpoint is not None
    ):
        config.training.autoencoder_checkpoint = str(args.autoencoder_checkpoint)

    return config


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


def build_datamodule(data_config: dict[str, Any]) -> SpatioTemporalDataModule:
    """Instantiate the DataModule from a config dictionary (from Pydantic)."""
    # 1. Direct Instantiation (e.g. Encoded Datasets)
    if data_config.get("_target_"):
        log.info("Instantiating datamodule from target: %s", data_config["_target_"])
        return instantiate(data_config)

    # 2. Standard SpatioTemporalDataModule construction
    dm_cfg = data_config.get("datamodule", {})
    if dm_cfg is None:
        dm_cfg = {}

    # Extract params that go into constructor vs kwargs
    data_path = data_config.get("data_path")

    data = None
    if data_config.get("use_simulator"):
        simulator = instantiate(data_config.get("simulator"))
        data = _generate_split(simulator, data_config.get("split", {}))

    if data_path is None and data is None:
        msg = "Either 'data_path' or 'use_simulator' must be provided."
        raise ValueError(msg)

    # Process kwargs
    batch_size = dm_cfg.pop("batch_size", 4)
    dtype = _as_dtype(dm_cfg.pop("dtype", "float32"))
    ftype = dm_cfg.pop("ftype", "torch")
    dataset_cls = dm_cfg.pop("dataset_cls", SpatioTemporalDataset)

    return SpatioTemporalDataModule(
        data_path=data_path,
        data=data,
        dataset_cls=dataset_cls,
        batch_size=batch_size,
        dtype=dtype,
        ftype=ftype,
        **dm_cfg,
    )


def resolve_auto_params(
    config: Config, input_shape: tuple, output_shape: tuple
) -> Config:
    """Resolve 'auto' values in the configuration using inferred data shapes."""
    # Resolve Steps
    if config.training.n_steps_input == "auto":
        config.training.n_steps_input = input_shape[1]

    if config.training.n_steps_output == "auto":
        config.training.n_steps_output = output_shape[1]

    # Resolve Stride
    if config.training.stride == "auto":
        config.training.stride = config.training.n_steps_output

    # Resolve Rollout Stride
    if config.training.rollout_stride == "auto":
        config.training.rollout_stride = config.training.stride

    return config
