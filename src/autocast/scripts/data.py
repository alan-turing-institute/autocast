"""Shared data and simulator configuration for AutoCast scripts."""

import logging
from typing import Any

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from autocast.data.datamodule import SpatioTemporalDataModule, TheWellDataModule
from autocast.types import Batch

log = logging.getLogger(__name__)


def build_datamodule(
    config: DictConfig,
) -> SpatioTemporalDataModule | TheWellDataModule:
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


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """Move a Batch to the specified device."""
    return Batch(
        input_fields=batch.input_fields.to(device),
        output_fields=batch.output_fields.to(device),
        constant_scalars=(
            batch.constant_scalars.to(device)
            if batch.constant_scalars is not None
            else None
        ),
        constant_fields=(
            batch.constant_fields.to(device)
            if batch.constant_fields is not None
            else None
        ),
    )
