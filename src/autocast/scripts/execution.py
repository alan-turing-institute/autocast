"""Shared execution utilities for AutoCast scripts."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from hydra.core.hydra_config import HydraConfig


def resolve_hydra_work_dir(work_dir: Path | None) -> Path:
    """Resolve working directory, falling back to Hydra output dir then cwd.

    Priority:
    1. ``work_dir`` if provided.
    2. Hydra ``runtime.output_dir`` if Hydra is initialised.
    3. Current working directory.
    """
    if work_dir is not None:
        return work_dir

    if HydraConfig.initialized():
        output_dir = HydraConfig.get().runtime.output_dir
        if output_dir:
            return Path(output_dir).resolve()

    return Path.cwd()


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Parameters
    ----------
    device:
        Named device (``"cpu"``, ``"cuda"``, ``"mps"``) or ``"auto"`` to
        select the best available accelerator.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_payload(checkpoint_path: Path) -> Mapping[str, Any]:
    """Load a checkpoint file and return the raw payload mapping."""
    checkpoint_real = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_real,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        msg = f"Checkpoint {checkpoint_real} does not contain a valid payload."
        raise TypeError(msg)
    return checkpoint


def extract_state_dict(
    checkpoint: Mapping[str, Any],
) -> OrderedDict[str, torch.Tensor]:
    """Extract and clean the state dict from a checkpoint payload."""
    if isinstance(checkpoint, Mapping):
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        msg = "Checkpoint payload does not contain a valid state_dict."
        raise TypeError(msg)
    if isinstance(state_dict, OrderedDict):
        state_dict = state_dict.copy()
    else:
        state_dict = OrderedDict(state_dict)
    state_dict.pop("_metadata", None)
    return state_dict
