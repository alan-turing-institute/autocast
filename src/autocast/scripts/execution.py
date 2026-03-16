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


def resolve_checkpoint_path(
    eval_cfg: Mapping[str, Any],
    work_dir: Path,
    *,
    missing_message: str,
) -> Path:
    """Resolve checkpoint path from eval config relative to eval/training work dirs."""
    checkpoint_value = eval_cfg.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError(missing_message)

    checkpoint_path = Path(str(checkpoint_value))
    if checkpoint_path.is_absolute():
        return checkpoint_path

    workdir_candidate = (work_dir / checkpoint_path).resolve()
    parent_candidate = (work_dir.parent / checkpoint_path).resolve()
    if workdir_candidate.exists():
        return workdir_candidate
    if parent_candidate.exists():
        return parent_candidate
    return workdir_candidate


def resolve_benchmark_csv_path(eval_cfg: Mapping[str, Any], work_dir: Path) -> Path:
    """Resolve benchmark CSV path from eval config and working directory."""
    benchmark_cfg = eval_cfg.get("benchmark", {})
    if not isinstance(benchmark_cfg, Mapping):
        benchmark_cfg = {}
    csv_path = benchmark_cfg.get("csv_path")
    if csv_path is not None:
        return Path(str(csv_path)).expanduser().resolve()
    return (work_dir / "benchmark_metrics.csv").resolve()


def benchmark_metric_rows(
    *,
    benchmark_type: str,
    checkpoint_path: Path,
    device: str,
    batch_size: int,
    n_warmup: int,
    n_benchmark: int,
    metrics: Mapping[str, float],
    stride: int | None = None,
    max_rollout_steps: int | None = None,
) -> list[dict[str, float | str | int | None]]:
    """Convert benchmark metrics mapping into flat CSV row dicts."""
    rows: list[dict[str, float | str | int | None]] = []
    for metric, value in metrics.items():
        rows.append(
            {
                "benchmark_type": benchmark_type,
                "checkpoint": str(checkpoint_path),
                "device": device,
                "batch_size": batch_size,
                "n_warmup": n_warmup,
                "n_benchmark": n_benchmark,
                "stride": stride,
                "max_rollout_steps": max_rollout_steps,
                "metric": metric,
                "value": float(value),
            }
        )
    return rows
