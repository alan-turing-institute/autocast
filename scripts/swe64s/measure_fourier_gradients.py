r"""Measure SWE64 decoded-output gradients at fixed loss weights on training windows.

Pass a trusted model checkpoint and its resolved training config, or the matching
local source preset. --train-data accepts a torch training split or a saved
training-window probe (data plus training_window_indices). Example from this tree:

uv run python scripts/swe64s/measure_fourier_gradients.py \
    --checkpoint parent.ckpt --source-config resolved_config.yaml \
    --train-data /path/to/train1000/train/data.pt \
    --normalization /path/to/train200/stats.yml --output gradients.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, cast

import torch
from calibrate_anomaly_weight import compose_config
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from autocast.data.dataset import SpatioTemporalDataset
from autocast.losses import WeightedSumLoss
from autocast.scripts.execution import extract_state_dict, load_checkpoint_payload
from autocast.scripts.setup import setup_epd_model
from autocast.types import collate_batches

PRESETS = {
    "crps_fft": "campaign_nested_crps_fft",
    "joint": "campaign_nested_joint_fourier_energy",
    "pairs": "campaign_nested_joint_fourier_pairs",
    "triads": "campaign_nested_joint_fourier_triads",
}


def measure_batch(
    objectives: dict[str, WeightedSumLoss], members: Tensor, truth: Tensor
) -> dict[str, Any]:
    """Compare all terms on a single frozen prediction. Ratios use output RMS."""
    if members.shape != (*truth.shape, 8) or truth.shape[-3:] != (64, 64, 3):
        msg = "Expected normalized 64x64 h/u/v fields and eight members."
        raise ValueError(msg)
    prediction = members.detach().requires_grad_(True)
    target = truth.detach()
    grid_weight = objectives["crps_fft"].weights[0]
    grid = objectives["crps_fft"].losses[0](prediction, target)
    (grid_grad,) = torch.autograd.grad(grid_weight * grid, prediction)
    grid_rms = grid_grad.square().mean().sqrt()
    if not torch.isfinite(grid_rms) or grid_rms <= 0:
        msg = "The weighted grid gradient must have finite, nonzero RMS."
        raise ValueError(msg)
    rows = {}
    for name, objective in objectives.items():
        spectral = objective.losses[1](prediction, target)
        (grad,) = torch.autograd.grad(spectral, prediction)
        rms = grad.square().mean().sqrt()
        if not torch.isfinite(spectral) or not torch.isfinite(rms) or rms <= 0:
            raise ValueError(
                f"Nonfinite score or invalid spectral gradient for {name}."
            )
        weight = objective.weights[1]
        rows[name] = {
            "spectral_loss": spectral.item(),
            "weight": weight,
            "raw_spectral_gradient_rms": rms.item(),
            "weighted_spectral_to_grid_rms": (weight * rms / grid_rms).item(),
            "cosine_with_grid": ((grad * grid_grad).mean() / (rms * grid_rms)).item(),
            "weighted_output_gradient_rms_by_huv": (
                weight * grad.square().mean(dim=(0, 1, 2, 3, 5)).sqrt()
            ).tolist(),
        }
    return {
        "grid_loss": grid.item(),
        "weighted_grid_gradient_rms": grid_rms.item(),
        "scores": rows,
    }


def training_windows(
    args: argparse.Namespace,
) -> tuple[SpatioTemporalDataset, list[int]]:
    """Gather sampled one-step windows from the memory-mapped training split."""
    payload = torch.load(
        args.train_data, mmap=True, map_location="cpu", weights_only=True
    )
    data = payload["data"]
    if data.ndim != 5 or data.shape[1] < 2 or data.shape[2:] != (64, 64, 3):
        msg = "Expected a SWE64 training tensor (trajectory, time, X, Y, h/u/v)."
        raise ValueError(msg)
    steps = data.shape[1] - 1
    count = args.batches * args.batch_size
    available = len(data) * steps
    if min(args.batches, args.batch_size) < 1 or count > available:
        raise ValueError(
            f"Requested {count} distinct windows from {available} available."
        )
    rng = torch.Generator().manual_seed(args.seed)
    selected = torch.randperm(available, generator=rng)[:count]
    traj, time = selected // steps, selected % steps
    windows = torch.stack((data[traj, time], data[traj, time + 1]), dim=1)
    dataset = SpatioTemporalDataset(
        data_path=None,
        data={"data": windows},
        use_normalization=True,
        normalization_path=str(args.normalization),
    )
    ids = payload.get("training_window_indices")
    if ids is not None:
        if steps != 1 or len(ids) != available:
            msg = "Exported training-window indices must match the one-step probes."
            raise ValueError(msg)
        return dataset, [int(ids[i]) for i in selected.tolist()]
    return dataset, selected.tolist()


def measure_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Generate one shared ensemble per batch from the raw parent weights."""
    dataset, indices = training_windows(args)
    cfg = (
        OmegaConf.load(args.source_config)
        if args.source_config
        else compose_config(args.source_preset)
    )
    if not isinstance(cfg, DictConfig):
        msg = "The source configuration must be a mapping."
        raise TypeError(msg)
    example = collate_batches([dataset[0]])
    model = setup_epd_model(
        cfg,
        {
            "channel_count": 3,
            "n_steps_input": 1,
            "n_steps_output": 1,
            "example_batch": example,
        },
        cast(Any, SimpleNamespace(train_dataset=dataset)),
    )
    checkpoint = load_checkpoint_payload(args.checkpoint)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    step = checkpoint.get("global_step")
    del checkpoint
    model.eval().to(args.device)
    objectives = {}
    sampling = {}
    for name, preset in PRESETS.items():
        config = compose_config(preset)
        if name in ("pairs", "triads"):
            if args.triads_per_draw is not None:
                config.joint_fourier_score.triads_per_draw = args.triads_per_draw
            sampling = {
                "triads_per_draw": config.joint_fourier_score.triads_per_draw,
                "tuple_seed": config.joint_fourier_score.seed,
            }
        objectives[name] = instantiate(config.model.loss_func).to(args.device)
    batches = []
    for number in range(args.batches):
        start = number * args.batch_size
        end = start + args.batch_size
        batch = collate_batches([dataset[i] for i in range(start, end)]).to(args.device)
        torch.manual_seed(args.seed + number + 1)
        with torch.no_grad():
            prediction = model(batch)
        result = measure_batch(objectives, prediction, batch.output_fields)
        batches.append({**result, "training_window_indices": indices[start:end]})
    ratios = {
        name: median(
            b["scores"][name]["weighted_spectral_to_grid_rms"] for b in batches
        )
        for name in PRESETS
    }
    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "checkpoint_weights": "raw",
        "source_config_or_preset": str(args.source_config or args.source_preset),
        "training_data": str(args.train_data),
        "normalization": str(args.normalization),
        "sample_seed": args.seed,
        "members": 8,
        "batch_size": args.batch_size,
        **sampling,
        "median_weighted_spectral_to_grid_rms": ratios,
        "batches": batches,
    }


def main() -> None:
    """Write measurements at the configured weights without fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-config", type=Path)
    source.add_argument("--source-preset", default=PRESETS["crps_fft"])
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=153164)
    parser.add_argument("--triads-per-draw", type=int)
    args = parser.parse_args()
    result = measure_checkpoint(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["median_weighted_spectral_to_grid_rms"], indent=2))


if __name__ == "__main__":
    main()
