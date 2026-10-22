"""Calibrate one joint-moment weight using training windows and inference only.

Use a fresh seed-42 control and the raw healthy 36,000-step control checkpoint.
The checkpoint supplies calibration outputs only: the eventual 256-epoch run
must initialize independently from scratch. Never train or open held-out splits.
Only load trusted checkpoints: the standard full checkpoint loader uses pickle.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
import subprocess
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from autocast.data.dataset import SpatioTemporalDataset
from autocast.losses.moments import SpectralMomentLoss
from autocast.scripts.execution import extract_state_dict, load_checkpoint_payload
from autocast.scripts.setup import setup_epd_model
from autocast.types import collate_batches

ROOT = Path(__file__).resolve().parents[2]
_CONTROL = runpy.run_path(str(Path(__file__).with_name("calibrate_anomaly_weight.py")))
compose_config = _CONTROL["compose_config"]
validate_source = _CONTROL["validate_source"]
_sha256 = _CONTROL["_sha256"]
BASELINE = _CONTROL["BASELINE"]
PRESET = "crps_fft_moments_spatial1d_residual_one_output_12block_256ep_8h"
BATCH_COUNT = 8
CHECKPOINT_STEP = 36000


def calibrate_batches(
    base: nn.Module,
    auxiliary: SpectralMomentLoss,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    gradient_fraction: float = 0.1,
) -> dict[str, Any]:
    """Cap the joint/base output-gradient RMS ratio on every supplied train batch.

    Gradients refer to the same normalized predicted fields, not model
    parameters. The mean and power components retain unit coefficients; their
    gradient vectors are summed before measuring the joint RMS. Negative
    corrected scores and zero individual component gradients are allowed.
    """
    if not math.isfinite(gradient_fraction) or not 0 < gradient_fraction <= 1:
        msg = "gradient_fraction must be finite and in (0, 1]."
        raise ValueError(msg)
    rows = []
    for provided_prediction, provided_target in batches:
        prediction = provided_prediction.detach().requires_grad_(True)
        target = provided_target.detach()
        primary = base(prediction, target)
        mean_score, power_score = auxiliary.components(prediction, target)
        joint_score = mean_score + power_score
        scores = (primary, mean_score, power_score, joint_score)
        if any(score.ndim != 0 or not torch.isfinite(score) for score in scores):
            msg = "Calibration needs finite scalar losses."
            raise ValueError(msg)
        (primary_gradient,) = torch.autograd.grad(primary, prediction)
        (mean_gradient,) = torch.autograd.grad(
            mean_score, prediction, retain_graph=True
        )
        (power_gradient,) = torch.autograd.grad(power_score, prediction)
        joint_gradient = mean_gradient + power_gradient
        gradients = (primary_gradient, mean_gradient, power_gradient, joint_gradient)
        primary_rms, mean_rms, power_rms, joint_rms = (
            float(gradient.square().mean().sqrt()) for gradient in gradients
        )
        if (
            not all(
                math.isfinite(value)
                for value in (primary_rms, mean_rms, power_rms, joint_rms)
            )
            or min(primary_rms, joint_rms) <= 0
        ):
            msg = "Calibration needs finite gradients and nonzero base/joint gradients."
            raise ValueError(msg)
        candidate = gradient_fraction * primary_rms / joint_rms
        if not math.isfinite(candidate) or candidate <= 0:
            msg = "Calibration cannot select a finite positive weight."
            raise ValueError(msg)
        rows.append(
            {
                "base_loss": float(primary.detach()),
                "mean_loss": float(mean_score.detach()),
                "power_loss": float(power_score.detach()),
                "joint_loss": float(joint_score.detach()),
                "base_gradient_rms": primary_rms,
                "mean_gradient_rms": mean_rms,
                "power_gradient_rms": power_rms,
                "joint_gradient_rms": joint_rms,
                "candidate_weight": candidate,
            }
        )
    if not rows:
        msg = "At least one training batch is required for calibration."
        raise ValueError(msg)
    weight = min(row["candidate_weight"] for row in rows)
    for row in rows:
        row["weighted_gradient_ratio"] = (
            weight * row["joint_gradient_rms"] / row["base_gradient_rms"]
        )
    return {
        "weight": weight,
        "gradient_fraction": gradient_fraction,
        "rule": "Min over cold/warm train batches of fraction * base_RMS / joint_RMS",
        "weight_override": f"model.loss_func.weights.2={weight:.17g}",
        "rows": rows,
    }


def prepare_calibration(  # noqa: PLR0915
    *,
    source_config: Path,
    checkpoint: Path,
    data_path: Path,
    device: str,
    seed: int = 153164,
    gradient_fraction: float = 0.1,
) -> dict[str, Any]:
    """Infer cold/warm outputs for eight fixed, training-only batches of 32."""
    if not math.isfinite(gradient_fraction) or not 0 < gradient_fraction <= 0.1:
        msg = "The matched protocol requires gradient_fraction in (0, 0.1]."
        raise ValueError(msg)
    cfg = compose_config(
        BASELINE, [f"datamodule.data_path={json.dumps(str(data_path))}"]
    )
    source = OmegaConf.load(source_config)
    if not isinstance(source, DictConfig):
        msg = "Source configuration must be a mapping."
        raise ValueError(msg)
    validate_source(source, cfg)
    model_seed, batch_size, members = (
        int(cfg.seed),
        int(cfg.datamodule.batch_size),
        int(cfg.model.n_members),
    )
    if (model_seed, batch_size, members) != (42, 32, 8):
        msg = "The matched control requires model seed 42, batch size 32 and 8 members."
        raise ValueError(msg)
    if list(cfg.model.loss_func.weights) != [1.0, 0.1]:
        msg = "Keep the base CRPS-FFT weights at [1.0, 0.1]."
        raise ValueError(msg)
    auxiliary_cfg = compose_config(PRESET).model.loss_func.losses[2]
    auxiliary = instantiate(auxiliary_cfg).to(device)
    base = instantiate(cfg.model.loss_func).to(device)
    manifest_path = ROOT / "local_hydra/local_experiment/swe64s/dataset.json"
    manifest = json.loads(manifest_path.read_text())
    train_path, stats_path = data_path / "train/data.pt", data_path / "stats.yml"
    train_hash = _sha256(train_path)
    if train_hash != manifest["sha256"]["train/data.pt"]:
        msg = "Training split does not match the fixed SWE64 benchmark."
        raise ValueError(msg)
    source_hash, checkpoint_hash = _sha256(source_config), _sha256(checkpoint)
    stats_hash = _sha256(stats_path)
    source_stats_path = Path(str(source.datamodule.normalization_path)).expanduser()
    if _sha256(source_stats_path) != stats_hash:
        msg = "Calibration normalization differs from the selected control."
        raise ValueError(msg)
    payload = load_checkpoint_payload(checkpoint)
    checkpoint_step = int(payload.get("global_step", 0))
    if checkpoint_step != CHECKPOINT_STEP:
        msg = "Use the selected raw healthy 36,000-step baseline checkpoint."
        raise ValueError(msg)
    raw_state = extract_state_dict(payload)
    del payload
    # A full datamodule would also open validation/test splits.
    dataset = SpatioTemporalDataset(
        data_path=str(train_path),
        n_steps_input=1,
        n_steps_output=1,
        stride=1,
        use_normalization=True,
        normalization_path=str(stats_path),
    )
    if len(dataset) != 200 * 320:
        msg = "Expected 64,000 one-step training windows."
        raise ValueError(msg)
    # Selection uses its own RNG, independent of model initialization and noise.
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[: BATCH_COUNT * batch_size].reshape(BATCH_COUNT, batch_size)
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": collate_batches([dataset[0]]),
    }

    def predicted_batches() -> Iterable[tuple[Tensor, Tensor]]:
        for stage in ("cold", "warm"):
            # Match the real training entrypoint's initialization, not the data seed.
            L.seed_everything(model_seed, workers=True)
            model = setup_epd_model(
                cfg, stats, cast(Any, SimpleNamespace(train_dataset=dataset))
            )
            if stage == "warm":
                model.load_state_dict(raw_state, strict=True)
            model.eval().to(device)
            for number, selected in enumerate(indices):
                batch = collate_batches([dataset[int(index)] for index in selected]).to(
                    device
                )
                # Match the noise seeds across stages; each forward draws 8 members.
                torch.manual_seed(seed + number + 1)
                with torch.no_grad():
                    prediction = model(batch)
                if prediction.shape != (*batch.output_fields.shape, members):
                    msg = "Calibration requires eight ensemble members per target."
                    raise ValueError(msg)
                yield prediction, batch.output_fields
            del model

    result = calibrate_batches(
        base, auxiliary, predicted_batches(), gradient_fraction=gradient_fraction
    )
    for row, (stage, number) in zip(
        result["rows"],
        (
            (stage, number)
            for stage in ("cold", "warm")
            for number in range(BATCH_COUNT)
        ),
        strict=True,
    ):
        row.update(
            stage=stage,
            batch_index=number,
            training_window_indices=indices[number].tolist(),
            prediction_seed=seed + number + 1,
        )
    return {
        **result,
        "protocol": "swe64s-joint-spectral-moment-weight-v1",
        "preset": PRESET,
        "baseline_preset": BASELINE,
        "source_config": str(source_config),
        "source_config_sha256": source_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint_step,
        "checkpoint_role": "Warm predictions only; never initialize the fresh run",
        "checkpoint_weights": "raw state_dict",
        "train_path": str(train_path),
        "train_sha256": train_hash,
        "normalization_sha256": stats_hash,
        "source_normalization_path": str(source_stats_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "dataset_uuid": manifest["uuid"],
        "auxiliary_config": OmegaConf.to_container(auxiliary_cfg, resolve=True),
        "base_weights": [1.0, 0.1],
        "component_weights": {"mean": 1.0, "power": 1.0},
        "batch_count_per_stage": BATCH_COUNT,
        "batch_size": batch_size,
        "members": members,
        "seed": seed,
        "model_seed": model_seed,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "split_used": "train",
        "training_executed": False,
        "optimizer_state_restored": False,
        "jobs_submitted": False,
        "limits": "The cap holds on these fixed cold/warm training draws only, "
        "not throughout the independently initialized 256-epoch fit. No held-out "
        "examples, redraw selection, fitted component scales or radial bands are used.",
    }


def main() -> None:
    """Write a new calibration record and print its required Hydra override."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-config", "checkpoint", "data-path", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=153164)
    parser.add_argument("--gradient-fraction", type=float, default=0.1)
    args = vars(parser.parse_args())
    output = args.pop("output").expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    for key in ("source_config", "checkpoint", "data_path"):
        args[key] = args[key].expanduser().resolve()
    result = prepare_calibration(**args)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["weight_override"])


if __name__ == "__main__":
    main()
