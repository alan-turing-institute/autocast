"""Choose one fixed auxiliary coefficient for the fresh V1/V2 controls.

Inference and output-gradient measurements on training windows only. No
optimizer, held-out examples, simulator redraws or training are used. The
coefficient caps each auxiliary's gradient RMS relative to the unchanged base
objective on these draws; it is not a bound on future optimization dynamics.
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
from omegaconf import OmegaConf
from torch import Tensor, nn

from autocast.data.dataset import SpatioTemporalDataset
from autocast.losses.vector import VelocityScoreLoss
from autocast.scripts.setup import setup_epd_model
from autocast.types import collate_batches

ROOT = Path(__file__).resolve().parents[2]
_HELPERS = runpy.run_path(str(Path(__file__).with_name("calibrate_anomaly_weight.py")))
compose_config = _HELPERS["compose_config"]
_sha256 = _HELPERS["_sha256"]
BASELINE = "crps_fft_spatial1d_residual_one_output_12block_256ep_8h"
PRESETS = (
    "crps_fft_joint_velocity_spatial1d_256ep",
    "crps_fft_helmholtz_velocity_spatial1d_256ep",
)


def calibrate_batches(
    base: nn.Module,
    auxiliaries: dict[str, nn.Module],
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    gradient_fraction: float = 0.1,
) -> dict[str, Any]:
    """Use the most restrictive output-gradient ratio across both controls."""
    if not math.isfinite(gradient_fraction) or not 0 < gradient_fraction <= 1:
        msg = "gradient_fraction must be finite and in (0, 1]."
        raise ValueError(msg)
    if not auxiliaries:
        msg = "At least one auxiliary is required."
        raise ValueError(msg)
    rows = []
    candidates = []
    for batch_index, (provided, provided_target) in enumerate(batches):
        prediction = provided.detach().requires_grad_(True)
        target = provided_target.detach()
        primary = base(prediction, target)
        if primary.ndim or not torch.isfinite(primary):
            msg = "Base loss must be a finite scalar."
            raise ValueError(msg)
        (base_gradient,) = torch.autograd.grad(primary, prediction)
        base_rms = float(base_gradient.square().mean().sqrt())
        if not math.isfinite(base_rms) or base_rms <= 0:
            msg = "Calibration requires a finite nonzero base gradient."
            raise ValueError(msg)
        for name, auxiliary in auxiliaries.items():
            score = auxiliary(prediction, target)
            if score.ndim or not torch.isfinite(score):
                msg = "Auxiliary loss must be a finite scalar."
                raise ValueError(msg)
            (gradient,) = torch.autograd.grad(score, prediction)
            rms = float(gradient.square().mean().sqrt())
            if not math.isfinite(rms) or rms <= 0:
                msg = "Calibration requires finite nonzero auxiliaries."
                raise ValueError(msg)
            candidate = gradient_fraction * base_rms / rms
            if not math.isfinite(candidate) or candidate <= 0:
                msg = "Cannot select a finite positive shared weight."
                raise ValueError(msg)
            candidates.append(candidate)
            rows.append(
                {
                    "batch_index": batch_index,
                    "auxiliary": name,
                    "base_loss": float(primary.detach()),
                    "auxiliary_loss": float(score.detach()),
                    "base_gradient_rms": base_rms,
                    "auxiliary_gradient_rms": rms,
                    "candidate_weight": candidate,
                    "gradient_cosine": float(
                        (gradient * base_gradient).mean() / (rms * base_rms)
                    ),
                }
            )
    if not rows:
        msg = "At least one training batch is required."
        raise ValueError(msg)
    weight = min(candidates)
    for row in rows:
        row["weighted_gradient_ratio"] = (
            weight * row["auxiliary_gradient_rms"] / row["base_gradient_rms"]
        )
    return {
        "weight": weight,
        "weight_override": f"vector_weight={weight:.17g}",
        "gradient_fraction": gradient_fraction,
        "rows": rows,
    }


def check_velocity_statistics(
    auxiliary: VelocityScoreLoss, channel_stds: Tensor
) -> None:
    """Reject stale preset scaling before a dataset-backed preflight."""
    actual = channel_stds[list(auxiliary.channels)].double().cpu()
    expected = torch.tensor(auxiliary.channel_stds, dtype=torch.float64)
    if not torch.allclose(actual, expected, rtol=1e-6, atol=0):
        msg = "Vector preset scales do not match training normalization."
        raise ValueError(msg)


def prepare_calibration(
    data_path: Path,
    device: str,
    gradient_fraction: float = 0.1,
    batch_count: int = 4,
    seed: int = 153164,
) -> dict[str, Any]:
    """Check one shared fresh-model coefficient using four train batches."""
    if not 1 <= batch_count <= 8:
        msg = "Use one to eight training batches for this bounded check."
        raise ValueError(msg)
    cfg = compose_config(
        BASELINE, [f"datamodule.data_path={json.dumps(str(data_path))}"]
    )
    base = instantiate(cfg.model.loss_func).to(device)
    auxiliaries = {
        name: instantiate(compose_config(preset).model.loss_func.losses[2]).to(device)
        for name, preset in zip(("joint", "helmholtz"), PRESETS, strict=True)
    }
    manifest_path = ROOT / "local_hydra/local_experiment/swe64s/dataset.json"
    manifest = json.loads(manifest_path.read_text())
    train_path, stats_path = data_path / "train/data.pt", data_path / "stats.yml"
    train_hash, stats_hash = _sha256(train_path), _sha256(stats_path)
    if train_hash != manifest["sha256"]["train/data.pt"]:
        msg = "Training split does not match the fixed SWE64 benchmark."
        raise ValueError(msg)
    dataset = SpatioTemporalDataset(
        data_path=str(train_path),
        n_steps_input=1,
        n_steps_output=1,
        stride=1,
        use_normalization=True,
        normalization_path=str(stats_path),
    )
    if len(dataset) != 64000 or dataset.norm is None:
        msg = "Expected the normalized 64,000-window training split."
        raise ValueError(msg)
    for auxiliary in auxiliaries.values():
        check_velocity_statistics(auxiliary, dataset.norm.flattened_stds["variable"])
    batch_size, members, model_seed = (
        int(cfg.datamodule.batch_size),
        int(cfg.model.n_members),
        int(cfg.seed),
    )
    if (batch_size, members, model_seed) != (32, 8, 42):
        msg = "Use the matched batch32, eight-member, seed42 setup."
        raise ValueError(msg)
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[: batch_count * batch_size].reshape(batch_count, batch_size)
    L.seed_everything(model_seed, workers=True)
    model = (
        setup_epd_model(
            cfg,
            {
                "channel_count": 3,
                "n_steps_input": 1,
                "n_steps_output": 1,
                "example_batch": collate_batches([dataset[0]]),
            },
            cast(Any, SimpleNamespace(train_dataset=dataset)),
        )
        .eval()
        .to(device)
    )

    def batches() -> Iterable[tuple[Tensor, Tensor]]:
        for number, selected in enumerate(indices):
            batch = collate_batches([dataset[int(i)] for i in selected]).to(device)
            torch.manual_seed(seed + number + 1)
            with torch.no_grad():
                prediction = model(batch)
            yield prediction, batch.output_fields

    result = calibrate_batches(
        base, auxiliaries, batches(), gradient_fraction=gradient_fraction
    )
    return {
        **result,
        "protocol": "swe64s-vector-weight-v1",
        "presets": PRESETS,
        "split_used": "train",
        "training_window_indices": indices.tolist(),
        "prediction_seeds": [seed + n + 1 for n in range(batch_count)],
        "model_seed": model_seed,
        "batch_size": batch_size,
        "members": members,
        "train_path": str(train_path),
        "train_sha256": train_hash,
        "normalization_sha256": stats_hash,
        "dataset_uuid": manifest["uuid"],
        "auxiliaries": {
            name: OmegaConf.to_container(
                compose_config(preset).model.loss_func.losses[2], resolve=True
            )
            for name, preset in zip(auxiliaries, PRESETS, strict=True)
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "training_executed": False,
        "jobs_submitted": False,
        "limits": "Output gradients on fresh-model training draws only; not a "
        "parameter-gradient or future-stability guarantee. Use the same weight "
        "in both runs without renormalizing their component scores.",
    }


def main() -> None:
    """Write a new private preflight record without starting a fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gradient-fraction", type=float, default=0.1)
    parser.add_argument("--batch-count", type=int, default=4)
    args = vars(parser.parse_args())
    output = args.pop("output").expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    args["data_path"] = args["data_path"].expanduser().resolve()
    result = prepare_calibration(**args)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["weight_override"])


if __name__ == "__main__":
    main()
