"""Prepare a train-only weight; never train, score held-out data or submit.

Only load trusted checkpoints: the standard full checkpoint loader uses pickle.
The existing spatial CRPS-FFT control supplies representative learned outputs,
not initial weights or optimizer state for the fresh auxiliary-loss experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from autocast.data.dataset import SpatioTemporalDataset
from autocast.losses import SpectralSpreadErrorLoss
from autocast.scripts.execution import extract_state_dict, load_checkpoint_payload
from autocast.scripts.setup import setup_epd_model
from autocast.types import collate_batches

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "crps_fft_spatial1d_residual_one_output_12block_256ep_8h"
PRESET = "crps_fft_anomaly_spatial1d_residual_one_output_12block_256ep_8h"


def compose_config(name: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose the matched preset without fitting or opening any datasets."""
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"local_experiment=swe64s/{name}",
                f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
                *(overrides or []),
            ],
        )


def _matches(expected: Any, actual: Any) -> bool:
    if expected == "auto":
        return True  # Shape placeholders may be resolved in saved configs.
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(_matches(value, actual[key]) for key, value in expected.items())
        )
    return expected == actual


def validate_source(source: DictConfig, baseline: DictConfig) -> None:
    """Reject a differently configured model or task as a calibration source."""
    if not _matches(
        OmegaConf.to_container(baseline.model, resolve=True),
        OmegaConf.to_container(source.model, resolve=True),
    ):
        msg = "Calibration source must match the spatial CRPS-FFT baseline model."
        raise ValueError(msg)
    for name in (
        "_target_",
        "batch_size",
        "n_steps_input",
        "n_steps_output",
        "stride",
        "start_frame",
        "use_normalization",
    ):
        if source.datamodule.get(name) != baseline.datamodule.get(name):
            raise ValueError(f"Calibration source changes datamodule.{name}.")


def calibrate_batches(
    base: nn.Module,
    auxiliary: SpectralSpreadErrorLoss,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    gradient_fraction: float = 0.1,
) -> dict[str, Any]:
    """Cap the auxiliary/base output-gradient RMS ratio on every input batch.

    This scales gradients with respect to the SAME normalized predicted fields,
    not parameter gradients or loss values. The finite-sample check is not a
    guarantee for future training. Constant/collapsed ensembles fail closed.
    """
    if not math.isfinite(gradient_fraction) or not 0 < gradient_fraction <= 1:
        msg = "gradient_fraction must be finite and in (0, 1]."
        raise ValueError(msg)
    rows = []
    for provided_prediction, provided_target in batches:
        prediction = provided_prediction.detach().requires_grad_(True)
        target = provided_target.detach()
        primary = base(prediction, target)
        extra = auxiliary(prediction, target)
        (primary_gradient,) = torch.autograd.grad(
            primary, prediction, retain_graph=True
        )
        (extra_gradient,) = torch.autograd.grad(extra, prediction)
        primary_rms = float(primary_gradient.square().mean().sqrt())
        extra_rms = float(extra_gradient.square().mean().sqrt())
        if not all(
            math.isfinite(v) for v in (float(primary.detach()), float(extra.detach()))
        ) or not all(math.isfinite(v) and v > 0 for v in (primary_rms, extra_rms)):
            msg = "Calibration needs finite losses and nonzero finite gradients."
            raise ValueError(msg)
        with torch.no_grad():
            spread, error = auxiliary.power_spectra(prediction, target)
            discrepancy = (1 + 1 / prediction.shape[-1]) * spread - error
        candidate = gradient_fraction * primary_rms / extra_rms
        if not math.isfinite(candidate) or candidate <= 0:
            msg = "Calibration cannot select a finite positive weight."
            raise ValueError(msg)
        rows.append(
            {
                "base_loss": float(primary.detach()),
                "auxiliary_loss": float(extra.detach()),
                "base_gradient_rms": primary_rms,
                "auxiliary_gradient_rms": extra_rms,
                "candidate_weight": candidate,
                "spread_power_t_band_c": spread.cpu().tolist(),
                "residual_power_t_band_c": error.cpu().tolist(),
                "squared_discrepancy_t_band_c": discrepancy.square().cpu().tolist(),
            }
        )
    if not rows:
        msg = "At least one training batch is required for calibration."
        raise ValueError(msg)
    weight = min(row["candidate_weight"] for row in rows)
    for row in rows:
        row["weighted_gradient_ratio"] = (
            weight * row["auxiliary_gradient_rms"] / row["base_gradient_rms"]
        )
    return {
        "weight": weight,
        "gradient_fraction": gradient_fraction,
        "rule": "Min over train batches of fraction * base_RMS / auxiliary_RMS",
        "weight_override": f"model.loss_func.weights.2={weight:.17g}",
        "rows": rows,
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_calibration(  # noqa: PLR0915
    *,
    source_config: Path,
    checkpoint: Path,
    data_path: Path,
    device: str,
    batch_count: int = 8,
    seed: int = 153164,
    gradient_fraction: float = 0.1,
) -> dict[str, Any]:
    """Read only training windows; run inference and output-space gradients."""
    if batch_count < 1:
        msg = "batch_count must be positive."
        raise ValueError(msg)
    cfg = compose_config(
        BASELINE, [f"datamodule.data_path={json.dumps(str(data_path))}"]
    )
    source = OmegaConf.load(source_config)
    if not isinstance(source, DictConfig):
        msg = "Source configuration must be a mapping."
        raise ValueError(msg)
    validate_source(source, cfg)
    auxiliary_cfg = compose_config(PRESET).model.loss_func.losses[2]
    auxiliary = instantiate(auxiliary_cfg)
    manifest = json.loads(
        (ROOT / "local_hydra/local_experiment/swe64s/dataset.json").read_text()
    )
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
    # Do not instantiate a full datamodule: that would also open held-out splits.
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
    batch_size = int(cfg.datamodule.batch_size)
    if batch_count * batch_size > len(dataset):
        msg = "Not enough unique training windows for calibration."
        raise ValueError(msg)
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[: batch_count * batch_size].reshape(batch_count, batch_size)
    example = collate_batches([dataset[0]])
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": example,
    }
    torch.manual_seed(seed)
    model = setup_epd_model(
        cfg, stats, cast(Any, SimpleNamespace(train_dataset=dataset))
    )
    payload = load_checkpoint_payload(checkpoint)
    checkpoint_step = int(payload.get("global_step", 0))
    if checkpoint_step <= 0:
        msg = "Use a trained control checkpoint, not a collapsed random initialization."
        raise ValueError(msg)
    model.load_state_dict(extract_state_dict(payload), strict=True)
    del payload
    model.eval().to(device)
    assert model.loss_func is not None

    def predicted_batches() -> Iterable[tuple[Tensor, Tensor]]:
        for number, selected in enumerate(indices):
            batch = collate_batches([dataset[int(index)] for index in selected]).to(
                device
            )
            torch.manual_seed(seed + number + 1)
            with torch.no_grad():
                prediction = model(batch)
            yield prediction, batch.output_fields

    result = calibrate_batches(
        model.loss_func,
        auxiliary,
        predicted_batches(),
        gradient_fraction=gradient_fraction,
    )
    for row, selected in zip(result["rows"], indices.tolist(), strict=True):
        row["training_window_indices"] = selected
    return {
        **result,
        "protocol": "swe64s-online-spectral-weight-v1",
        "preset": PRESET,
        "source_config": str(source_config),
        "source_config_sha256": source_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint_step,
        "train_path": str(train_path),
        "train_sha256": train_hash,
        "normalization_sha256": stats_hash,
        "source_normalization_path": str(source_stats_path),
        "dataset_uuid": manifest["uuid"],
        "auxiliary_config": OmegaConf.to_container(auxiliary_cfg, resolve=True),
        "batch_size": batch_size,
        "members": int(cfg.model.n_members),
        "seed": seed,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "split_used": "train",
        "training_executed": False,
        "optimizer_state_restored": False,
        "jobs_submitted": False,
        "limits": "A learned control sets an initial gradient scale, not a guarantee "
        "during the fresh fit. Mean error, process-local pooling and finite-batch "
        "effects remain. No held-out examples or redraws select this weight.",
    }


def main() -> None:
    """Write a new calibration record and print its required Hydra override."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-config", "checkpoint", "data-path", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-count", type=int, default=8)
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
