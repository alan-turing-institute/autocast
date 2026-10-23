"""Resolve train-only moment gradients by mode and ensemble direction.

This is an inference/output-gradient diagnostic, not training or a parameter
gradient estimate. Only load trusted full checkpoints (the loader uses pickle).
No validation/test windows or simulator redraws are opened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lightning as L
import torch
from einops import reduce
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from autocast.data.dataset import SpatioTemporalDataset
from autocast.losses.moments import SpectralMomentLoss
from autocast.scripts.execution import extract_state_dict, load_checkpoint_payload
from autocast.scripts.setup import setup_epd_model
from autocast.types import collate_batches

ROOT = Path(os.environ.get("AUTOCAST_ROOT", Path(__file__).resolve().parents[2]))
_HELPER = runpy.run_path(str(ROOT / "scripts/swe64s/calibrate_moment_weight.py"))
compose_config = _HELPER["compose_config"]
validate_source = _HELPER["validate_source"]
_sha256 = _HELPER["_sha256"]
BASELINE = _HELPER["BASELINE"]
PRESET = _HELPER["PRESET"]
COMPONENTS = ("physical", "fourier", "moment_mean", "moment_power")


def validate_resolved_source(source: DictConfig, candidate: DictConfig) -> None:
    """Resolve only the known three-field decoder placeholder before validation."""
    expected = deepcopy(candidate)
    if expected.model.decoder.output_channels not in (None, 3):
        msg = "The matched SWE task requires three decoder output channels."
        raise ValueError(msg)
    # Saved training configurations resolve this nullable shape placeholder.
    # Do not turn arbitrary None values into wildcards in the shared validator.
    expected.model.decoder.output_channels = 3
    validate_source(source, expected)


def radial_shells(height: int, width: int, *, device: torch.device) -> Tensor:
    """Return floor-radius shells in integer domain Fourier wavenumbers."""
    ky = torch.fft.fftfreq(height, device=device) * height
    kx = torch.fft.fftfreq(width, device=device) * width
    return torch.sqrt(ky[:, None].square() + kx[None, :].square()).floor().long()


def ensemble_parts(gradient: Tensor) -> dict[str, Tensor]:
    """Split a gradient orthogonally, retaining the same member-axis measure."""
    common = gradient.mean(dim=-1, keepdim=True).expand_as(gradient)
    return {"total": gradient, "common": common, "differential": gradient - common}


def _shell_sum(value: Tensor, shells: Tensor) -> Tensor:
    """Sum a HWC density into channel/shell bins, normalized by grid size."""
    result = value.new_zeros((value.shape[-1], int(shells.max()) + 1))
    for channel in range(value.shape[-1]):
        result[channel].scatter_add_(0, shells.flatten(), value[..., channel].flatten())
    return result / shells.numel()


def _density(left: Tensor, right: Tensor) -> Tensor:
    return reduce((left.conj() * right).real, "b t h w c m -> h w c", "mean")


def _cosine(dot: float, left_energy: float, right_energy: float) -> float | None:
    denominator = math.sqrt(left_energy * right_energy)
    return max(-1.0, min(1.0, dot / denominator)) if denominator > 0 else None


def gradient_statistics(
    gradient: Tensor,
    baseline: Tensor,
    prediction: Tensor,
    *,
    coefficient: float,
) -> dict[str, Any]:
    """Measure normalized-output gradients, not finite-step network changes.

    Shell energies sum to per-channel mean-square gradients (Parseval). The
    differential part is the direction that can change member anomalies when
    outputs are free variables. Its signed anomaly-power derivative is per unit
    output-space gradient-descent step, not a prediction of optimizer behavior.
    """
    if gradient.ndim != 6 or gradient.shape != baseline.shape:
        msg = "Expected matching BTHWCM gradient and baseline tensors."
        raise ValueError(msg)
    if prediction.shape != gradient.shape or gradient.shape[-1] < 2:
        msg = "Prediction and gradient must match, with at least two members."
        raise ValueError(msg)
    if not math.isfinite(coefficient) or coefficient < 0:
        msg = "coefficient must be finite and nonnegative."
        raise ValueError(msg)
    if not all(
        torch.isfinite(value).all() for value in (gradient, baseline, prediction)
    ):
        msg = "Diagnostic requires finite predictions and gradients."
        raise ValueError(msg)
    shells = radial_shells(gradient.shape[2], gradient.shape[3], device=gradient.device)
    anomaly = prediction - prediction.mean(dim=-1, keepdim=True)
    anomaly_fft = torch.fft.fftn(anomaly, dim=(2, 3), norm="ortho")
    result: dict[str, Any] = {"coefficient": coefficient, "parts": {}}
    base_parts = ensemble_parts(baseline)
    for name, part in ensemble_parts(gradient).items():
        spectrum = torch.fft.fftn(part, dim=(2, 3), norm="ortho")
        base_spectrum = torch.fft.fftn(base_parts[name], dim=(2, 3), norm="ortho")
        energy = _shell_sum(_density(spectrum, spectrum), shells)
        base_energy = _shell_sum(_density(base_spectrum, base_spectrum), shells)
        dot = _shell_sum(_density(spectrum, base_spectrum), shells)
        anomaly_dot = _shell_sum(_density(spectrum, anomaly_fft), shells)
        total_energy = float(energy.sum(dim=-1).mean())
        total_base_energy = float(base_energy.sum(dim=-1).mean())
        total_dot = float(dot.sum(dim=-1).mean())
        result["parts"][name] = {
            "gradient_rms": math.sqrt(total_energy),
            "weighted_gradient_rms": coefficient * math.sqrt(total_energy),
            "baseline_gradient_rms": math.sqrt(total_base_energy),
            "weighted_rms_ratio_to_baseline": (
                coefficient * math.sqrt(total_energy / total_base_energy)
                if total_base_energy > 0
                else None
            ),
            "cosine_to_baseline": _cosine(total_dot, total_energy, total_base_energy),
            "gradient_energy_channel_shell": energy.cpu().tolist(),
            "weighted_gradient_energy_channel_shell": (coefficient**2 * energy)
            .cpu()
            .tolist(),
            "baseline_gradient_energy_channel_shell": base_energy.cpu().tolist(),
            "dot_with_baseline_channel_shell": dot.cpu().tolist(),
            # Zero for the common part, up to roundoff. The total and differential
            # parts agree because anomalies sum to zero along the member axis.
            "anomaly_power_descent_derivative_channel_shell": (
                -2 * coefficient * anomaly_dot
            )
            .cpu()
            .tolist(),
        }
    return result


def diagnose_batch(
    physical: nn.Module,
    fourier: nn.Module,
    moments: SpectralMomentLoss,
    prediction: Tensor,
    target: Tensor,
    *,
    weights: tuple[float, ...] = (0.0, 0.0005, 0.002),
) -> dict[str, Any]:
    """Separate all four scalar components on the exact same M-member draw."""
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        msg = "weights must be a nonempty set of finite nonnegative values."
        raise ValueError(msg)
    value = prediction.detach().requires_grad_(True)
    observed = target.detach()
    mean, power = moments.components(value, observed)
    scores = (physical(value, observed), fourier(value, observed), mean, power)
    if any(score.ndim or not torch.isfinite(score) for score in scores):
        msg = "Diagnostic requires finite scalar component losses."
        raise ValueError(msg)
    gradients = dict(
        zip(
            COMPONENTS,
            (
                torch.autograd.grad(score, value, retain_graph=index < len(scores) - 1)[
                    0
                ]
                for index, score in enumerate(scores)
            ),
            strict=True,
        )
    )
    baseline = gradients["physical"] + 0.1 * gradients["fourier"]
    joint = gradients["moment_mean"] + gradients["moment_power"]
    return {
        "losses": {
            name: float(score.detach())
            for name, score in zip(COMPONENTS, scores, strict=True)
        },
        "components": {
            name: gradient_statistics(gradient, baseline, value.detach(), coefficient=w)
            for name, gradient, w in (
                ("physical", gradients["physical"], 1.0),
                ("fourier", gradients["fourier"], 0.1),
                ("baseline", baseline, 1.0),
                ("moment_mean", gradients["moment_mean"], 1.0),
                ("moment_power", gradients["moment_power"], 1.0),
            )
        },
        "weighted_joint_moments": {
            str(weight): gradient_statistics(
                joint, baseline, value.detach(), coefficient=weight
            )
            for weight in weights
        },
    }


def validate_checkpoint(
    checkpoint: Path, *, expected_sha256: str, expected_step: int
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Fail closed on changed artifacts or counters; load raw model state only."""
    actual_hash = _sha256(checkpoint)
    if actual_hash != expected_sha256:
        msg = "Checkpoint SHA256 does not match the approved artifact."
        raise ValueError(msg)
    payload = load_checkpoint_payload(checkpoint)
    if int(payload.get("global_step", -1)) != expected_step:
        msg = "Checkpoint step does not match the approved artifact."
        raise ValueError(msg)
    raw = extract_state_dict(payload)
    return raw, {"path": str(checkpoint), "sha256": actual_hash, "step": expected_step}


def prepare_diagnostic(  # noqa: PLR0915
    *,
    baseline_config: Path,
    baseline_checkpoint: Path,
    baseline_sha256: str,
    moments_config: Path,
    moments_checkpoint: Path,
    moments_sha256: str,
    data_path: Path,
    device: str,
    batch_count: int = 8,
    draws_per_batch: int = 2,
    seed: int = 153164,
) -> dict[str, Any]:
    """Compare raw 36k control/34k moment outputs on identical train draws."""
    if batch_count < 1 or draws_per_batch < 1 or batch_count * 32 > 64000:
        msg = "Use positive batch/draw counts within the fixed training split."
        raise ValueError(msg)
    manifest_path = ROOT / "local_hydra/local_experiment/swe64s/dataset.json"
    manifest = json.loads(manifest_path.read_text())
    train_path, stats_path = data_path / "train/data.pt", data_path / "stats.yml"
    train_hash, stats_hash = _sha256(train_path), _sha256(stats_path)
    if train_hash != manifest["sha256"]["train/data.pt"]:
        msg = "Training split differs from the fixed SWE64 benchmark."
        raise ValueError(msg)
    # Instantiate the train dataset directly: a full datamodule opens held-out data.
    dataset = SpatioTemporalDataset(
        data_path=str(train_path),
        n_steps_input=1,
        n_steps_output=1,
        stride=1,
        use_normalization=True,
        normalization_path=str(stats_path),
    )
    if len(dataset) != 64000:
        msg = "Expected 64,000 one-step training windows."
        raise ValueError(msg)
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[: batch_count * 32].reshape(batch_count, 32)
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": collate_batches([dataset[0]]),
    }
    baseline_cfg = compose_config(BASELINE)
    physical, fourier = (
        instantiate(item).to(device) for item in baseline_cfg.model.loss_func.losses
    )
    moments = SpectralMomentLoss().to(device)
    stages = {}
    for name, preset, config_path, checkpoint, sha, step in (
        (
            "baseline",
            BASELINE,
            baseline_config,
            baseline_checkpoint,
            baseline_sha256,
            36000,
        ),
        ("moments", PRESET, moments_config, moments_checkpoint, moments_sha256, 34000),
    ):
        cfg = compose_config(
            preset, [f"datamodule.data_path={json.dumps(str(data_path))}"]
        )
        source = OmegaConf.load(config_path)
        if not isinstance(source, DictConfig):
            msg = "Saved source configuration must be a mapping."
            raise ValueError(msg)
        validate_resolved_source(source, cfg)
        if (
            int(cfg.seed),
            int(cfg.datamodule.batch_size),
            int(cfg.model.n_members),
        ) != (42, 32, 8):
            msg = "Expected seed 42, batch 32, M=8."
            raise ValueError(msg)
        if (
            _sha256(Path(str(source.datamodule.normalization_path)).expanduser())
            != stats_hash
        ):
            msg = "Source normalization differs from the training split."
            raise ValueError(msg)
        raw, checkpoint_record = validate_checkpoint(
            checkpoint, expected_sha256=sha, expected_step=step
        )
        L.seed_everything(42, workers=True)
        model = setup_epd_model(
            cfg, stats, cast(Any, SimpleNamespace(train_dataset=dataset))
        )
        model.load_state_dict(raw, strict=True)
        del raw
        model.eval().to(device)
        rows = []
        for batch_index, selected in enumerate(indices):
            batch = collate_batches([dataset[int(index)] for index in selected]).to(
                device
            )
            for draw in range(draws_per_batch):
                noise_seed = seed + 1 + batch_index * draws_per_batch + draw
                torch.manual_seed(noise_seed)
                with torch.no_grad():
                    prediction = model(batch)
                if prediction.shape != (*batch.output_fields.shape, 8):
                    msg = "Expected eight output ensemble members."
                    raise ValueError(msg)
                row = diagnose_batch(
                    physical, fourier, moments, prediction, batch.output_fields
                )
                row.update(
                    batch_index=batch_index,
                    draw=draw,
                    prediction_seed=noise_seed,
                    training_window_indices=selected.tolist(),
                )
                rows.append(row)
        stages[name] = {
            "checkpoint": checkpoint_record,
            "source_config": str(config_path),
            "source_config_sha256": _sha256(config_path),
            "rows": rows,
        }
        del model
    return {
        "protocol": "swe64s-moment-gradient-directions-v1",
        "stages": stages,
        "channels": ["h", "u", "v"],
        "members": 8,
        "batch_size": 32,
        "batch_count": batch_count,
        "draws_per_batch": draws_per_batch,
        "seed": seed,
        "train_path": str(train_path),
        "train_sha256": train_hash,
        "normalization_sha256": stats_hash,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": _sha256(Path(__file__)),
        "checkpoint_weights": "raw",
        "split_used": "train",
        "training_executed": False,
        "optimizer_state_restored": False,
        "jobs_submitted": False,
        "shell_definition": "floor(sqrt(kx^2+ky^2)); full orthonormal spatial FFT",
        "gradient_coordinates": (
            "normalized reconstructed full-state predictions BTHWCM"
        ),
        "energy_definition": (
            "Sum over shell modes / HW; mean over batch, lead and member; "
            "channel separate"
        ),
        "anomaly_descent_definition": (
            "-2 lambda Re[FFT(anomaly)*conj(FFT(output_gradient))], "
            "same reduction; negative reduces power locally"
        ),
        "limits": (
            "Output-space infinitesimal diagnostics, not parameter/optimizer "
            "response. Repeated noise draws of the same windows are not "
            "independent examples. No held-out data or redraws used."
        ),
    }


def main() -> None:
    """Write a new diagnostic JSON without overwriting earlier evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "baseline-config",
        "baseline-checkpoint",
        "moments-config",
        "moments-checkpoint",
        "data-path",
        "output",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("baseline-sha256", "moments-sha256"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-count", type=int, default=8)
    parser.add_argument("--draws-per-batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=153164)
    args = vars(parser.parse_args())
    output = args.pop("output").expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    for key, value in args.items():
        if isinstance(value, Path):
            args[key] = value.expanduser().resolve()
    result = prepare_diagnostic(**args)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
