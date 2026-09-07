"""Evaluate fit and ensemble structure for shallow-water CRPS experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from einops import rearrange
from omegaconf import DictConfig, open_dict

from autocast.scripts.config import save_resolved_config
from autocast.scripts.eval.encoder_processor_decoder import (
    _resolve_teacher_forcing_ratio,
)
from autocast.scripts.execution import (
    extract_state_dict,
    load_checkpoint_payload,
    resolve_checkpoint_path,
    resolve_device,
    resolve_hydra_work_dir,
)
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.utils import get_default_config_path
from autocast.types import Batch

log = logging.getLogger(__name__)


def _channel_values(values: torch.Tensor) -> list[float]:
    return values.detach().cpu().tolist()


def compute_fit_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
) -> dict[str, list[float] | float]:
    """Compute channelwise ensemble fit metrics for physical-space tensors."""
    if prediction.ndim != truth.ndim + 1:
        msg = "prediction must have one trailing ensemble dimension"
        raise ValueError(msg)
    if prediction.shape[:-1] != truth.shape:
        msg = "prediction and truth shapes do not match"
        raise ValueError(msg)

    ensemble_mean = prediction.mean(dim=-1)
    channel_reduction_dims = tuple(range(truth.ndim - 1))
    member_reduction_dims = (*channel_reduction_dims, prediction.ndim - 1)
    mean_mae = (ensemble_mean - truth).abs().mean(channel_reduction_dims)
    member_mae = (prediction - truth.unsqueeze(-1)).abs().mean(member_reduction_dims)
    persistence_mae = (persistence - truth).abs().mean(channel_reduction_dims)
    spread = prediction.std(dim=-1).mean(channel_reduction_dims)

    return {
        "ensemble_mean_mae": _channel_values(mean_mae),
        "expected_member_mae": _channel_values(member_mae),
        "persistence_mae": _channel_values(persistence_mae),
        "ensemble_mean_over_persistence": _channel_values(
            mean_mae / persistence_mae.clamp_min(1.0e-20)
        ),
        "mean_spread": _channel_values(spread),
        "ensemble_mean_mae_overall": float(mean_mae.mean().cpu()),
        "persistence_mae_overall": float(persistence_mae.mean().cpu()),
    }


def _mean_std(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean().cpu()),
        "std": float(values.std().cpu()),
    }


def compute_swe_structure_metrics(
    prediction: torch.Tensor,
    *,
    high_k_cutoff: float = 6.0,
) -> dict[str, Any]:
    """Measure spatial structure of ``u``/``v`` ensemble-member anomalies."""
    if prediction.ndim != 6:
        msg = "expected prediction shape (batch, time, x, y, channel, member)"
        raise ValueError(msg)
    if prediction.shape[-2] != 3:
        msg = "SWE diagnostics require exactly three channels ordered as h, u, v"
        raise ValueError(msg)
    if prediction.shape[-1] < 2:
        msg = "SWE structure diagnostics require at least two ensemble members"
        raise ValueError(msg)

    anomaly = prediction - prediction.mean(dim=-1, keepdim=True)
    velocity = anomaly[..., 1:3, :]
    fields = rearrange(velocity, "b t x y c m -> (b t c m) x y")
    n_x, n_y = fields.shape[-2:]

    k_x = torch.fft.fftfreq(n_x, device=prediction.device) * n_x
    k_y = torch.fft.fftfreq(n_y, device=prediction.device) * n_y
    k_radius = torch.sqrt(k_x[:, None].square() + k_y[None, :].square())
    spectrum = torch.fft.fft2(fields, norm="ortho").abs().square()
    total_energy = spectrum.sum((-2, -1))
    high_k_energy = spectrum[:, k_radius >= high_k_cutoff].sum(-1)
    high_k_fraction = high_k_energy / total_energy.clamp_min(1.0e-20)

    field_energy = fields.square().mean((-2, -1)).clamp_min(1.0e-20)
    neighbor_corr_x = (fields * fields.roll(1, dims=-2)).mean((-2, -1)) / field_energy
    neighbor_corr_y = (fields * fields.roll(1, dims=-1)).mean((-2, -1)) / field_energy

    u = rearrange(velocity[..., 0, :], "b t x y m -> (b t m) x y")
    v = rearrange(velocity[..., 1, :], "b t x y m -> (b t m) x y")
    du_dx = (u.roll(-1, -2) - u.roll(1, -2)) / 2
    du_dy = (u.roll(-1, -1) - u.roll(1, -1)) / 2
    dv_dx = (v.roll(-1, -2) - v.roll(1, -2)) / 2
    dv_dy = (v.roll(-1, -1) - v.roll(1, -1)) / 2
    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy
    div_vort_ratio = divergence.square().mean((-2, -1)) / vorticity.square().mean(
        (-2, -1)
    ).clamp_min(1.0e-20)

    return {
        "high_k_cutoff": high_k_cutoff,
        "high_k_fraction": _mean_std(high_k_fraction),
        "absolute_high_k_energy": _mean_std(high_k_energy / (n_x * n_y)),
        "neighbor_corr_x": _mean_std(neighbor_corr_x),
        "neighbor_corr_y": _mean_std(neighbor_corr_y),
        "divergence_vorticity_ratio": _mean_std(div_vort_ratio),
    }


def _persistence_prediction(
    model: Any,
    batch: Batch,
    *,
    n_steps: int,
    teacher_forcing_ratio: float,
) -> torch.Tensor:
    if teacher_forcing_ratio == 1.0:
        previous_states = torch.cat(
            (batch.input_fields[:, -1:], batch.output_fields[:, : n_steps - 1]),
            dim=1,
        )
        return model.denormalize_tensor(previous_states)
    if teacher_forcing_ratio == 0.0:
        initial = model.denormalize_tensor(batch.input_fields[:, -1:])
        return initial.expand(-1, n_steps, *initial.shape[2:])
    msg = (
        "SWE diagnostics support teacher_forcing_ratio values 0 or 1 so the "
        "persistence baseline is unambiguous."
    )
    raise ValueError(msg)


def run_swe_crps_diagnostics(
    cfg: DictConfig,
    *,
    work_dir: Path,
) -> dict[str, Any]:
    """Run one full-trajectory SWE ensemble diagnostic and write JSON results."""
    eval_cfg = cfg.get("eval", {})
    teacher_forcing_ratio = _resolve_teacher_forcing_ratio(eval_cfg)
    free_running_only = bool(eval_cfg.get("free_running_only", True))
    n_members = int(eval_cfg.get("n_members", cfg.model.get("n_members", 1)))
    if n_members < 2:
        msg = "SWE CRPS diagnostics require eval.n_members >= 2"
        raise ValueError(msg)

    checkpoint_path = resolve_checkpoint_path(
        eval_cfg,
        work_dir,
        missing_message="Provide eval.checkpoint=/path/to/checkpoint.ckpt",
    )
    datamodule, cfg, stats = setup_datamodule(cfg)
    if not isinstance(stats.get("example_batch"), Batch):
        msg = "SWE CRPS diagnostics require a raw Batch datamodule"
        raise TypeError(msg)

    with open_dict(cfg.model):
        cfg.model.n_members = n_members
    model = setup_epd_model(cfg, stats, datamodule=datamodule)
    checkpoint = load_checkpoint_payload(checkpoint_path)
    load_result = model.load_state_dict(
        extract_state_dict(checkpoint, use_ema=bool(eval_cfg.get("use_ema", False))),
        strict=True,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        msg = "Checkpoint parameters do not match the configured EPD model"
        raise RuntimeError(msg)

    device = resolve_device(str(eval_cfg.get("accelerator", "auto")))
    model.to(device).eval()
    datamodule.setup("test")
    batch = next(
        iter(
            datamodule.rollout_test_dataloader(
                batch_size=int(eval_cfg.get("batch_size", 1))
            )
        )
    ).to(device)

    max_steps = int(eval_cfg.get("max_rollout_steps", 100))
    n_steps = min(max_steps, int(batch.output_fields.shape[1]))
    torch.manual_seed(int(cfg.get("seed", 42)))
    with torch.no_grad():
        prediction, truth = model.rollout(
            batch,
            stride=1,
            max_rollout_steps=n_steps,
            teacher_forcing_ratio=teacher_forcing_ratio,
            free_running_only=free_running_only,
            n_members=n_members,
        )
    if truth is None:
        msg = "Rollout returned no ground truth"
        raise RuntimeError(msg)

    persistence = _persistence_prediction(
        model,
        batch,
        n_steps=truth.shape[1],
        teacher_forcing_ratio=teacher_forcing_ratio,
    )
    results: dict[str, Any] = {
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "mode": "teacher_forced" if teacher_forcing_ratio == 1.0 else "free_running",
        "teacher_forcing_ratio": teacher_forcing_ratio,
        "n_members": n_members,
        "n_steps": int(truth.shape[1]),
        "channels": ["h", "u", "v"],
        "fit": compute_fit_metrics(prediction, truth, persistence),
        "structure": compute_swe_structure_metrics(
            prediction,
            high_k_cutoff=float(cfg.get("swe_crps", {}).get("high_k_cutoff", 6.0)),
        ),
    }

    work_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, work_dir, filename="resolved_swe_crps_config.yaml")
    output_path = work_dir / "swe_crps_diagnostics.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote SWE CRPS diagnostics to %s", output_path)
    return results


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:
    """CLI entry point for shallow-water CRPS diagnostics."""
    logging.basicConfig(level=logging.INFO)
    results = run_swe_crps_diagnostics(cfg, work_dir=resolve_hydra_work_dir(None))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
