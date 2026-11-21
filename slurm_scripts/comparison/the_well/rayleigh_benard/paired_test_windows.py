"""Paired next-window evaluation on selected The Well trajectories."""

# Keep manifest preparation independent of the GPU runtime imports.
# ruff: noqa: PLC0415

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import cast

import yaml


def make_selection(
    trajectories, seed=0, *, frames=200, stride=1, windows_per_trajectory=8
):
    """Select distinct contexts per trajectory, preserving physical IDs."""
    rng = random.Random(seed)
    files = list(trajectories["files"].items())
    windows = []
    for trajectory in trajectories["indices"]:
        local = trajectory
        for file_info in files:
            filename, count = file_info
            if local < count:
                break
            local -= count
        else:
            raise ValueError(f"Unknown trajectory {trajectory}")
        for start in rng.sample(range(frames - 4 * stride), windows_per_trajectory):
            windows.append(
                {
                    "trajectory_idx": trajectory,
                    "file": filename,
                    "trajectory_in_file": local,
                    "context_frame": start,
                }
            )
    windows.sort(key=lambda row: (row["trajectory_idx"], row["context_frame"]))
    selection = {
        "seed": seed,
        "n_steps_input": 1,
        "n_steps_output": 4,
        "frames_per_trajectory": frames,
        "windows_per_trajectory": windows_per_trajectory,
        "files": dict(files),
        "windows": windows,
    }
    if stride != 1:
        selection["stride"] = stride
    return selection


def validate_selection(dataset, selection):
    """Validate the archived manifest with the maintained window selector."""
    from autocast.data.well_windows import validate_well_window_selection

    return validate_well_window_selection(dataset, selection)


def physical_prediction(model, batch):
    """Use the existing physical-output prediction API for both model families."""
    prediction = model.predict_step(batch, batch_idx=0)
    truth = model.denormalize_tensor(batch.output_fields)
    return prediction, truth


def run(  # noqa: PLR0912, PLR0915
    args,
    *,
    configure_encode_once=None,
    predict_window=physical_prediction,
    max_windows=None,
    reference_diagnostics=False,
    device="cuda",
):
    """Score the selected windows once with shared setup and metric functions."""
    import lightning as L
    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import DictConfig, OmegaConf
    from torch.utils.data import Subset

    from autocast.data.datamodule import TheWellDataModule
    from autocast.external.lola.field_transform import named_field_indices
    from autocast.metrics.coverage import Coverage
    from autocast.scripts.data import batch_to_device
    from autocast.scripts.eval.encoder_processor_decoder import (
        AVAILABLE_METRICS,
        AVAILABLE_METRICS_ENSEMBLE,
        _build_lola_raw_datamodule_config,
        _extract_processor_state_dict,
        _inject_lola_autoencoder_components,
        _is_processor_only_checkpoint,
        _load_lola_autoencoder_config_from_datamodule,
    )
    from autocast.scripts.execution import extract_state_dict, load_checkpoint_payload
    from autocast.scripts.setup import setup_datamodule, setup_epd_model

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "RUNNING").open("x"):
        pass
    selection = yaml.safe_load(args.manifest.read_text())
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    (out / "window_selection.yaml").write_bytes(args.manifest.read_bytes())
    cfg = OmegaConf.load(args.config)
    if not isinstance(cfg, DictConfig):
        msg = "Expected an evaluation configuration mapping"
        raise TypeError(msg)
    if getattr(args, "checkpoint", None) is not None:
        cfg.eval.checkpoint = str(args.checkpoint.resolve())
    if cfg.eval.mode == "encode_once" and configure_encode_once is not None:
        cfg = configure_encode_once(cfg)
    elif cfg.eval.mode == "encode_once":
        loaded = _load_lola_autoencoder_config_from_datamodule(cfg)
        if loaded is None:
            msg = "Cannot resolve the frozen LOLA autoencoder"
            raise ValueError(msg)
        ae_cfg, ae_run = loaded
        raw_cfg = _build_lola_raw_datamodule_config(ae_cfg, ae_run, cfg.datamodule)
        cfg = _inject_lola_autoencoder_components(cfg, ae_cfg, ae_run)
        cfg.datamodule = raw_cfg
    elif cfg.eval.mode != "ambient":
        msg = "Expected a raw-field ambient or encode_once evaluation"
        raise ValueError(msg)
    cfg.datamodule.batch_size = 1
    cfg.datamodule.num_workers = cfg.eval.get("num_workers", 2)
    cfg.datamodule.cache_small = False
    cfg.eval.n_members = args.members
    cfg.model.n_members = args.members
    cfg.eval.compute_test_metrics = True
    cfg.eval.compute_rollout_metrics = False
    cfg.eval.compute_rollout_coverage = False
    cfg.eval.batch_indices = []
    cfg.eval.benchmark.enabled = False
    cfg.eval.benchmark_rollout.enabled = False
    if cfg.eval.get("use_ema", False):
        msg = "This comparison uses the selected raw checkpoint weights"
        raise ValueError(msg)
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    datamodule, cfg, stats = setup_datamodule(cfg)
    if not isinstance(datamodule, TheWellDataModule):
        msg = "Paired window evaluation requires TheWellDataModule"
        raise TypeError(msg)
    indices = validate_selection(datamodule.test_dataset, selection)
    windows = selection["windows"]
    if max_windows is not None:
        indices, windows = indices[:max_windows], windows[:max_windows]
    fields = selection.get("fields")
    field_order = None
    if fields is not None:
        field_order = named_field_indices(
            datamodule.test_dataset.well_metadata.field_names.values(), fields
        )
        cfg.model.encoder.field_indices = field_order
    n_channels = stats["channel_count"]
    # The test loader accepts a map-style Subset at runtime.
    datamodule.test_dataset = Subset(datamodule.test_dataset, indices)  # pyright: ignore[reportAttributeAccessIssue]
    OmegaConf.save(cfg, out / "resolved_test_config.yaml")
    checkpoint = Path(cfg.eval.checkpoint)
    payload = load_checkpoint_payload(checkpoint)
    if (
        args.expected_step is not None
        and payload.get("global_step") != args.expected_step
    ):
        msg = "Checkpoint global_step differs from the selected endpoint"
        raise ValueError(msg)
    model = setup_epd_model(cfg, stats, datamodule=datamodule)
    if _is_processor_only_checkpoint(payload):
        model.processor.load_state_dict(
            _extract_processor_state_dict(payload, use_ema=False), strict=True
        )
    else:
        model.load_state_dict(extract_state_dict(payload, use_ema=False), strict=True)
    checkpoint_info = {
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
    }
    del payload
    if bool(model.norm is not None) != bool(cfg.datamodule.use_normalization):
        msg = "Model/data normalization disagree"
        raise ValueError(msg)
    device = torch.device(device)
    model = model.eval().to(device)
    names = [
        "mse",
        "rmse",
        "vrmse_v2",
        "psrmse_low",
        "psrmse_mid",
        "psrmse_high",
        "crps",
        "afcrps",
        "spread",
        "skill",
        "ssr",
    ]
    metrics = {}
    levels = []
    reference_score = None
    if reference_diagnostics:
        from slurm_scripts.comparison.the_well.rayleigh_benard import (
            complete_diagnostics,
        )

        reference_score = complete_diagnostics.score_leads
        levels = complete_diagnostics.LEVELS
        metric_names = [
            *complete_diagnostics.METRICS,
            *(f"coverage_{level:g}" for level in levels),
        ]
    else:
        metrics = {
            name: (AVAILABLE_METRICS | AVAILABLE_METRICS_ENSEMBLE)[name]().to(device)
            for name in names
        }
        metrics["coverage_0.9"] = Coverage(coverage_level=0.9).to(device)
        metric_names = list(metrics)
    rows = []
    started = time.monotonic()
    with (
        torch.inference_mode(),
        (out / "per_window_lead_channel.csv").open("x") as stream,
    ):
        for position, raw_batch in enumerate(datamodule.test_dataloader()):
            batch = batch_to_device(raw_batch, device)
            pred, truth = predict_window(model, batch)
            if field_order is not None:
                truth = truth[..., field_order]
            if (
                pred.shape[:-1] != truth.shape
                or pred.shape[1] != 4
                or pred.shape[-1] != args.members
            ):
                raise ValueError(
                    f"Unexpected forecast shapes: {pred.shape}, {truth.shape}"
                )
            if reference_score is not None:
                # Each score is per channel. Split the native Euler grid so
                # Coverage's per-lead torch.quantile stays below 2**24 inputs.
                pieces = [
                    reference_score(p, t)
                    for p, t in zip(
                        pred.split(2, dim=-2), truth.split(2, dim=-1), strict=True
                    )
                ]
                scores = {
                    name: np.concatenate([part[name] for part in pieces], axis=-1)
                    for name in pieces[0]
                }
            else:
                scores = {
                    name: metric.score(pred, truth).cpu().numpy()[0]
                    for name, metric in metrics.items()
                }
            if not all(
                v.shape == (4, n_channels) and np.isfinite(v).all()
                for v in scores.values()
            ):
                raise ValueError(f"Invalid metrics at selected window {position}")
            block = []
            for lead in range(4):
                for channel in range(n_channels):
                    block.append(
                        {
                            "selection_position": position,
                            **windows[position],
                            "dataset_index": indices[position],
                            "lead": lead + 1,
                            "channel": channel,
                            **(
                                {
                                    "field": fields[channel],
                                    "lead_raw_frames": (lead + 1) * selection["stride"],
                                }
                                if fields is not None
                                else {}
                            ),
                            **{
                                name: float(v[lead, channel])
                                for name, v in scores.items()
                            },
                        }
                    )
            pd.DataFrame(block).to_csv(stream, index=False, header=position == 0)
            stream.flush()
            rows.extend(block)
            if position == 0 or (position + 1) % 32 == 0:
                print(
                    json.dumps(
                        {
                            "completed_windows": position + 1,
                            "total_windows": len(indices),
                            "elapsed_s": time.monotonic() - started,
                            "peak_gpu_allocated_gib": (
                                torch.cuda.max_memory_allocated() / 1024**3
                                if device.type == "cuda"
                                else None
                            ),
                        }
                    ),
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    if len(frame) != len(indices) * 4 * n_channels:
        msg = "Evaluation did not consume every selected window"
        raise ValueError(msg)
    frame.groupby("selection_position")[metric_names].mean().to_csv(
        out / "per_window_metrics.csv"
    )
    frame.groupby(["lead", "channel"])[metric_names].mean().to_csv(
        out / "per_lead_channel_metrics.csv"
    )
    summary = cast("pd.Series", frame[metric_names].mean())
    if levels:
        coverage = np.asarray(summary[[f"coverage_{q:g}" for q in levels]])
        summary["coverage_mae"] = float(np.abs(coverage - levels).mean())
        if frame["trajectory_idx"].nunique() > 1:
            from slurm_scripts.comparison.the_well.rayleigh_benard import (
                plot_diagnostics,
            )

            uncertainty = plot_diagnostics.trajectory_stats(frame, [])
            uncertainty["coverage_mae_mean"], uncertainty["coverage_mae_se"] = (
                plot_diagnostics.calibration_mae_stats(frame)
            )
            uncertainty.to_csv(out / "trajectory_summary.csv", index=False)
    summary.rename("mean").to_csv(out / "summary.csv")
    pd.DataFrame([{"window": "all", "batch_idx": "all", **summary.to_dict()}]).to_csv(
        out / "evaluation_metrics.csv", index=False
    )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "source_config": str(args.config.resolve()),
                "checkpoint": str(checkpoint),
                **checkpoint_info,
                "window_manifest_sha256": manifest_hash,
                "n_windows": len(indices),
                "n_members": args.members,
                "targets": "raw physical fields",
                "normalizer_inverted": model.norm is not None,
                "sampling_seed": 42,
                "inference_seconds": time.monotonic() - started,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    (out / "COMPLETE").write_text(
        f"{len(indices)} windows, four leads, {n_channels} channels, "
        "physical-unit scoring\n"
    )


def main():
    """Prepare a shared selection or run one checkpoint on that selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--trajectories", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--members", type=int, default=16)
    evaluate.add_argument("--expected-step", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        selection = make_selection(yaml.safe_load(args.trajectories.read_text()))
        with args.manifest.open("x") as stream:
            yaml.safe_dump(selection, stream, sort_keys=False)
    else:
        run(args)


if __name__ == "__main__":
    main()
