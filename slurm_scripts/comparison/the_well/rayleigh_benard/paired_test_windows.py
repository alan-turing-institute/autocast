"""RB-only paired next-window evaluation; shared library defaults are unchanged."""

# Keep manifest preparation independent of the GPU runtime imports.
# ruff: noqa: PLC0415

import argparse
import hashlib
import json
import random
import time
from bisect import bisect_right
from pathlib import Path
from typing import cast

import yaml


def make_selection(trajectories, seed=0):
    """Select eight distinct contexts per trajectory, preserving physical IDs."""
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
        # These RB files contain 200 frames; 1 input + 4 outputs gives 196 starts.
        for start in rng.sample(range(196), 8):
            windows.append(
                {
                    "trajectory_idx": trajectory,
                    "file": filename,
                    "trajectory_in_file": local,
                    "context_frame": start,
                }
            )
    windows.sort(key=lambda row: (row["trajectory_idx"], row["context_frame"]))
    return {
        "seed": seed,
        "n_steps_input": 1,
        "n_steps_output": 4,
        "frames_per_trajectory": 200,
        "windows_per_trajectory": 8,
        "files": dict(files),
        "windows": windows,
    }


def validate_selection(dataset, selection):
    """Reject changed file ordering, frame counts, strides, or window identities."""
    from autocast.data.transforms import LogScalars

    well = dataset.well_dataset
    expected_files = list(selection["files"])
    actual_files = [Path(path).name for path in well.files_paths]
    if actual_files != expected_files:
        msg = "Test file inventory/order differs from the saved selection"
        raise ValueError(msg)
    if list(well.n_trajectories_per_file) != list(selection["files"].values()):
        msg = "Test trajectory counts differ from the saved selection"
        raise ValueError(msg)
    if (
        well.n_steps_input,
        well.n_steps_output,
        well.min_dt_stride,
        well.max_dt_stride,
    ) != (1, 4, 1, 1):
        msg = "Paired evaluation requires 1-to-4 windows with fixed stride 1"
        raise ValueError(msg)
    if any(n != 200 for n in well.n_steps_per_trajectory) or any(
        n != 196 for n in well.n_windows_per_trajectory
    ):
        msg = "RB frame/window counts differ from the saved selection"
        raise ValueError(msg)
    if well.transform is not None and not isinstance(well.transform, LogScalars):
        msg = "Paired test windows must not use random augmentation"
        raise ValueError(msg)
    identities = [
        (row["file"], row["trajectory_in_file"], row["context_frame"])
        for row in selection["windows"]
    ]
    if len(identities) != 512 or len(set(identities)) != 512:
        msg = "Expected exactly 512 distinct windows"
        raise ValueError(msg)
    counts = {}
    indices = []
    for row in selection["windows"]:
        file_idx = expected_files.index(row["file"])
        trajectory = (
            sum(well.n_trajectories_per_file[:file_idx]) + row["trajectory_in_file"]
        )
        expected = (
            max(int(well.file_index_offsets[file_idx]), 0)
            + row["trajectory_in_file"] * 196
            + row["context_frame"]
        )
        if not (
            0 <= row["trajectory_in_file"] < well.n_trajectories_per_file[file_idx]
            and 0 <= row["context_frame"] < 196
            and bisect_right(well.file_index_offsets, expected) - 1 == file_idx
            and row["trajectory_idx"] == trajectory
        ):
            raise ValueError(f"Window identity mismatch: {row}")
        indices.append(expected)
        counts[trajectory] = counts.get(trajectory, 0) + 1
    if len(counts) != 64 or set(counts.values()) != {8}:
        msg = "Expected eight windows from each of 64 trajectories"
        raise ValueError(msg)
    return indices


def physical_prediction(model, batch):
    """Use the existing physical-output prediction API for both model families."""
    prediction = model.predict_step(batch, batch_idx=0)
    truth = model.denormalize_tensor(batch.output_fields)
    return prediction, truth


def run(args):  # noqa: PLR0912, PLR0915
    """Score the selected windows once with shared setup and metric functions."""
    import lightning as L
    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import DictConfig, OmegaConf
    from torch.utils.data import Subset

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
    if cfg.eval.mode == "encode_once":
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
    cfg.datamodule.num_workers = 2
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
    indices = validate_selection(datamodule.test_dataset, selection)
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
    device = torch.device("cuda")
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
    metrics = {
        name: (AVAILABLE_METRICS | AVAILABLE_METRICS_ENSEMBLE)[name]().to(device)
        for name in names
    }
    metrics["coverage_0.9"] = Coverage(coverage_level=0.9).to(device)
    rows = []
    started = time.monotonic()
    with (
        torch.inference_mode(),
        (out / "per_window_lead_channel.csv").open("x") as stream,
    ):
        for position, raw_batch in enumerate(datamodule.test_dataloader()):
            batch = batch_to_device(raw_batch, device)
            pred, truth = physical_prediction(model, batch)
            if (
                pred.shape[:-1] != truth.shape
                or pred.shape[1] != 4
                or pred.shape[-1] != args.members
            ):
                raise ValueError(
                    f"Unexpected forecast shapes: {pred.shape}, {truth.shape}"
                )
            scores = {
                name: metric.score(pred, truth).cpu().numpy()[0]
                for name, metric in metrics.items()
            }
            if not all(
                v.shape == (4, 4) and np.isfinite(v).all() for v in scores.values()
            ):
                raise ValueError(f"Invalid metrics at selected window {position}")
            block = []
            for lead in range(4):
                for channel in range(4):
                    block.append(
                        {
                            "selection_position": position,
                            **selection["windows"][position],
                            "dataset_index": indices[position],
                            "lead": lead + 1,
                            "channel": channel,
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
                            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated()
                            / 1024**3,
                        }
                    ),
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 512 * 4 * 4:
        msg = "Evaluation did not consume exactly 512 complete windows"
        raise ValueError(msg)
    frame.groupby("selection_position")[list(metrics)].mean().to_csv(
        out / "per_window_metrics.csv"
    )
    frame.groupby(["lead", "channel"])[list(metrics)].mean().to_csv(
        out / "per_lead_channel_metrics.csv"
    )
    summary = cast("pd.Series", frame[list(metrics)].mean())
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
                "n_windows": 512,
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
        "512 windows, four leads, four channels; physical-unit scoring\n"
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
