"""Prepare and run frozen-checkpoint RB diagnostics without training.

Invoke as a module from the repository root. Preparation writes scripts but
never submits them. Raw-field configs from paired_test_windows.py preserve
each model's normalization and embed the frozen autoencoder configuration.
"""

from __future__ import annotations

# Keep preparation independent of runtime model imports.
# ruff: noqa: PLC0415
import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import yaml

HERE = Path(__file__).resolve().parent
MODULE = "slurm_scripts.comparison.the_well.rayleigh_benard.complete_diagnostics"
METRICS = [
    "vrmse_v2",
    "crps",
    "ssr",
    "spread",
    "skill",
    "psrmse_low",
    "psrmse_mid",
    "psrmse_high",
    "energy",
]
LEVELS = [i / 20 for i in range(1, 20)]


def sha256(path: Path) -> str:
    """Hash large checkpoint files without loading them into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_sha256(path: Path, expected: str | None, label: str) -> str:
    """Require the archived identity, rather than accepting the current file."""
    if not expected:
        raise ValueError(f"Missing archived {label} hash")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Archived {label} hash changed: {path}")
    return actual


def save_json(path: Path, value: object) -> None:
    """Save provenance with intentional newlines."""
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_record(root: Path) -> dict:
    """Record source identity and refuse uncommitted evaluation code."""

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        msg = "Commit tracked source changes before preparing diagnostics"
        raise ValueError(msg)
    git("ls-files", "--error-unmatch", str(Path(__file__).relative_to(root)))
    return {"source_commit": git("rev-parse", "HEAD"), "source_root": str(root)}


def validate_config(cfg, entry: dict) -> None:
    """Reject changed prediction tasks, normalized targets, EMA or samplers."""
    if cfg.eval.mode != entry["mode"] or cfg.eval.get("use_ema", False):
        msg = "Expected the archived representation and raw weights"
        raise ValueError(msg)
    data = cfg.datamodule
    if (
        data.n_steps_input,
        data.n_steps_output,
        data.min_dt_stride,
        data.max_dt_stride,
    ) != (1, 4, 1, 1):
        msg = "Expected the archived 1-to-4 stride-one task"
        raise ValueError(msg)
    if data.well_dataset_name != "rayleigh_benard":
        msg = "Expected raw Rayleigh-Benard fields"
        raise ValueError(msg)
    if entry["mode"] == "encode_once":
        if data.use_normalization or "encoder" not in cfg.model:
            msg = "Use the archived, injected raw-field latent config"
            raise ValueError(msg)
        if "flow" in cfg.model.processor._target_.lower() and (
            cfg.model.processor.integrator != "ab3"
            or cfg.model.processor.flow_ode_steps != 16
        ):
            msg = "The archived FM sampler is AB3 with 16 ODE steps"
            raise ValueError(msg)


def prepare(args) -> None:  # noqa: PLR0915
    """Pin configs, checkpoints, normalization, codec and selections on disk."""
    from omegaconf import OmegaConf, open_dict

    from slurm_scripts.comparison.the_well.rayleigh_benard.paired_test_windows import (
        make_selection,
    )

    root = HERE.parents[3]
    record = source_record(root)
    recipe = yaml.safe_load(args.recipe.read_text())
    cohort = yaml.safe_load((HERE / "test64_seed0.yaml").read_text())
    paired = make_selection(cohort)
    # Refuse to overwrite a prepared campaign or any old evaluation output.
    args.output.mkdir(parents=True, exist_ok=False)
    assets = args.output / "assets"
    assets.mkdir()
    (args.output / "cohort.yaml").write_bytes((HERE / "test64_seed0.yaml").read_bytes())
    (args.output / "windows.yaml").write_text(yaml.safe_dump(paired, sort_keys=False))
    (args.output / "recipe.yaml").write_bytes(args.recipe.read_bytes())
    models = {}
    for name, entry in recipe["models"].items():
        if not name.replace("_", "").isalnum():
            msg = "Model names must be alphanumeric with underscores"
            raise ValueError(msg)
        source = args.outputs_root / entry["config"]
        cfg = OmegaConf.load(source)
        validate_config(cfg, entry)
        # Compare physical identities, not only the count of sampled windows.
        archived = yaml.safe_load((source.parent / "window_selection.yaml").read_text())
        if archived != paired:
            raise ValueError(f"Archived paired-window selection changed for {name}")
        checkpoint = Path(cfg.eval.checkpoint).resolve()
        checkpoint_hash = verify_sha256(
            checkpoint, entry.get("checkpoint_sha256"), f"{name} checkpoint"
        )
        dependencies = {str(checkpoint): checkpoint_hash}
        with open_dict(cfg):
            cfg.datamodule.batch_size = 1
            cfg.datamodule.num_workers = args.workers
            cfg.datamodule.cache_small = False
            cfg.datamodule.max_rollout_steps = 200
            if args.raw_data_root:
                cfg.datamodule.well_base_path = str(args.raw_data_root)
                cfg.datamodule.normalization_path = str(
                    args.raw_data_root / "rayleigh_benard/stats.yaml"
                )
            cfg.eval.n_members = cfg.model.n_members = 16
            cfg.eval.use_ema = False
            cfg.logging.wandb.enabled = False
            if entry["mode"] == "encode_once":
                for part in ["encoder", "decoder"]:
                    if args.autoencoder_root:
                        cfg.model[part].runpath = str(args.autoencoder_root)
                    weights = Path(cfg.model[part].runpath) / "state.pth"
                    dependencies[str(weights)] = verify_sha256(
                        weights,
                        entry.get("codec_sha256", {}).get(part),
                        f"{name} {part} codec",
                    )
            # Archive normalization rather than relying on a mutable stats file.
            if cfg.datamodule.use_normalization:
                stats = Path(cfg.datamodule.normalization_path)
                archived_stats = assets / f"{name}-stats.yaml"
                archived_stats.write_bytes(stats.read_bytes())
                cfg.datamodule.normalization_path = str(archived_stats)
                dependencies[str(archived_stats)] = verify_sha256(
                    archived_stats,
                    entry.get("normalization_sha256"),
                    f"{name} normalization",
                )
        config = assets / f"{name}.yaml"
        OmegaConf.save(cfg, config)
        models[name] = {
            **entry,
            "config": str(config),
            "config_sha256": sha256(config),
            "checkpoint": str(checkpoint),
            "dependencies": dependencies,
            "source_config": str(source),
            "source_config_sha256": sha256(source),
        }
    state = {
        **record,
        "models": models,
        "horizon": args.horizon,
        "members": 16,
        "rollout_start": 16,
        "rollout_seed": 0,
        "paired_seed": 42,
        "lockfile_sha256": sha256(root / "uv.lock"),
        "energy_definition": (
            "alpha=1; Euclidean spatial vector per lead/channel; V-statistic"
        ),
        "cohort_sha256": sha256(args.output / "cohort.yaml"),
        "windows_sha256": sha256(args.output / "windows.yaml"),
        "benchmark": {"n_warmup": 5, "n_benchmark": 50, "batch_size": 1},
    }
    manifest = args.output / "manifest.json"
    save_json(manifest, state)
    for name in models:
        for phase in ["pilot", "production"]:
            command = [
                "uv",
                "run",
                "--frozen",
                "--no-sync",
                "python",
                "-m",
                MODULE,
                "run",
                "--manifest",
                str(manifest),
                "--model",
                name,
            ]
            if phase == "pilot":
                command.append("--pilot")
            script = args.output / f"{name}-{phase}.sbatch"
            script.write_text(
                "\n".join(
                    [
                        "#!/bin/bash",
                        f"#SBATCH --job-name=rb-diag-{name}-{phase}",
                        f"#SBATCH --account={args.account}",
                        "#SBATCH --partition=workq",
                        "#SBATCH --nodes=1",
                        "#SBATCH --gpus=1",
                        "#SBATCH --ntasks=1",
                        "#SBATCH --cpus-per-task=8",
                        "#SBATCH --mem=115G",
                        "#SBATCH --time="
                        + ("00:30:00" if phase == "pilot" else "04:00:00"),
                        "#SBATCH --no-requeue",
                        f"#SBATCH --output={args.output}/{name}-{phase}-%j.out",
                        "set -euo pipefail",
                        "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CUDA_VISIBLE_DEVICES",
                        "export OMP_NUM_THREADS=8 MPLBACKEND=Agg",
                        "export RUNTIME_TYPECHECKING=false",
                        "export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
                        'export PATH="$HOME/.local/bin:$PATH"',
                        f"cd {shlex.quote(str(root))}",
                        # The installed editable package may belong to another checkout.
                        'export PYTHONPATH="$PWD/src:$PWD"',
                        "exec srun --input=none " + shlex.join(command),
                        "",
                    ]
                )
            )
    print(json.dumps({"manifest": str(manifest), "submitted": False}, indent=2))


def score_leads(prediction, truth, *, chunk_size: int = 1) -> dict:
    """Use existing scores with bounded time chunks, including spatial energy."""
    import torch

    from autocast.metrics.coverage import Coverage
    from autocast.metrics.ensemble import EnergyScore
    from autocast.scripts.eval.encoder_processor_decoder import (
        AVAILABLE_METRICS,
        AVAILABLE_METRICS_ENSEMBLE,
    )

    if prediction.shape[:-1] != truth.shape or truth.ndim < 4:
        msg = "Expected B,T,...,C truth and B,T,...,C,M predictions"
        raise ValueError(msg)
    if chunk_size < 1 or truth.shape[0] != 1:
        msg = "Score one trajectory at a time with positive time chunks"
        raise ValueError(msg)
    metrics = {
        name: (AVAILABLE_METRICS | AVAILABLE_METRICS_ENSEMBLE)[name]()
        for name in METRICS
        if name != "energy"
    }
    # A one-time-slice spatial norm equals the paper's per-lead energy score.
    # Do not average chunk-level joint temporal energy scores across time.
    metrics["energy"] = EnergyScore(vector_dims="spatial")
    metrics.update({f"coverage_{q:g}": Coverage(coverage_level=q) for q in LEVELS})
    pieces = {name: [] for name in metrics}
    for start in range(0, truth.shape[1], chunk_size):
        stop = start + chunk_size
        for name, metric in metrics.items():
            value = metric.to(truth.device).score(
                prediction[:, start:stop], truth[:, start:stop]
            )
            if value.shape != (
                1,
                min(chunk_size, truth.shape[1] - start),
                truth.shape[-1],
            ):
                raise ValueError(f"Unexpected {name} shape: {value.shape}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {name} scores")
            pieces[name].append(value[0].detach().cpu())
    return {name: torch.cat(values).numpy() for name, values in pieces.items()}


def write_scores(stream, *, prediction, truth, identity: dict, first: bool) -> None:
    """Write every trajectory/lead/channel score from this inference pass."""
    scores = score_leads(prediction, truth)
    writer = csv.DictWriter(stream, fieldnames=[*identity, "lead", "channel", *scores])
    if first:
        writer.writeheader()
    for lead in range(truth.shape[1]):
        for channel in range(truth.shape[-1]):
            writer.writerow(
                {
                    **identity,
                    "lead": lead + 1,
                    "channel": channel,
                    **{
                        key: float(value[lead, channel])
                        for key, value in scores.items()
                    },
                }
            )
    stream.flush()


def validate_prediction(prediction, truth, *, frames: int) -> None:
    """Require native resolution, all four fields and every ensemble member."""
    expected = tuple(truth.shape)
    if (
        expected[:2] != (1, frames)
        or expected[-1] != 4
        or expected[2:-1] not in [(512, 128), (128, 512), (512 * 128,)]
        or tuple(prediction.shape) != (*expected, 16)
    ):
        msg = (
            "Expected native 512x128 four-field forecasts with sixteen members; "
            f"received prediction {tuple(prediction.shape)} and truth {expected}"
        )
        raise ValueError(msg)


def load_model(cfg, entry: dict):
    """Restore strict raw checkpoint weights through the maintained setup API."""
    import torch

    from autocast.data.datamodule import TheWellDataModule
    from autocast.scripts.eval.encoder_processor_decoder import (
        _extract_processor_state_dict,
        _is_processor_only_checkpoint,
    )
    from autocast.scripts.execution import extract_state_dict, load_checkpoint_payload
    from autocast.scripts.setup import setup_datamodule, setup_epd_model

    datamodule, cfg, stats = setup_datamodule(cfg)
    # Frozen paired configs use the raw Well loader for both representations.
    datamodule = cast(TheWellDataModule, datamodule)
    model = setup_epd_model(cfg, stats, datamodule=datamodule)
    payload = load_checkpoint_payload(Path(entry["checkpoint"]))
    if payload.get("global_step") != entry["expected_step"]:
        msg = "Checkpoint optimizer step changed"
        raise ValueError(msg)
    if _is_processor_only_checkpoint(payload):
        model.processor.load_state_dict(
            _extract_processor_state_dict(payload, use_ema=False), strict=True
        )
    else:
        model.load_state_dict(extract_state_dict(payload, use_ema=False), strict=True)
    if bool(model.norm is not None) != bool(cfg.datamodule.use_normalization):
        msg = "Model and data normalization disagree"
        raise ValueError(msg)
    del payload
    device = torch.device("cuda")
    return model.eval().to(device), datamodule, cfg, device


def run(args) -> None:  # noqa: PLR0912, PLR0915
    """Collect paired scores, seeded trajectory rollouts and end-to-end latency."""
    import lightning as L
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset

    from autocast.benchmarking.inference import _time_calls
    from autocast.scripts.data import batch_to_device
    from autocast.scripts.eval.encoder_processor_decoder import (
        _apply_rollout_subset,
        _build_encode_once_rollout_predict,
        _crop_rollout_batch_start,
    )
    from slurm_scripts.comparison.the_well.rayleigh_benard.paired_test_windows import (
        physical_prediction,
        validate_selection,
    )

    manifest = args.manifest.resolve()
    state = json.loads(manifest.read_text())
    source = source_record(Path(state["source_root"]))
    if source["source_commit"] != state["source_commit"]:
        msg = "Run from the pinned source commit"
        raise ValueError(msg)
    entry = state["models"][args.model]
    if sha256(Path(entry["config"])) != entry["config_sha256"]:
        msg = "Prepared model config changed"
        raise ValueError(msg)
    for path, expected in entry["dependencies"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Pinned dependency changed: {path}")
    for name, key in [
        ("cohort.yaml", "cohort_sha256"),
        ("windows.yaml", "windows_sha256"),
    ]:
        if sha256(manifest.parent / name) != state[key]:
            msg = "Prepared selection changed"
            raise ValueError(msg)
    if not args.pilot:
        pilot_dir = manifest.parent / args.model / "pilot"
        if not (pilot_dir / "COMPLETE").exists():
            msg = "A successful pilot is required before production"
            raise ValueError(msg)
        pilot = json.loads((pilot_dir / "provenance.json").read_text())
        if not pilot["pilot"]:
            msg = "Expected explicit pilot provenance"
            raise ValueError(msg)
        if pilot["manifest_sha256"] != sha256(manifest):
            msg = "Pilot used a different manifest"
            raise ValueError(msg)
    output = manifest.parent / args.model / ("pilot" if args.pilot else "production")
    output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(entry["config"])
    validate_config(cfg, entry)
    L.seed_everything(state["paired_seed"], workers=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    model, datamodule, cfg, device = load_model(cfg, entry)
    selection = yaml.safe_load((manifest.parent / "windows.yaml").read_text())
    indices = validate_selection(datamodule.test_dataset, selection)
    positions = list(range(8 if args.pilot else len(indices)))
    paired_loader = DataLoader(
        Subset(datamodule.test_dataset, [indices[i] for i in positions]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=datamodule.test_dataloader().collate_fn,
    )
    first_batch = None
    with torch.inference_mode(), (output / "paired.csv").open("x") as stream:
        for count, (position, raw) in enumerate(
            zip(positions, paired_loader, strict=True)
        ):
            batch = batch_to_device(raw, device)
            if first_batch is None:
                first_batch = batch
            pred, truth = physical_prediction(model, batch)
            validate_prediction(pred, truth, frames=4)
            write_scores(
                stream,
                prediction=pred,
                truth=truth,
                identity={
                    "selection_position": position,
                    **selection["windows"][position],
                },
                first=count == 0,
            )
            del pred, truth, batch
            print(f"paired {count + 1}/{len(positions)}", flush=True)
    # Use the native checked subset policy; explicit IDs travel with every row.
    cfg.eval.rollout_subset = str(manifest.parent / "cohort.yaml")
    cfg.eval.trajectory_statistics.enabled = False
    cohort = _apply_rollout_subset(datamodule, cfg.eval)
    if cohort is None:
        msg = "The frozen rollout selection must be applied"
        raise ValueError(msg)
    trajectory_ids = cohort["indices"][:2] if args.pilot else cohort["indices"]
    loader = datamodule.rollout_test_dataloader(batch_size=1)
    horizon = state["horizon"]
    steps = math.ceil(horizon / 4)
    predict_rollout: Callable[[Any], tuple[Any, Any]]
    if entry["mode"] == "encode_once":
        predict_rollout = _build_encode_once_rollout_predict(
            model,
            rollout_stride=4,
            max_rollout_steps=steps,
            free_running_only=True,
            n_members=16,
            device=device,
            rollout_start=16,
        )
    else:

        def predict_ambient(batch):
            return model.rollout(
                _crop_rollout_batch_start(batch, 16),
                stride=4,
                max_rollout_steps=steps,
                free_running_only=True,
                n_members=16,
            )

        predict_rollout = predict_ambient

    L.seed_everything(state["rollout_seed"], workers=True)
    completed_rollouts = 0
    with torch.inference_mode(), (output / "rollout.csv").open("x") as stream:
        for position, raw in enumerate(loader):
            if position == len(trajectory_ids):
                break
            batch = batch_to_device(raw, device)
            pred, truth = predict_rollout(batch)
            if truth is None or min(pred.shape[1], truth.shape[1]) < horizon:
                msg = "The selected rollout cannot supply the common horizon"
                raise ValueError(msg)
            validate_prediction(pred[:, :horizon], truth[:, :horizon], frames=horizon)
            write_scores(
                stream,
                prediction=pred[:, :horizon],
                truth=truth[:, :horizon],
                identity={
                    "selection_position": position,
                    "trajectory_idx": trajectory_ids[position],
                },
                first=position == 0,
            )
            del pred, truth, batch
            completed_rollouts += 1
            print(f"rollout {position + 1}/{len(trajectory_ids)}", flush=True)
    # Timings exclude loader and metric costs, and include the physical-output
    # predict_step API (normalization, encoding, sampling and decoding).
    if completed_rollouts != len(trajectory_ids) or first_batch is None:
        msg = "Inference did not consume the full selected cohort"
        raise ValueError(msg)
    benchmark_batch = first_batch

    def timed_prediction() -> None:
        model.predict_step(benchmark_batch, 0)

    warmup, repeats = (1, 2) if args.pilot else (5, 50)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        seconds = _time_calls(timed_prediction, device, warmup, repeats)
    save_json(
        output / "latency.json",
        {
            "seconds_per_16_member_forecast": seconds,
            "batch_size": 1,
            "members": 16,
            "warmup": warmup,
            "repeats": repeats,
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "definition": "physical predict_step, no loader/metrics; includes codec",
        },
    )
    save_json(
        output / "provenance.json",
        {
            "model": args.model,
            "label": entry["label"],
            "pilot": args.pilot,
            "manifest_sha256": sha256(manifest),
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["dependencies"][entry["checkpoint"]],
            "global_step": entry["expected_step"],
            "source_commit": state["source_commit"],
            "horizon": horizon,
            "members": 16,
            "rollout_start": 16,
            "rollout_seed": state["rollout_seed"],
            "paired_seed": state["paired_seed"],
            "paired_windows": len(positions),
            "rollout_trajectories": trajectory_ids,
            "cohort_sha256": state["cohort_sha256"],
            "windows_sha256": state["windows_sha256"],
            "energy_definition": state["energy_definition"],
            "versions": {"torch": torch.__version__, "lightning": L.__version__},
            "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            "files": {
                name: sha256(output / name)
                for name in ["paired.csv", "rollout.csv", "latency.json"]
            },
        },
    )
    (output / "COMPLETE").write_text("Frozen-checkpoint diagnostics complete.\n")


def main() -> None:
    """Expose preparation and explicit evaluation-only execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--recipe", type=Path, default=HERE / "diagnostics_models.yaml")
    prep.add_argument("--outputs-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--raw-data-root", type=Path)
    prep.add_argument("--autoencoder-root", type=Path)
    prep.add_argument("--account", required=True)
    prep.add_argument("--workers", type=int, default=2)
    prep.add_argument("--horizon", type=int, default=180, choices=[100, 180])
    execute = sub.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--model", required=True)
    execute.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        for name in ["outputs_root", "output", "raw_data_root", "autoencoder_root"]:
            if getattr(args, name) is not None:
                setattr(args, name, getattr(args, name).expanduser().resolve())
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
