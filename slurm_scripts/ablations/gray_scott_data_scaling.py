"""Prepare Gray-Scott jobs and explicit continuations; never submit jobs.

The prepare command writes resolved configs and batch scripts for inspection.
Only the execute command, invoked inside an allocation, runs a stage.
"""

from __future__ import annotations

import argparse

# ruff: noqa: PLC0415 -- Workload imports are deferred during preparation.
import copy
import hashlib
import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "ablations/training_data/gray_scott_5x/"
RUNS = {
    "ae": ("autoencoder", "ae_dc_large", 512),
    "crps24": ("encoder_processor_decoder", "crps_vit_azula_large", 399),
    "crps48": ("encoder_processor_decoder", "crps_vit_azula_large_2x", 798),
    "fm24": ("processor", "fm_vit_large", 2830),
    "fm48": ("processor", "fm_vit_large_2x", 5660),
}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _source_commit() -> str:
    if subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip():
        msg = "Commit the checkout before preparing a campaign."
        raise RuntimeError(msg)
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _check_source(state: dict) -> None:
    if _source_commit() != state["source_commit"]:
        msg = "The checkout changed after preparation."
        raise RuntimeError(msg)


def _save_config(root: Path, name: str, cfg) -> dict:
    path = root / "configs" / f"{name}.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    return {"config": str(path), "config_sha256": _digest(path)}


def _batch_script(campaign: Path, name: str, stage: dict, checkpoint=None) -> str:
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "python",
        "-m",
        "slurm_scripts.ablations.gray_scott_data_scaling",
        "execute",
        "--campaign",
        str(campaign),
        "--stage",
        name,
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    ranks = 1 if stage["kind"] in ("cache", "eval") else 4
    log = Path(stage["workdir"]) / "slurm-%j.out"
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name=gs5x-{name}",
            "#SBATCH --nodes=1",
            f"#SBATCH --gpus={ranks}",
            f"#SBATCH --ntasks-per-node={ranks}",
            "#SBATCH --time=24:00:00",
            "#SBATCH --mem=115G" if ranks == 1 else "#SBATCH --mem=0",
            f"#SBATCH --output={log}",
            "set -euo pipefail",
            "umask 0002",
            f"cd {shlex.quote(str(ROOT))}",
            # Establish the deadline before Python startup and dataset loading.
            "export AUTOCAST_ALLOCATION_END_TIME="
            '"${SLURM_JOB_END_TIME:-$(( $(date +%s) + 86400 ))}"',
            "exec srun " + shlex.join(command),
            "",
        ]
    )


def prepare(
    root: Path,
    data: Path,
    reference: Path,
    *,
    include_fm48=False,
    baseline_manifest: Path | None = None,
) -> dict:
    """Write a fresh, immutable campaign without invoking a scheduler."""
    commit = _source_commit()
    if root.exists():
        raise FileExistsError(root)
    for directory in (data, reference):
        if not directory.is_absolute():
            msg = "Dataset paths must be absolute."
            raise ValueError(msg)
    os.environ.update(
        {
            "AUTOCAST_GS_5X_DATA": str(data),
            "AUTOCAST_GS_REFERENCE_DATA": str(reference),
            "AUTOCAST_GS_5X_CACHE": str(root / "latents"),
            "AUTOCAST_GS_5X_AE_CHECKPOINT": str(root / "ae/autoencoder.ckpt"),
        }
    )
    state = {"source_commit": commit, "root": str(root), "stages": {}}
    (root / "configs").mkdir(parents=True)
    campaign = root / "campaign.json"
    with initialize_config_dir(
        config_dir=str(ROOT / "src/autocast/configs"), version_base=None
    ):
        for name, (entry, experiment, epochs) in RUNS.items():
            if name == "fm48" and not include_fm48:
                continue
            cfg = compose(
                config_name=entry,
                overrides=[
                    f"local_experiment={PREFIX}{experiment}",
                    f"hydra.searchpath=[file://{ROOT}/local_hydra]",
                ],
            )
            workdir = root / name
            workdir.mkdir()
            checkpoint = workdir / str(cfg.output.checkpoint_name)
            with open_dict(cfg):
                cfg.output.checkpoint_path = str(checkpoint)
                cfg.resume_weights_only = False
                cfg.reset_resume_time_budget = False
                cfg.trainer.plugins = [
                    {
                        "_target_": (
                            "lightning.pytorch.plugins.environments.SLURMEnvironment"
                        ),
                        "auto_requeue": False,
                    }
                ]
                for callback in cfg.trainer.callbacks:
                    if str(callback.get("_target_", "")).endswith("ModelCheckpoint"):
                        callback.dirpath = str(workdir / "checkpoints")
                cfg.trainer.callbacks.append(
                    {
                        "_target_": "autocast.callbacks.allocation.AllocationDeadline",
                        "reserve_seconds": 900,
                    }
                )
                cfg.logging.wandb.id = uuid.uuid4().hex[:8]
                cfg.logging.wandb.resume = "allow"
                cfg.logging.wandb.name = f"gs5x_{name}"
            assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == epochs
            stage = {
                "kind": entry,
                "workdir": str(workdir),
                "checkpoint": str(checkpoint),
                "epochs": epochs,
                "requires": ["cache"] if name.startswith("fm") else [],
                **_save_config(root, name, cfg),
            }
            state["stages"][name] = stage

        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"local_experiment={PREFIX}cache_latents",
                f"hydra.searchpath=[file://{ROOT}/local_hydra]",
            ],
        )
        (root / "cache").mkdir()
        state["stages"]["cache"] = {
            "kind": "cache",
            "workdir": str(root / "cache"),
            "requires": ["ae"],
            **_save_config(root, "cache", cfg),
        }
    _prepare_evaluations(root, data, reference, state)
    if baseline_manifest is not None:
        _prepare_baseline_evaluations(root, data, reference, state, baseline_manifest)
    _write(campaign, state)
    for name, stage in state["stages"].items():
        (root / f"{name}.sh").write_text(_batch_script(campaign, name, stage))
    return state


def _prepare_evaluations(root: Path, data: Path, reference: Path, state: dict) -> None:
    """Keep both test cohorts independent of the training latent cache."""
    # Both cohorts use each fitted model's own normalization and frozen AE.
    # FM is evaluated by encoding raw fields, not by reusing the old test cache.
    for parent in list(state["stages"]):
        if not parent.startswith(("crps", "fm")):
            continue
        training_cfg = OmegaConf.load(state["stages"][parent]["config"])
        for cohort, directory in (("original", reference), ("expanded", data)):
            name = f"eval_{parent}_{cohort}"
            raw = OmegaConf.load(state["stages"]["cache"]["config"])
            cfg = _raw_eval_config(
                training_cfg,
                raw,
                directory / "test/data.pt",
                autoencoder_checkpoint=(
                    root / "ae/autoencoder.ckpt" if parent.startswith("fm") else None
                ),
            )
            workdir = root / name
            workdir.mkdir()
            state["stages"][name] = {
                "kind": "eval",
                "parent": parent,
                "cohort": cohort,
                "workdir": str(workdir),
                "requires": [parent],
                **_save_config(root, name, cfg),
            }


def _raw_eval_config(training, raw, test: Path, *, autoencoder_checkpoint=None):
    """Retain each model's fitted normalization and AE for a raw test cohort."""
    if not str(raw.datamodule._target_).endswith("SpatioTemporalDataModule"):
        msg = "Raw evaluation requires the saved spatiotemporal data config."
        raise ValueError(msg)
    cfg = copy.deepcopy(training)
    with open_dict(cfg):
        cfg.datamodule = copy.deepcopy(raw.datamodule)
        cfg.datamodule.full_trajectory_mode = False
        cfg.datamodule.autoencoder_mode = False
        cfg.datamodule.n_steps_input = 1
        cfg.datamodule.n_steps_output = 4
        cfg.datamodule.stride = 1
        cfg.datamodule.validation_stride = 1
        cfg.datamodule.test_data_path = str(test)
        cfg.datamodule.batch_size = 4
        cfg.eval.mode = "encode_once" if autoencoder_checkpoint else "ambient"
        cfg.eval.use_ema = False
        cfg.eval.devices = 1
        cfg.eval.n_members = 10
        cfg.eval.metrics = OmegaConf.load(
            ROOT / "src/autocast/configs/eval/encoder_processor_decoder.yaml"
        ).metrics
        cfg.eval.batch_size = 4
        # Saved evaluation configs can name existing published outputs.
        # Route every artifact into this campaign's fresh stage directory.
        cfg.eval.csv_path = None
        cfg.eval.video_dir = None
        cfg.eval.rollout_snapshot_dir = None
        cfg.eval.dump_rollout_tensors = False
        cfg.eval.dump_rollout_path = None
        cfg.eval.max_rollout_steps = 25
        cfg.eval.metric_windows = [None]
        cfg.eval.metric_windows_rollout = [[0, 1], [0, 4], [6, 12], [13, 30], [31, 99]]
        cfg.eval.max_test_batches = None
        cfg.eval.max_rollout_batches = None
        cfg.eval.compute_rollout_metrics = True
        cfg.eval.compute_rollout_coverage = True
        cfg.eval.trajectory_statistics.enabled = True
        cfg.eval.trajectory_statistics.sampling_seed = 42
        cfg.eval.trajectory_statistics.include_per_timestep = True
        cfg.eval.trajectory_statistics.overwrite_existing = False
        cfg.eval.trajectory_statistics.output_dir = None
        if autoencoder_checkpoint:
            cfg.autoencoder_checkpoint = str(autoencoder_checkpoint)
            cfg.model.encoder = copy.deepcopy(raw.model.encoder)
            cfg.model.decoder = copy.deepcopy(raw.model.decoder)
        cfg.logging.wandb.enabled = False
    return cfg


def _prepare_baseline_evaluations(root, data, reference, state, manifest: Path):
    """Prepare original models on both cohorts from their own saved configs."""
    specifications = OmegaConf.load(manifest)
    if not isinstance(specifications, DictConfig):
        msg = "Baseline manifest must contain FM and CRPS mappings."
        raise ValueError(msg)
    for method in ("fm", "crps"):
        specification = specifications[method]
        keys = ["config", "checkpoint"]
        if method == "fm":
            keys += ["autoencoder_config", "autoencoder_checkpoint"]
        paths = {key: Path(str(specification[key])).expanduser() for key in keys}
        if not all(path.is_absolute() for path in paths.values()):
            msg = "Baseline config and checkpoint paths must be absolute."
            raise ValueError(msg)
        training = OmegaConf.load(paths["config"])
        raw = (
            OmegaConf.load(paths["autoencoder_config"]) if method == "fm" else training
        )
        for cohort, directory in (("original", reference), ("expanded", data)):
            name = f"eval_baseline_{method}_{cohort}"
            cfg = _raw_eval_config(
                training,
                raw,
                directory / "test/data.pt",
                autoencoder_checkpoint=paths.get("autoencoder_checkpoint"),
            )
            cfg.eval.checkpoint = str(paths["checkpoint"])
            workdir = root / name
            workdir.mkdir()
            state["stages"][name] = {
                "kind": "eval",
                "cohort": cohort,
                "baseline": method,
                "checkpoint": str(paths["checkpoint"]),
                "workdir": str(workdir),
                "requires": [],
                "source_config_sha256": _digest(paths["config"]),
                **_save_config(root, name, cfg),
            }
    state["baseline_manifest_sha256"] = _digest(manifest)


def load_stage(campaign: Path, name: str) -> tuple[dict, dict, DictConfig]:
    """Refuse changed code or configs before any allocation performs work."""
    state = _read(campaign)
    _check_source(state)
    stage = state["stages"][name]
    path = Path(stage["config"])
    if _digest(path) != stage["config_sha256"]:
        msg = "Prepared config changed; prepare a new campaign."
        raise RuntimeError(msg)
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    return state, stage, cfg


def _require_completed(state: dict, name: str) -> None:
    status = Path(state["stages"][name]["workdir"]) / "status.json"
    if not status.is_file() or _read(status)["status"] != "complete":
        msg = f"Stage {name} has not completed its epoch or time budget."
        raise RuntimeError(msg)


def _checkpoint_status(stage: dict, cfg) -> dict:
    payload = torch.load(
        stage["checkpoint"], map_location="cpu", mmap=True, weights_only=False
    )
    epochs = payload["loops"]["fit_loop"]["epoch_progress"]["total"]["completed"]
    elapsed = payload["callbacks"]["Timer"]["time_elapsed"]["train"]
    days, hours, minutes, seconds = map(int, str(cfg.trainer.max_time).split(":"))
    cap = days * 86400 + hours * 3600 + minutes * 60 + seconds
    done = epochs >= stage["epochs"] or elapsed >= cap
    return {
        "status": "complete" if done else "continuation_required",
        "completed_epochs": epochs,
        "target_epochs": stage["epochs"],
        "global_step": payload["global_step"],
        "elapsed_training_seconds": elapsed,
        "epoch_horizon_complete": epochs >= stage["epochs"],
        "checkpoint": stage["checkpoint"],
        "checkpoint_sha256": _digest(Path(stage["checkpoint"])),
    }


def _execute_eval(state: dict, stage: dict, cfg: DictConfig, workdir: Path) -> None:
    """Evaluate the final selected model on one explicit raw test cohort."""
    from autocast.scripts.eval.encoder_processor_decoder import run_evaluation

    if "baseline" in stage:
        selected = Path(stage["checkpoint"])
    elif stage["parent"].startswith("crps"):
        parent = state["stages"][stage["parent"]]
        candidates = list(
            (Path(parent["workdir"]) / "checkpoints").glob(
                "best-multiwinkler-overall-*.ckpt"
            )
        )
        if len(candidates) != 1:
            msg = "Expected one validation-selected CRPS checkpoint."
            raise ValueError(msg)
        selected = candidates[0]
    else:
        parent = state["stages"][stage["parent"]]
        selected = Path(parent["checkpoint"])
    cfg.eval.checkpoint = str(selected.resolve(strict=True))
    if any(workdir.glob("*.csv")):
        raise FileExistsError(workdir)
    run_evaluation(cfg, work_dir=workdir)


def _execute_cache(cfg: DictConfig, workdir: Path, checkpoint) -> None:
    """Require a complete frozen AE before encoding every requested trajectory."""
    if checkpoint is not None:
        msg = "Latent caching cannot resume."
        raise ValueError(msg)
    from autocast.scripts.cache_latents import cache_latents

    target = Path(cfg.cache_latents.output_dir)
    if target.exists():
        raise FileExistsError(target)
    cache_latents(cfg, target, device="cuda")
    metadata = _read(target / "metadata.json")
    actual = {
        key: value["num_trajectories"] for key, value in metadata["splits"].items()
    }
    assert actual == {"train": 1200, "valid": 120, "test": 24}, actual
    _write(workdir / "status.json", {"status": "complete", "counts": actual})


def execute(campaign: Path, name: str, checkpoint: Path | None) -> None:
    """Run only inside an allocation; continuations require an explicit path."""
    if "SLURM_JOB_ID" not in os.environ:
        msg = "Execute stages inside an allocation using the prepared batch script."
        raise RuntimeError(msg)
    state, stage, cfg = load_stage(campaign, name)
    for dependency in stage["requires"]:
        _require_completed(state, dependency)
    workdir = Path(stage["workdir"])
    if stage["kind"] == "eval":
        _execute_eval(state, stage, cfg, workdir)
        return
    if stage["kind"] == "cache":
        _execute_cache(cfg, workdir, checkpoint)
        return

    if checkpoint is None and Path(stage["checkpoint"]).exists():
        msg = "A checkpoint already exists; prepare an explicit continuation."
        raise FileExistsError(msg)
    if checkpoint is not None:
        checkpoint, _ = _continuation_checkpoint(stage, checkpoint)
        with open_dict(cfg):
            cfg.resume_from_checkpoint = str(checkpoint)
            cfg.logging.wandb.resume = "must"
            cfg.trainer.callbacks.append(
                {
                    "_target_": "autocast.callbacks.allocation.VerifyResumeState",
                    "checkpoint": str(checkpoint),
                }
            )
    if stage["kind"] == "autoencoder":
        from autocast.scripts.training import train_autoencoder

        train_autoencoder(cfg, workdir)
    elif stage["kind"] == "encoder_processor_decoder":
        from autocast.scripts.train.encoder_processor_decoder import run_epd_training

        run_epd_training(cfg, work_dir=workdir)
    else:
        from autocast.scripts.setup import setup_datamodule, setup_processor_model
        from autocast.scripts.training import run_training

        datamodule, cfg, stats = setup_datamodule(cfg)
        L.seed_everything(cfg.seed, workers=True)
        model = setup_processor_model(cfg, stats, datamodule=datamodule)
        run_training(
            cfg,
            model,
            datamodule,
            workdir,
            skip_test=True,
            output_checkpoint_path=stage["checkpoint"],
            job_type="train-processor",
        )
    if int(os.environ.get("SLURM_PROCID", "0")) == 0:
        status = _checkpoint_status(stage, cfg)
        _write(workdir / "status.json", status)
        print(json.dumps(status), flush=True)


def _continuation_checkpoint(stage: dict, checkpoint: Path) -> tuple[Path, int]:
    """Continue only a recorded, cleanly stopped training segment."""
    if "epochs" not in stage:
        msg = "Only training stages support continuation."
        raise ValueError(msg)
    status = _read(Path(stage["workdir"]) / "status.json")
    if status["status"] != "continuation_required":
        msg = "This experiment has already completed its epoch or time budget."
        raise ValueError(msg)
    checkpoint = checkpoint.resolve(strict=True)
    if _digest(checkpoint) != status.get("checkpoint_sha256"):
        msg = "Checkpoint does not match the recorded checkpoint identity."
        raise ValueError(msg)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if payload["global_step"] != status["global_step"]:
        msg = "Checkpoint does not match the latest recorded training step."
        raise ValueError(msg)
    return checkpoint, status["global_step"]


def main(argv=None) -> None:
    """Prepare and inspect jobs; submitting remains an explicit manual action."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-root", type=Path, required=True)
    prepare_parser.add_argument("--data", type=Path, required=True)
    prepare_parser.add_argument("--reference-data", type=Path, required=True)
    prepare_parser.add_argument("--include-fm48", action="store_true")
    prepare_parser.add_argument("--baseline-manifest", type=Path)
    for action in ("preview", "continue", "execute"):
        child = sub.add_parser(action)
        child.add_argument("--campaign", type=Path, required=True)
        child.add_argument("--stage", required=True)
        child.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        state = prepare(
            args.run_root.resolve(),
            args.data,
            args.reference_data,
            include_fm48=args.include_fm48,
            baseline_manifest=args.baseline_manifest,
        )
        print(json.dumps(state, indent=2))
    elif args.action == "execute":
        execute(args.campaign.resolve(), args.stage, args.checkpoint)
    else:
        _, stage, _ = load_stage(args.campaign, args.stage)
        if args.action == "continue":
            checkpoint = args.checkpoint or Path(stage["checkpoint"])
            checkpoint, step = _continuation_checkpoint(stage, checkpoint)
            script = args.campaign.parent / f"{args.stage}-continue-{step}.sh"
            if script.exists():
                raise FileExistsError(script)
            script.write_text(
                _batch_script(args.campaign.resolve(), args.stage, stage, checkpoint)
            )
            print(f"Prepared continuation: sbatch {shlex.quote(str(script))}")
        else:
            print(_batch_script(args.campaign.resolve(), args.stage, stage))


if __name__ == "__main__":
    main()
