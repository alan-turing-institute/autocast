"""Prepare ambient RB jobs locally; never submit or execute them."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import uuid
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from slurm_scripts.comparison.the_well.rayleigh_benard.paired_test_windows import (
    make_selection,
)

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = {
    "matched": "the_well/rayleigh_benard/crps_spatiotemporal_ambient_matched",
    "lola": "the_well/rayleigh_benard/crps_spatiotemporal_ambient",
}
COHORT = Path(__file__).with_name("test64_seed0.yaml")
REFERENCE_EPOCHS = 4096


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_commit() -> str:
    if subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip():
        msg = "Commit the checkout before preparing jobs."
        raise RuntimeError(msg)
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _start(output: Path) -> str:
    commit = _source_commit()
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    return commit


def _write_job(output, name, cfg, command, *, commit, gpus, minutes, run_dir, nodes=1):
    path = output / "configs" / f"{name}.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    checksum = _digest(path)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=rb-ambient-{name}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --gpus-per-node={gpus}",
        f"#SBATCH --ntasks-per-node={gpus}",
        f"#SBATCH --cpus-per-task={16 if gpus == 4 else 8}",
        f"#SBATCH --time={minutes // 60:02d}:{minutes % 60:02d}:00",
        "#SBATCH --no-requeue",
        "#SBATCH --mem=0" if gpus == 4 else "#SBATCH --mem=115G",
        f"#SBATCH --output={shlex.quote(str(output / (name + '-%j.out')))}",
        "set -euo pipefail",
        "umask 0002",
        ': "${SLURM_JOB_ID:?Run inside a Slurm allocation}"',
        f"cd {shlex.quote(str(ROOT))}",
        'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"',
        'export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
        "export RUNTIME_TYPECHECKING=false",
        f'test "$(git rev-parse HEAD)" = {shlex.quote(commit)}',
        'test -z "$(git status --porcelain)"',
        "printf '%s  %s\\n' " + shlex.join([checksum, str(path)]) + " | sha256sum -c -",
    ]
    if nodes > 1:
        lines += [
            "module load brics/nccl/2.26.6-1",
            "export NCCL_DEBUG=INFO",
            "export NCCL_DEBUG_FILE=" + shlex.quote(str(output / "nccl-%h-%p.log")),
        ]
    run_dir = shlex.quote(str(run_dir))
    lines += [
        f"if [[ -e {run_dir} || -L {run_dir} ]]; then",
        "  echo 'Run directory exists; use fresh output or full-state resume.' >&2",
        "  exit 1",
        "fi",
        "exec srun --kill-on-bad-exit=1 "
        + shlex.join(["uv", "run", "--frozen", "--no-sync", *command]),
        "",
    ]
    (output / f"{name}.sh").write_text("\n".join(lines))
    return {
        "config": str(path),
        "config_sha256": checksum,
        "nodes": nodes,
        "gpus_per_node": gpus,
        "gpus": nodes * gpus,
        "minutes": minutes,
    }


def prepare_training(  # noqa: PLR0915
    output: Path,
    datasets: Path,
    phase: str,
    microbatch=None,
    epochs=None,
    *,
    nodes=8,
    size="matched",
):
    """Preserve 32 real windows and 256 member predictions per optimizer update."""
    if size not in EXPERIMENTS:
        msg = "Use matched or lola for the model size."
        raise ValueError(msg)
    if nodes not in (1, 2, 4, 8) or not datasets.is_absolute():
        msg = "Use an absolute dataset root and 1, 2, 4 or 8 four-GPU nodes."
        raise ValueError(msg)
    microbatch = 8 // nodes if microbatch is None else microbatch
    if microbatch not in (1, 2, 4, 8) or microbatch > 8 // nodes:
        msg = "Use a microbatch that preserves 32 real windows across all ranks."
        raise ValueError(msg)
    if phase not in ("smoke", "timing", "train"):
        raise ValueError(phase)
    target = (
        {"smoke": 1, "timing": 16, "train": REFERENCE_EPOCHS}[phase]
        if epochs is None
        else epochs
    )
    if (phase == "train" and target != REFERENCE_EPOCHS) or (
        phase == "smoke" and target != 1
    ):
        msg = "Matched production requires 4096 epochs; smoke requires one."
        raise ValueError(msg)
    if target < 1 or (phase == "timing" and (target < 8 or target % 8)):
        msg = "Use positive epochs; timing requires a multiple of eight."
        raise ValueError(msg)
    commit = _start(output)
    with initialize_config_dir(str(ROOT / "src/autocast/configs"), version_base=None):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"local_experiment={EXPERIMENTS[size]}",
                f"hydra.searchpath=[file://{ROOT}/local_hydra]",
            ],
        )
    accumulation = (8 // nodes) // microbatch
    workdir = output / "run"
    checkpoint = workdir / (
        "encoder_processor_decoder.ckpt" if phase == "train" else f"{phase}.ckpt"
    )
    with open_dict(cfg):
        cfg.datamodule.well_base_path = str(datasets)
        cfg.datamodule.batch_size = microbatch
        cfg.optimizer.cosine_epochs = REFERENCE_EPOCHS
        cfg.trainer.max_epochs = target
        cfg.trainer.num_nodes = nodes
        cfg.trainer.max_time = None
        cfg.trainer.accumulate_grad_batches = accumulation
        cfg.trainer.limit_train_batches = (
            accumulation if phase == "smoke" else 64 * accumulation
        )
        cfg.trainer.limit_val_batches = 1 if phase == "smoke" else 16 * accumulation
        cfg.trainer.check_val_every_n_epoch = 1 if phase == "smoke" else 8
        cfg.trainer.plugins = [
            {
                "_target_": "lightning.pytorch.plugins.environments.SLURMEnvironment",
                "auto_requeue": False,
            }
        ]
        for callback in cfg.trainer.callbacks:
            if str(callback.get("_target_", "")).endswith("ModelCheckpoint"):
                callback.dirpath = str(workdir / "checkpoints")
        if phase != "train":
            cfg.trainer.callbacks[0].every_n_epochs = target
        else:
            # Check only at complete epochs; measure epoch/save time before
            # relying on this 30-minute allocation headroom in production.
            cfg.trainer.callbacks.append(
                {
                    "_target_": "lightning.pytorch.callbacks.Timer",
                    "duration": "00:23:30:00",
                    "interval": "epoch",
                }
            )
        cfg.output.checkpoint_path = str(checkpoint)
        cfg.output.save_config = True
        cfg.resume_from_checkpoint = None
        cfg.resume_weights_only = False
        cfg.reset_resume_time_budget = False
        cfg.logging.wandb.enabled = phase == "train"
        cfg.logging.wandb.id = uuid.uuid4().hex[:8]
        cfg.logging.wandb.resume = "allow"
        cfg.logging.wandb.name = f"rb_ambient_{size}_crps_{phase}"
    command = [
        "python",
        "-m",
        "autocast.scripts.train.encoder_processor_decoder",
        "--config-path",
        str(output / "configs"),
        "--config-name",
        phase,
        f"hydra.run.dir={workdir}",
    ]
    job = _write_job(
        output,
        phase,
        cfg,
        command,
        commit=commit,
        gpus=4,
        minutes={"smoke": 15, "timing": 30 if nodes == 8 else 240, "train": 1440}[
            phase
        ],
        run_dir=workdir,
        nodes=nodes,
    )
    state = {
        "source_commit": commit,
        "datasets_root": str(datasets),
        "size": size,
        "nodes": nodes,
        "microbatch": microbatch,
        "phase": phase,
        "epochs": target,
        "accumulation": accumulation,
        "optimizer_updates_per_epoch": 1 if phase == "smoke" else 64,
        "jobs": {phase: job},
    }
    (output / "preparation.json").write_text(json.dumps(state, indent=2) + "\n")
    return state


def prepare_evaluation(
    output: Path, source_config: Path, checkpoint: Path, expected_step: int
):
    """Reuse the completed comparison's 64 trajectories and 512 physical windows."""
    cfg = OmegaConf.load(source_config)
    if (
        not isinstance(cfg, DictConfig)
        or not checkpoint.is_absolute()
        or expected_step < 1
    ):
        msg = "Supply a saved config, absolute checkpoint and positive step."
        raise ValueError(msg)
    if not str(cfg.datamodule._target_).endswith("TheWellDataModule") or not str(
        cfg.model.encoder._target_
    ).endswith("IdentityEncoder"):
        msg = "Evaluation requires this ambient model's saved raw-data config."
        raise ValueError(msg)
    commit = _start(output)
    cohort = output / "test64_seed0.yaml"
    cohort.write_bytes(COHORT.read_bytes())
    windows = output / "paired_test512_seed0.yaml"
    OmegaConf.save(
        OmegaConf.create(
            make_selection(OmegaConf.to_container(OmegaConf.load(cohort), resolve=True))
        ),
        windows,
    )
    with open_dict(cfg):
        cfg.datamodule.batch_size = 1
        cfg.datamodule.max_rollout_steps = 200
        cfg.eval.mode = "ambient"
        cfg.eval.checkpoint = str(checkpoint)
        cfg.eval.use_ema = False
        cfg.eval.devices = cfg.eval.num_nodes = 1
        cfg.eval.n_members = 16
        cfg.eval.batch_size = 1
        cfg.eval.chunk_size = 8
        cfg.eval.transpose_spatial = True
        cfg.eval.batch_indices = [0, 1, 2, 3, 4, 5]
        cfg.eval.rollout_member_indices = [0]
        cfg.eval.rollout_member_render_mode = "both"
        cfg.eval.rollout_start = 16
        cfg.eval.max_rollout_steps = 46
        cfg.eval.rollout_subset = str(cohort)
        cfg.eval.compute_test_metrics = False
        cfg.eval.compute_rollout_metrics = True
        cfg.eval.compute_rollout_coverage = True
        cfg.eval.compute_rollout_autoencoded_target_metrics = False
        # A null rollout limit inherits max_test_batches in the shared runner.
        cfg.eval.max_test_batches = cfg.eval.max_rollout_batches = None
        cfg.eval.metric_windows_rollout = [None, [0, 20], [20, 60], [60, 183], [0, 84]]
        cfg.eval.csv_path = None
        cfg.eval.video_dir = None
        cfg.eval.rollout_snapshot_dir = None
        cfg.eval.trajectory_statistics.enabled = False
        cfg.eval.dump_rollout_tensors = False
        cfg.eval.benchmark.enabled = False
        cfg.eval.benchmark_rollout.enabled = False
        cfg.logging.wandb.enabled = False
    path = output / "configs" / "rollout.yaml"
    code = (
        "import sys\nfrom pathlib import Path\n"
        "import lightning as L\nimport torch\nfrom omegaconf import OmegaConf\n"
        "from autocast.scripts.eval.encoder_processor_decoder import run_evaluation\n"
        "cfg = OmegaConf.load(sys.argv[1])\n"
        "payload = torch.load(cfg.eval.checkpoint, map_location='cpu',\n"
        "    mmap=True, weights_only=False)\n"
        "if payload.get('global_step') != int(sys.argv[3]):\n"
        "    raise ValueError('Checkpoint step differs from selected endpoint')\n"
        "del payload\nL.seed_everything(42, workers=True)\n"
        "run_evaluation(cfg, work_dir=Path(sys.argv[2]))\n"
    )
    rollout = _write_job(
        output,
        "rollout",
        cfg,
        ["python", "-c", code, str(path), str(output / "rollout"), str(expected_step)],
        commit=commit,
        gpus=1,
        minutes=1439,
        run_dir=output / "rollout",
    )
    paired = _write_job(
        output,
        "paired",
        cfg,
        [
            "python",
            "-m",
            "slurm_scripts.comparison.the_well.rayleigh_benard.paired_test_windows",
            "run",
            "--config",
            str(output / "configs/paired.yaml"),
            "--manifest",
            str(windows),
            "--output",
            str(output / "paired"),
            "--members",
            "16",
            "--expected-step",
            str(expected_step),
        ],
        commit=commit,
        gpus=1,
        minutes=240,
        run_dir=output / "paired",
    )
    state = {
        "source_commit": commit,
        "source_config": str(source_config.resolve()),
        "source_config_sha256": _digest(source_config),
        "checkpoint": str(checkpoint),
        "expected_step": expected_step,
        "cohort_sha256": _digest(cohort),
        "windows_sha256": _digest(windows),
        "jobs": {"rollout": rollout, "paired": paired},
    }
    (output / "preparation.json").write_text(json.dumps(state, indent=2) + "\n")
    return state


def main():
    """Write resolved configs and batch scripts; never submit jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("training")
    train.add_argument("--phase", choices=["smoke", "timing", "train"], default="smoke")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--datasets-root", type=Path, required=True)
    train.add_argument("--nodes", type=int, choices=[1, 2, 4, 8], default=8)
    train.add_argument("--size", choices=list(EXPERIMENTS), default="matched")
    train.add_argument("--microbatch", type=int)
    train.add_argument("--epochs", type=int)
    evaluate = sub.add_parser("evaluation")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()
    if args.action == "training":
        state = prepare_training(
            args.output.resolve(),
            args.datasets_root,
            args.phase,
            args.microbatch,
            args.epochs,
            nodes=args.nodes,
            size=args.size,
        )
    else:
        state = prepare_evaluation(
            args.output.resolve(), args.config, args.checkpoint, args.expected_step
        )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
