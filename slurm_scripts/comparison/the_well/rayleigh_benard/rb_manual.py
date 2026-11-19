"""Prepare and verify RB allocations; submission requires running queue.sh."""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from omegaconf import DictConfig, OmegaConf

from autocast.callbacks.ema import EMACallback
from autocast.scripts.train.encoder_processor_decoder import run_epd_training
from autocast.utils.device import pin_local_cuda_device

COMMIT = "4b2c0499ff637496f540ab7c38742057ec57de15"
UPDATES = 64
EPOCHS = 4096
STAGES = ("first", "second", "third", "fourth")


def digest(path):
    """Hash an artifact without loading it into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    """Reject an inconsistent preparation or restored state."""
    if not condition:
        raise ValueError(message)


def same(left, right):
    """Compare nested restored tensors and metadata exactly."""
    # Same comparison pattern as the existing GS full-state resume verifier.
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same(left[k], right[k]) for k in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def checkpoint_summary(path, *, updates=UPDATES):
    """Require a full checkpoint from a completed capped epoch."""
    path = Path(path)
    require(
        path.is_file() and not path.is_symlink(),
        "Use a concrete final checkpoint, not an alias.",
    )
    c = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    require(bool(c.get("state_dict")), "Model weights are missing.")
    fit = c["loops"]["fit_loop"]
    progress = fit["epoch_progress"]["current"]
    completed = int(progress["completed"])
    step = int(c["global_step"])
    require(
        completed > 0 and all(int(v) == completed for v in progress.values()),
        "Checkpoint is not at a completed epoch boundary.",
    )
    require(
        step == completed * updates, "Optimizer progress differs from completed epochs."
    )
    batch = fit["epoch_loop.batch_progress"]
    require(
        batch["is_last_batch"]
        and all(int(v) == updates for v in batch["current"].values()),
        "The final epoch has incomplete batches.",
    )
    require(
        len(c.get("optimizer_states", [])) == 1
        and bool(c["optimizer_states"][0]["state"]),
        "Full optimizer state is missing.",
    )
    require(
        all(
            int(
                state["step"].item()
                if torch.is_tensor(state["step"])
                else state["step"]
            )
            == step
            for state in c["optimizer_states"][0]["state"].values()
        ),
        "Adam step counters differ from global progress.",
    )
    require(len(c.get("lr_schedulers", [])) == 1, "Scheduler state is missing.")
    scheduler = c["lr_schedulers"][0]
    require(
        scheduler["last_epoch"] == completed, "Scheduler and epoch progress disagree."
    )
    expected_lr = 1e-4 * (1 + math.cos(math.pi * completed / EPOCHS)) / 2
    rates = [g["lr"] for g in c["optimizer_states"][0]["param_groups"]]
    require(
        all(math.isclose(lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12) for lr in rates),
        "Checkpoint LR differs from the full 4096-epoch cosine.",
    )
    require(
        bool(c.get("ema_state_dict"))
        and bool(c["callbacks"].get("EMACallback", {}).get("ema_model_state")),
        "EMA state is missing.",
    )
    require(
        int(c["callbacks"]["EMACallback"]["ema_model_state"]["n_averaged"]) == step,
        "EMA progress differs from optimizer progress.",
    )
    return {
        "epoch": int(c["epoch"]),
        "completed_epochs": completed,
        "global_step": step,
        "learning_rates": rates,
        "scheduler_epoch": scheduler["last_epoch"],
        "optimizer_parameter_states": len(c["optimizer_states"][0]["state"]),
        "full_state": True,
        "epoch_boundary": True,
    }


class ResumeGuard(L.Callback):
    """Verify restored state on every rank before its first optimizer update."""

    def __init__(self, record, expected_step=0, checkpoint=None, world_size=32):
        self.record = Path(record)
        self.expected_step = expected_step
        self.checkpoint = checkpoint
        self.world_size = world_size

    def verify(self, trainer, model):
        require(
            trainer.world_size == self.world_size, "Unexpected distributed world size."
        )
        require(
            trainer.global_step == self.expected_step, "Wrong restored global step."
        )
        require(trainer.max_epochs == EPOCHS, "Total epoch horizon changed.")
        if self.checkpoint is not None:
            c = torch.load(
                self.checkpoint, map_location="cpu", mmap=True, weights_only=False
            )
            require(
                same(model.state_dict(), c["state_dict"]),
                "Restored model differs from checkpoint.",
            )
            require(
                len(trainer.optimizers) == 1
                and same(trainer.optimizers[0].state_dict(), c["optimizer_states"][0]),
                "Optimizer state was not fully restored.",
            )
            require(
                same(
                    trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                    c["lr_schedulers"][0],
                ),
                "Scheduler state was not fully restored.",
            )
            for callback in trainer.callbacks:
                if not isinstance(callback, (EMACallback, ModelCheckpoint)):
                    continue
                expected = c["callbacks"][callback.state_key]
                actual = callback.state_dict()
                if isinstance(callback, ModelCheckpoint):
                    expected = {
                        k: v for k, v in expected.items() if k != "current_score"
                    }
                    actual = {k: v for k, v in actual.items() if k != "current_score"}
                    require(
                        not callback.best_model_path
                        or Path(callback.best_model_path).is_file(),
                        "Previous best checkpoint is unavailable.",
                    )
                require(
                    same(actual, expected),
                    f"Callback state differs: {callback.state_key}",
                )
            timer = next(c for c in trainer.callbacks if isinstance(c, Timer))
            require(timer._offset == 0, "Allocation timer was not reset.")
            remaining = timer.time_remaining()
            require(
                remaining is not None and remaining > 0,
                "New allocation has no timer budget.",
            )
        else:
            require(
                not trainer.optimizers[0].state_dict()["state"],
                "Initial segment already has optimizer history.",
            )
            require(
                all(g["lr"] == 1e-4 for g in trainer.optimizers[0].param_groups),
                "Wrong initial learning rate.",
            )
        return {
            "status": "passed",
            "ranks": trainer.world_size,
            "global_step": trainer.global_step,
            "learning_rates": [g["lr"] for g in trainer.optimizers[0].param_groups],
            "full_state_resume": self.checkpoint is not None,
        }

    def on_train_start(self, trainer, pl_module):
        error, result = None, None
        try:
            result = self.verify(trainer, pl_module)
        except Exception as exc:
            error = f"rank {trainer.global_rank}: {type(exc).__name__}: {exc}"
        errors = [error]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            errors = [None] * trainer.world_size
            torch.distributed.all_gather_object(errors, error)
        require(not any(errors), f"RB restored-state verification failed: {errors}")
        if trainer.is_global_zero:
            self.record.write_text(json.dumps(result, indent=2) + "\n")
            print("RB_STATE_VERIFIED " + json.dumps(result), flush=True)
        trainer.strategy.barrier("rb-state-verified")


def set_budget(cfg, hours):
    """Reserve thirty minutes inside each manual allocation."""
    timers = [
        c for c in cfg["trainer"]["callbacks"] if c["_target_"].endswith(".Timer")
    ]
    require(
        len(timers) == 1 and timers[0]["interval"] == "epoch",
        "Expected one epoch-boundary Timer.",
    )
    require(hours in (12, 14, 24), "Use a 12-, 14- or 24-hour allocation.")
    timers[0]["duration"] = f"00:{hours - 1:02d}:30:00"
    cfg["trainer"]["max_time"] = None


def scientific(cfg):
    """Remove allocation settings before comparing experiment configs."""
    result = copy.deepcopy(cfg)
    for key in ("resume_from_checkpoint", "reset_resume_time_budget"):
        result.pop(key, None)
    result["logging"]["wandb"].pop("resume", None)
    result["output"].pop("checkpoint_path", None)
    callbacks = []
    for callback in result["trainer"]["callbacks"]:
        if callback["_target_"] == "rb_manual.ResumeGuard":
            continue
        if callback["_target_"].endswith(".Timer"):
            callback.pop("duration", None)
        callbacks.append(callback)
    result["trainer"]["callbacks"] = callbacks
    return result


def write_stage(destination, plan, cfg, stage, hours, checkpoint_sha=None):
    """Write a guarded batch script without submitting it."""
    destination.mkdir(parents=True, exist_ok=False)
    target = Path(plan["target_root"])
    remote_stage = target / stage
    config_path = destination / "train.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config_path, resolve=True)
    q = shlex.quote
    helper_check = (
        "printf '%s  %s\\n' "
        + shlex.join([plan["helper_sha256"], str(target / "rb_manual.py")])
        + " | sha256sum -c -"
    )
    config_check = (
        "printf '%s  %s\\n' "
        + shlex.join([digest(config_path), str(remote_stage / "train.yaml")])
        + " | sha256sum -c -"
    )
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=rb-matched-{stage}",
        "#SBATCH --nodes=8",
        "#SBATCH --gpus-per-node=4",
        "#SBATCH --ntasks-per-node=4",
        "#SBATCH --cpus-per-task=16",
        f"#SBATCH --time={hours:02d}:00:00",
        "#SBATCH --mem=0",
        "#SBATCH --no-requeue",
        f"#SBATCH --output={remote_stage}/slurm-%j.out",
        "set -euo pipefail",
        "umask 0002",
        ': "${SLURM_JOB_ID:?Run inside a Slurm allocation}"',
        f"cd {q(plan['checkout'])}",
        f'test "$(git rev-parse HEAD)" = {q(COMMIT)}',
        'test -z "$(git status --porcelain)"',
        helper_check,
        config_check,
        f"mkdir {q(str(target / '.active-segment'))}",
        f"trap 'rmdir {q(str(target / '.active-segment'))}' EXIT",
        "export PYTHONPATH="
        + q(plan["checkout"] + "/src:" + plan["checkout"] + ":" + str(target)),
        'export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
        "export RUNTIME_TYPECHECKING=false",
        "module load brics/nccl/2.26.6-1",
        "export NCCL_DEBUG=INFO",
        f"export NCCL_DEBUG_FILE={q(str(remote_stage / 'nccl-%h-%p.log'))}",
    ]
    run = target / "run"
    if stage == "first":
        lines += [
            f"if [[ -e {q(str(run))} || -L {q(str(run))} ]]; then",
            "  echo 'Run already exists; use continuation.' >&2",
            "  exit 1",
            "fi",
        ]
    else:
        lines += [
            f"test -d {q(str(run))}",
            f"test ! -e {q(str(remote_stage / 'completed.json'))}",
            "printf '%s  %s\\n' "
            + shlex.join([str(checkpoint_sha), cfg["resume_from_checkpoint"]])
            + " | sha256sum -c -",
        ]
    uv = [
        "uv",
        "run",
        "--project",
        plan["checkout"],
        "--frozen",
        "--no-sync",
        "python",
        str(target / "rb_manual.py"),
    ]
    lines += [
        "srun --kill-on-bad-exit=1 "
        + shlex.join(
            [
                *uv,
                "run",
                "--config",
                str(remote_stage / "train.yaml"),
                "--work-dir",
                str(run),
            ]
        ),
        config_check,
        helper_check,
        f'test "$(git rev-parse HEAD)" = {q(COMMIT)}',
        'test -z "$(git status --porcelain)"',
        shlex.join(
            [
                *uv,
                "seal",
                "--config",
                str(remote_stage / "train.yaml"),
                "--stage-dir",
                str(remote_stage),
            ]
        ),
        "",
    ]
    script = destination / "train.sh"
    script.write_text("\n".join(lines))
    script.chmod(0o755)


def prepare_first(source, destination, target_root, *, first_hours=14, size="matched"):
    """Freeze the reviewed scientific revision into a manual allocation plan."""
    require(first_hours in (12, 14, 24), "Use a 12-, 14- or 24-hour first allocation.")
    require(size in ("matched", "lola"), "Use matched or lola for the model size.")
    require(
        source["head"] == COMMIT and not source["git_status"],
        "Source checkout must be clean at the reviewed commit.",
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(source["config"])
    cfg.update(
        resume_from_checkpoint=None,
        resume_weights_only=False,
        reset_resume_time_budget=False,
    )
    target = Path(target_root)
    require(target.is_absolute(), "Use an absolute target root.")
    require(
        cfg["trainer"]["num_nodes"] == 8 and cfg["trainer"]["devices"] == 4,
        "Wrong rank layout.",
    )
    require(
        cfg["datamodule"]["batch_size"]
        == cfg["trainer"]["accumulate_grad_batches"]
        == 1,
        "Wrong global batch.",
    )
    require(
        cfg["trainer"]["limit_train_batches"] == UPDATES
        and cfg["trainer"]["max_epochs"] == cfg["optimizer"]["cosine_epochs"] == EPOCHS,
        "Training horizon changed.",
    )
    backbone = cfg["model"]["processor"]["backbone"]
    width, heads = (1024, 8) if size == "matched" else (2048, 16)
    require(
        backbone["hid_channels"] == width
        and backbone["attention_heads"] == heads
        and backbone["hid_blocks"] == 16,
        "Backbone differs from the selected model size.",
    )
    cfg["output"]["checkpoint_path"] = str(target / "first/training-final.ckpt")
    for callback in cfg["trainer"]["callbacks"]:
        if callback["_target_"].endswith("ModelCheckpoint"):
            callback["dirpath"] = str(target / "run/checkpoints")
    set_budget(cfg, first_hours)
    cfg["trainer"]["callbacks"].append(
        {
            "_target_": "rb_manual.ResumeGuard",
            "record": str(target / "first/start-verified.json"),
        }
    )
    helper = destination / "rb_manual.py"
    shutil.copyfile(__file__, helper)
    plan = {
        "target_root": str(target),
        "checkout": source["checkout"],
        "source_commit": COMMIT,
        "source_config_sha256": source["source_config_sha256"],
        "helper_sha256": digest(helper),
        "primary_endpoint": {"global_step": UPDATES * EPOCHS, "use_ema": False},
        "first_allocation_hours": first_hours,
        "size": size,
        "continuation_allocation_hours": 24,
        "jobs_submitted": False,
    }
    write_stage(destination / "first", plan, cfg, "first", first_hours)
    plan["first_config_sha256"] = digest(destination / "first/train.yaml")
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def seal(config, stage_dir):
    """Freeze a completed segment only after collective saving finishes."""
    config, stage = Path(config), Path(stage_dir)
    cfg = cast(
        dict[str, Any], OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    )
    source = Path(cfg["output"]["checkpoint_path"])
    summary = checkpoint_summary(source)
    target = stage / "final.ckpt"
    require(
        not target.exists() and not target.is_symlink(),
        "Frozen checkpoint already exists.",
    )
    before = source.stat()
    shutil.copyfile(source, target)
    after = source.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "Checkpoint changed while freezing it.",
    )
    require(digest(source) == digest(target), "Frozen checkpoint checksum differs.")
    result = {
        **summary,
        "checkpoint": str(target),
        "checkpoint_sha256": digest(target),
        "config": str(config),
        "config_sha256": digest(config),
        "wandb_id": cfg["logging"]["wandb"]["id"],
        "source_commit": COMMIT,
    }
    require(
        not (stage / "completed.json").exists(), "Completion record already exists."
    )
    (stage / "completed.json").write_text(json.dumps(result, indent=2) + "\n")
    if summary["global_step"] == UPDATES * EPOCHS:
        (stage.parent / "completed.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print("RB_SEGMENT_COMPLETE " + json.dumps(result), flush=True)
    return result


def read_completion(plan_path, completed_path):
    """Check checkpoint ownership, progress and the original experiment identity."""
    plan = json.loads(Path(plan_path).read_text())
    complete = json.loads(Path(completed_path).read_text())
    require(
        plan["source_commit"] == complete["source_commit"] == COMMIT,
        "Source identities disagree.",
    )
    require(digest(__file__) == plan["helper_sha256"], "Operational helper changed.")
    require(
        digest(complete["config"]) == complete["config_sha256"],
        "Original segment config changed.",
    )
    require(
        digest(complete["checkpoint"]) == complete["checkpoint_sha256"],
        "Frozen checkpoint changed.",
    )
    root = Path(plan["target_root"])
    config = Path(complete["config"])
    require(
        config.parent.parent == root
        and config.parent.name in STAGES
        and config.name == "train.yaml"
        and Path(complete["checkpoint"]) == config.parent / "final.ckpt",
        "Completion record belongs to a different run.",
    )
    first_path = root / "first/train.yaml"
    require(
        digest(first_path) == plan["first_config_sha256"],
        "First segment config changed.",
    )
    summary = checkpoint_summary(complete["checkpoint"])
    require(
        summary["global_step"] == complete["global_step"] <= UPDATES * EPOCHS,
        "Wrong checkpoint progress.",
    )
    cfg = cast(
        dict[str, Any],
        OmegaConf.to_container(OmegaConf.load(complete["config"]), resolve=True),
    )
    require(
        cfg["logging"]["wandb"]["id"] == complete["wandb_id"],
        "W&B run identity changed.",
    )
    first = OmegaConf.to_container(OmegaConf.load(first_path), resolve=True)
    require(scientific(cfg) == scientific(first), "Scientific settings changed.")
    return plan, complete, summary, cfg


def prepare_resume(plan_path, completed_path, destination, *, stage="second", hours=24):
    """Bind a continuation to its predecessor's verified completed checkpoint."""
    plan, complete, summary, cfg = read_completion(plan_path, completed_path)
    require(summary["global_step"] < UPDATES * EPOCHS, "Training already complete.")
    require(stage in STAGES[1:], "Use second, third or fourth for continuation.")
    previous = STAGES[STAGES.index(stage) - 1]
    require(
        Path(complete["config"]).parent.name == previous,
        "Completion record does not identify the preceding segment.",
    )
    original = scientific(cfg)
    cfg.update(
        resume_from_checkpoint=complete["checkpoint"],
        resume_weights_only=False,
        reset_resume_time_budget=True,
    )
    cfg["output"]["checkpoint_path"] = str(
        Path(plan["target_root"]) / stage / "training-final.ckpt"
    )
    cfg["logging"]["wandb"]["resume"] = "must"
    set_budget(cfg, hours)
    cfg["trainer"]["callbacks"] = [
        c
        for c in cfg["trainer"]["callbacks"]
        if c["_target_"] != "rb_manual.ResumeGuard"
    ]
    cfg["trainer"]["callbacks"].append(
        {
            "_target_": "rb_manual.ResumeGuard",
            "record": str(Path(plan["target_root"]) / stage / "start-verified.json"),
            "expected_step": summary["global_step"],
            "checkpoint": complete["checkpoint"],
        }
    )
    require(
        scientific(cfg) == original,
        "Scientific settings changed during resume preparation.",
    )
    write_stage(
        Path(destination), plan, cfg, stage, hours, complete["checkpoint_sha256"]
    )
    return summary


def prepare_chain_stage(plan_path, stage):
    """Prepare a continuation at job start, or verify an already finished run."""
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan["target_root"])
    require(stage in STAGES[1:], "Invalid chain segment.")
    if (root / "completed.json").exists():
        _, _, summary, _ = read_completion(plan_path, root / "completed.json")
        require(summary["global_step"] == UPDATES * EPOCHS, "Run is not complete.")
        return {"already_complete": True, **summary}
    index = STAGES.index(stage)
    return prepare_resume(
        plan_path,
        root / STAGES[index - 1] / "completed.json",
        root / stage,
        stage=stage,
        hours=plan["allocation_hours"][index],
    )


def prepare_chain(source, destination, target_root, *, last_hours=12):
    """Write four bounded, afterok-linked allocations without submitting them."""
    require(last_hours in (12, 24), "Use a 12- or 24-hour final allocation.")
    plan = prepare_first(source, destination, target_root, first_hours=24, size="lola")
    destination, root = Path(destination), Path(target_root)
    plan["allocation_hours"] = [24, 24, 24, last_hours]
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    uv = ["uv", "run", "--project", plan["checkout"], "--frozen", "--no-sync", "python"]
    for stage, hours in zip(STAGES[1:], plan["allocation_hours"][1:], strict=True):
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=rb-lola-{stage}",
            "#SBATCH --nodes=8",
            "#SBATCH --gpus-per-node=4",
            "#SBATCH --ntasks-per-node=4",
            "#SBATCH --cpus-per-task=16",
            f"#SBATCH --time={hours:02d}:00:00",
            "#SBATCH --mem=0",
            "#SBATCH --no-requeue",
            f"#SBATCH --output={root}/continue-{stage}-%j.out",
            "set -euo pipefail",
            ': "${SLURM_JOB_ID:?Run inside a Slurm allocation}"',
            "cd " + shlex.quote(plan["checkout"]),
            "export PYTHONPATH="
            + shlex.quote(
                plan["checkout"] + "/src:" + plan["checkout"] + ":" + str(root)
            ),
            'export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
            "export RUNTIME_TYPECHECKING=false",
            "printf '%s  %s\\n' "
            + shlex.join([plan["helper_sha256"], str(root / "rb_manual.py")])
            + " | sha256sum -c -",
            shlex.join(
                [
                    *uv,
                    str(root / "rb_manual.py"),
                    "prepare-chain-stage",
                    "--plan",
                    str(root / "plan.json"),
                    "--stage",
                    stage,
                ]
            ),
            "if [ -f " + shlex.quote(str(root / "completed.json")) + " ]; then",
            "  echo 'Full endpoint already verified; no further training.'",
            "else",
            "  bash " + shlex.quote(str(root / stage / "train.sh")),
            "fi",
            "",
        ]
        path = destination / f"continue-{stage}.sh"
        path.write_text("\n".join(lines))
        path.chmod(0o755)
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'test "$#" -eq 2 || { echo "Usage: queue.sh ACCOUNT PARTITION" >&2; exit 2; }',
        "receipt=" + shlex.quote(str(root / "submitted-jobs.tsv")),
        'test ! -e "$receipt"',
        "mkdir " + shlex.quote(str(root / ".submission-lock")),
        "trap "
        + shlex.quote("rmdir " + shlex.quote(str(root / ".submission-lock")))
        + " EXIT",
        ': > "$receipt"',
        'previous=""',
    ]
    for stage in STAGES:
        script = root / (
            "first/train.sh" if stage == "first" else f"continue-{stage}.sh"
        )
        lines += [
            "options=(--parsable --kill-on-invalid-dep=yes "
            '"--account=$1" "--partition=$2")',
            'if [ -n "$previous" ]; then',
            '  options+=("--dependency=afterok:$previous")',
            "fi",
            'job=$(sbatch "${options[@]}" ' + shlex.quote(str(script)) + ")",
            'previous="${job%%;*}"',
            "printf '%s\\t%s\\n' " + shlex.quote(stage) + ' "$previous" >> "$receipt"',
            'echo "Submitted ' + stage + ': $previous"',
        ]
    path = destination / "queue.sh"
    path.write_text("\n".join([*lines, ""]))
    path.chmod(0o755)
    return plan


def main():
    """Prepare files, inspect a checkpoint or execute an allocated segment."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    first = sub.add_parser("prepare-first")
    first.add_argument("--source-json", type=Path, required=True)
    first.add_argument("--destination", type=Path, required=True)
    first.add_argument("--target-root", type=Path, required=True)
    first.add_argument("--first-hours", type=int, choices=(12, 14, 24), default=14)
    first.add_argument("--size", choices=("matched", "lola"), default="matched")
    chain = sub.add_parser("prepare-chain")
    chain.add_argument("--source-json", type=Path, required=True)
    chain.add_argument("--destination", type=Path, required=True)
    chain.add_argument("--target-root", type=Path, required=True)
    chain.add_argument("--last-hours", type=int, choices=(12, 24), default=12)
    next_stage = sub.add_parser("prepare-chain-stage")
    next_stage.add_argument("--plan", type=Path, required=True)
    next_stage.add_argument("--stage", choices=STAGES[1:], required=True)
    resume = sub.add_parser("prepare-resume")
    resume.add_argument("--plan", type=Path, required=True)
    resume.add_argument("--completed", type=Path, required=True)
    resume.add_argument("--destination", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    finish = sub.add_parser("seal")
    finish.add_argument("--config", type=Path, required=True)
    finish.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare-first":
        print(
            json.dumps(
                prepare_first(
                    json.loads(args.source_json.read_text()),
                    args.destination,
                    args.target_root,
                    first_hours=args.first_hours,
                    size=args.size,
                ),
                indent=2,
            )
        )
    elif args.action == "prepare-chain":
        print(
            json.dumps(
                prepare_chain(
                    json.loads(args.source_json.read_text()),
                    args.destination,
                    args.target_root,
                    last_hours=args.last_hours,
                ),
                indent=2,
            )
        )
    elif args.action == "prepare-chain-stage":
        print(json.dumps(prepare_chain_stage(args.plan, args.stage), indent=2))
    elif args.action == "prepare-resume":
        print(
            json.dumps(
                prepare_resume(args.plan, args.completed, args.destination), indent=2
            )
        )
    elif args.action == "seal":
        seal(args.config, args.stage_dir)
    else:
        require(os.environ.get("SLURM_JOB_ID"), "Run inside a Slurm allocation.")
        logging.basicConfig(level=logging.INFO)
        pin_local_cuda_device()
        cfg = OmegaConf.load(args.config)
        require(isinstance(cfg, DictConfig), "Expected a mapping configuration.")
        args.work_dir.mkdir(parents=True, exist_ok=True)
        run_epd_training(cast(DictConfig, cfg), work_dir=args.work_dir)


if __name__ == "__main__":
    main()
