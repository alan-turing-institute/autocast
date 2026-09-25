"""Resume the GS U-Net ablation after checking its full restored state.

Launch this module with four Slurm tasks. The checkpoint's directory must
also contain the backed-up original resolved_config.yaml. --verify-only
checks restoration without optimizer updates or writes to the original run.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, Timer
from omegaconf import DictConfig, OmegaConf, open_dict

from autocast.callbacks.ema import EMACallback
from autocast.callbacks.metrics import ValidationMetricPlotCallback
from autocast.scripts.train.encoder_processor_decoder import run_epd_training
from autocast.scripts.training import TrainingTimerCallback

RUN_ID = "k1d3p5ge"


def _same(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


class VerifyGSResume(Callback):
    """Fail on every rank if restored GS state differs from the source."""

    def __init__(self, checkpoint: str, record_dir: str, verify_only: bool):
        self.checkpoint = Path(checkpoint)
        self.record_dir = Path(record_dir)
        self.verify_only = verify_only

    def _verify(self, trainer, model) -> dict:
        source = torch.load(
            self.checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
        assert trainer.world_size == 4
        assert trainer.max_epochs == 541
        assert trainer.global_step == source["global_step"] == 182070
        assert _same(model.state_dict(), source["state_dict"])
        assert len(trainer.optimizers) == len(source["optimizer_states"]) == 1
        assert _same(trainer.optimizers[0].state_dict(), source["optimizer_states"][0])
        assert _same(
            trainer.lr_scheduler_configs[0].scheduler.state_dict(),
            source["lr_schedulers"][0],
        )
        checked = 0
        best_paths = {}
        for callback in trainer.callbacks:
            if not isinstance(
                callback,
                (ModelCheckpoint, EMACallback, ValidationMetricPlotCallback),
            ):
                continue
            expected = source["callbacks"][callback.state_key]
            actual = callback.state_dict()
            if isinstance(callback, ModelCheckpoint):
                # Lightning recomputes the transient score on new validation.
                expected = {k: v for k, v in expected.items() if k != "current_score"}
                actual = {k: v for k, v in actual.items() if k != "current_score"}
                if callback.best_model_path:
                    assert Path(callback.best_model_path).is_file()
                    best_paths[callback.state_key] = callback.best_model_path
            assert _same(actual, expected), callback.state_key
            checked += 1
        assert checked == 11
        reporter = next(
            c for c in trainer.callbacks if isinstance(c, TrainingTimerCallback)
        )
        old_reporter = source["callbacks"]["TrainingTimerCallback"]
        current = reporter.state_dict()
        assert current["epoch_times_s"] == old_reporter["epoch_times_s"]
        old_elapsed = old_reporter["training_runtime_elapsed_s"]
        assert old_elapsed <= current["training_runtime_elapsed_s"] < old_elapsed + 120
        timer = next(c for c in trainer.callbacks if isinstance(c, Timer))
        old_timer = source["callbacks"]["Timer"]["time_elapsed"]["train"]
        assert old_timer <= timer.time_elapsed() < old_timer + 120
        remaining = timer.time_remaining()
        assert remaining is not None
        assert 41000 < remaining < 42000
        return {
            "status": "passed",
            "ranks": trainer.world_size,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "optimizer_parameter_states": len(source["optimizer_states"][0]["state"]),
            "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
            "scheduler_epoch": source["lr_schedulers"][0]["last_epoch"],
            "callbacks_verified": checked,
            "elapsed_training_s": current["training_runtime_elapsed_s"],
            "remaining_training_s": remaining,
            "best_checkpoint_paths": best_paths,
        }

    def on_train_start(self, trainer, pl_module):
        error = None
        result = None
        try:
            result = self._verify(trainer, pl_module)
        except Exception as exc:  # Report a failed rank to every participant.
            error = f"rank {trainer.global_rank}: {type(exc).__name__}: {exc}"
        errors = [None] * trainer.world_size
        torch.distributed.all_gather_object(errors, error)
        if any(errors):
            message = f"GS resume verification failed: {errors}"
            raise RuntimeError(message)
        if trainer.is_global_zero:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            name = (
                "preflight_verified.json"
                if self.verify_only
                else "resume_verified.json"
            )
            (self.record_dir / name).write_text(json.dumps(result, indent=2) + "\n")
            print("GS_RESUME_VERIFIED " + json.dumps(result), flush=True)
        trainer.strategy.barrier("gs-resume-verified")
        if self.verify_only:
            print("GS_PREFLIGHT_COMPLETE: no optimizer updates", flush=True)
            raise SystemExit(0)


def main() -> None:
    """Load the saved scientific config and resume the original logical run."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-run-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    original = args.original_run_dir.resolve(strict=True)
    record_dir = checkpoint.parent
    cfg = OmegaConf.load(record_dir / "resolved_config.yaml")
    assert isinstance(cfg, DictConfig)
    assert cfg.logging.wandb.name == "unet_m8_crps_gs"
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 541
    assert cfg.trainer.devices == 4
    assert cfg.trainer.num_nodes == 1
    assert cfg.model.n_members == 8
    assert cfg.datamodule.batch_size == 32
    assert cfg.trainer.max_time == "00:23:59:00"
    assert cfg.output.skip_test
    directory = original / "autocast" / RUN_ID / "checkpoints"
    assert directory.is_dir()
    with open_dict(cfg):
        cfg.resume_from_checkpoint = str(checkpoint)
        cfg.resume_weights_only = False
        cfg.reset_resume_time_budget = False
        cfg.logging.wandb.id = RUN_ID
        cfg.logging.wandb.resume = "must"
        cfg.logging.wandb.entity = "turing-core"
        for callback in cfg.trainer.callbacks:
            if str(callback.get("_target_", "")).endswith("ModelCheckpoint"):
                callback.dirpath = str(directory)
        cfg.trainer.callbacks.append(
            {
                "_target_": "slurm_scripts.ablations.resume_unet_gs.VerifyGSResume",
                "checkpoint": str(checkpoint),
                "record_dir": str(record_dir),
                "verify_only": args.verify_only,
            }
        )
        if args.verify_only:
            cfg.logging.wandb.enabled = False
            cfg.trainer.num_sanity_val_steps = 0
    work_dir = record_dir / "preflight" if args.verify_only else original
    work_dir.mkdir(parents=True, exist_ok=True)
    run_epd_training(cfg, work_dir=work_dir)


if __name__ == "__main__":
    main()
