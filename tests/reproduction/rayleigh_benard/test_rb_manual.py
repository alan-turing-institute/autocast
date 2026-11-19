"""CPU-only checks for checkpoint identity and manual segment preparation."""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from omegaconf import OmegaConf
from rb_reproduction import load_rb_tool
from torch.utils.data import DataLoader, TensorDataset

from autocast.callbacks.ema import EMACallback
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.scripts.config import save_resolved_config
from autocast.scripts.training import (
    TrainingTimerCallback,
    _attach_reset_timer_callback,
)

ops = load_rb_tool("rb_manual")


@pytest.fixture(scope="module")
def checkpoint_file(tmp_path_factory):
    """Build a native full checkpoint without external files or datasets."""
    folder = tmp_path_factory.mktemp("rb_manual_source")
    loader = DataLoader(TensorDataset(torch.arange(1.0, 6.0)), batch_size=1)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=4096,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=folder,
        callbacks=[EMACallback(decay=0.9), StopAtEpoch(2)],
    )
    trainer.fit(TinyModel(), loader)
    path = folder / "resume.ckpt"
    trainer.save_checkpoint(path)
    return path


@pytest.fixture
def original(checkpoint_file):
    return torch.load(
        checkpoint_file, map_location="cpu", mmap=True, weights_only=False
    )


@pytest.fixture
def payload(original):
    c = copy.deepcopy(original)
    c["global_step"] = 128
    fit = c["loops"]["fit_loop"]
    fit["epoch_loop.state_dict"]["_batches_that_stepped"] = 128
    for name, value in (("total", 128), ("current", 64)):
        fit["epoch_loop.batch_progress"][name] = dict.fromkeys(
            fit["epoch_loop.batch_progress"][name], value
        )
    for state in c["optimizer_states"][0]["state"].values():
        state["step"] = torch.tensor(128.0)
    c["callbacks"]["EMACallback"]["ema_model_state"]["n_averaged"] = torch.tensor(128)
    return c


def test_native_completed_checkpoint_is_accepted(checkpoint_file):
    result = ops.checkpoint_summary(checkpoint_file, updates=5)
    assert result["completed_epochs"] == 2
    assert result["global_step"] == 10
    assert result["full_state"]
    assert result["epoch_boundary"]


@pytest.mark.parametrize(
    "damage",
    [
        "weights_only",
        "partial_epoch",
        "wrong_step",
        "wrong_lr",
        "missing_scheduler",
        "missing_ema",
        "adam_step",
        "ema_step",
    ],
)
def test_rejects_incomplete_or_inconsistent_checkpoint(tmp_path, payload, damage):
    if damage == "weights_only":
        payload.pop("optimizer_states")
    elif damage == "partial_epoch":
        payload["loops"]["fit_loop"]["epoch_loop.batch_progress"]["is_last_batch"] = (
            False
        )
    elif damage == "wrong_step":
        payload["global_step"] += 1
    elif damage == "wrong_lr":
        payload["optimizer_states"][0]["param_groups"][0]["lr"] = 1e-4
    elif damage == "missing_scheduler":
        payload.pop("lr_schedulers")
    elif damage == "missing_ema":
        payload.pop("ema_state_dict")
    elif damage == "adam_step":
        next(iter(payload["optimizer_states"][0]["state"].values()))["step"] = (
            torch.tensor(1.0)
        )
    else:
        payload["callbacks"]["EMACallback"]["ema_model_state"]["n_averaged"] = (
            torch.tensor(1)
        )
    path = tmp_path / "bad.ckpt"
    torch.save(payload, path)
    with pytest.raises((ValueError, KeyError)):
        ops.checkpoint_summary(path)


def test_rejects_mutable_alias(tmp_path, checkpoint_file):
    path = tmp_path / "alias.ckpt"
    path.symlink_to(checkpoint_file)
    with pytest.raises(ValueError, match="concrete"):
        ops.checkpoint_summary(path)


@pytest.mark.parametrize("first_hours", [12, 14])
def test_two_segments_preserve_scientific_settings_and_identity(
    tmp_path, payload, REPO_ROOT, first_hours
):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                "local_experiment=the_well/rayleigh_benard/crps_spatiotemporal_ambient_matched",
                f"hydra.searchpath=[file://{REPO_ROOT}/local_hydra]",
            ],
        )
    cfg.trainer.num_nodes = 8
    cfg.datamodule.batch_size = 1
    cfg.trainer.max_time = None
    cfg.logging.wandb.id = "fixture-run"
    cfg.trainer.callbacks.append(
        {
            "_target_": "lightning.pytorch.callbacks.Timer",
            "duration": "00:23:30:00",
            "interval": "epoch",
        }
    )
    source: dict[str, Any] = {
        "head": ops.COMMIT,
        "git_status": "",
        "checkout": str(REPO_ROOT),
        "source_config_sha256": "fixture",
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    target = tmp_path / "plan"
    plan = ops.prepare_first(source, target, target, first_hours=first_hours)
    first = cast(
        dict[str, Any],
        OmegaConf.to_container(
            OmegaConf.load(target / "first/train.yaml"), resolve=True
        ),
    )
    timers = [
        cb for cb in first["trainer"]["callbacks"] if cb["_target_"].endswith(".Timer")
    ]
    assert timers[0]["duration"] == f"00:{first_hours - 1:02d}:30:00"
    assert plan["first_allocation_hours"] == first_hours
    assert (
        f"#SBATCH --time={first_hours:02d}:00:00"
        in (target / "first/train.sh").read_text()
    )
    assert first["trainer"]["max_epochs"] == first["optimizer"]["cosine_epochs"] == 4096
    assert first["datamodule"] == source["config"]["datamodule"]
    assert first["model"] == source["config"]["model"]
    checkpoint = Path(first["output"]["checkpoint_path"])
    torch.save(payload, checkpoint)
    completion = ops.seal(target / "first/train.yaml", target / "first")
    assert completion["global_step"] == 128
    assert not Path(completion["checkpoint"]).is_symlink()
    # Later alias changes must not alter the frozen continuation source.
    checkpoint.write_bytes(b"different later output")
    assert ops.digest(completion["checkpoint"]) == completion["checkpoint_sha256"]
    ops.prepare_resume(
        target / "plan.json", target / "first/completed.json", target / "second"
    )
    second = cast(
        dict[str, Any],
        OmegaConf.to_container(
            OmegaConf.load(target / "second/train.yaml"), resolve=True
        ),
    )
    assert ops.scientific(first) == ops.scientific(second)
    assert second["resume_weights_only"] is False
    assert second["reset_resume_time_budget"] is True
    assert second["logging"]["wandb"]["id"] == first["logging"]["wandb"]["id"]
    assert second["logging"]["wandb"]["resume"] == "must"
    assert second["resume_from_checkpoint"] == completion["checkpoint"]
    assert (
        next(
            cb
            for cb in second["trainer"]["callbacks"]
            if cb["_target_"].endswith(".Timer")
        )["duration"]
        == "00:23:30:00"
    )
    for stage in ("first", "second"):
        script = target / stage / "train.sh"
        if os.name == "posix":
            subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text()
        assert "--no-requeue" in text
        assert "--kill-on-bad-exit=1" in text
        assert "\nsbatch " not in text
    assert plan["primary_endpoint"] == {"global_step": 262144, "use_ema": False}
    # A changed source checkpoint must fail before another job can be prepared.
    with Path(completion["checkpoint"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        ops.prepare_resume(
            target / "plan.json", target / "first/completed.json", target / "third"
        )


@pytest.fixture
def lola_source(REPO_ROOT):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                "local_experiment=the_well/rayleigh_benard/crps_spatiotemporal_ambient",
                f"hydra.searchpath=[file://{REPO_ROOT}/local_hydra]",
            ],
        )
    cfg.trainer.num_nodes = 8
    cfg.datamodule.batch_size = 1
    cfg.trainer.max_time = None
    cfg.trainer.callbacks.append(
        {
            "_target_": "lightning.pytorch.callbacks.Timer",
            "duration": "00:23:30:00",
            "interval": "epoch",
        }
    )
    return {
        "head": ops.COMMIT,
        "git_status": "",
        "checkout": str(REPO_ROOT),
        "source_config_sha256": "fixture",
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def _complete_chain_segment(target, stage, original, completed):
    payload = copy.deepcopy(original)
    step = completed * 64
    payload["epoch"] = completed - 1
    payload["global_step"] = step
    progress = payload["loops"]["fit_loop"]["epoch_progress"]
    for key in ("current", "total"):
        progress[key] = dict.fromkeys(progress[key], completed)
    lr = 1e-4 * (1 + math.cos(math.pi * completed / 4096)) / 2
    for state in payload["optimizer_states"][0]["state"].values():
        state["step"] = torch.tensor(float(step))
    for group in payload["optimizer_states"][0]["param_groups"]:
        group["lr"] = lr
    payload["lr_schedulers"][0].update(last_epoch=completed, _last_lr=[lr])
    payload["callbacks"]["EMACallback"]["ema_model_state"]["n_averaged"] = torch.tensor(
        step
    )
    config = target / stage / "train.yaml"
    torch.save(payload, OmegaConf.load(config).output.checkpoint_path)
    return ops.seal(config, target / stage)


@pytest.mark.parametrize("finish_at", ["second", "fourth"])
def test_lola_chain_preserves_identity_and_skips_finished_training(
    tmp_path, lola_source, payload, finish_at
):
    target = tmp_path / "chain"
    plan = ops.prepare_chain(lola_source, target, target)
    assert plan["size"] == "lola"
    assert plan["allocation_hours"] == [24, 24, 24, 12]
    first = OmegaConf.to_container(
        OmegaConf.load(target / "first/train.yaml"), resolve=True
    )
    for index, stage in enumerate(ops.STAGES):
        if index:
            result = ops.prepare_chain_stage(target / "plan.json", stage)
            if index > ops.STAGES.index(finish_at):
                assert result["already_complete"]
                assert not (target / stage).exists()
                continue
        cfg = cast(
            dict[str, Any],
            OmegaConf.to_container(
                OmegaConf.load(target / stage / "train.yaml"), resolve=True
            ),
        )
        assert ops.scientific(cfg) == ops.scientific(first)
        assert cfg["trainer"]["max_epochs"] == cfg["optimizer"]["cosine_epochs"] == 4096
        assert cfg["model"]["processor"]["backbone"]["hid_channels"] == 2048
        completed = 4096 if stage == finish_at else (index + 1) * 2
        _complete_chain_segment(target, stage, payload, completed)
    assert json.loads((target / "completed.json").read_text())["global_step"] == 262144
    if os.name == "posix":
        for script in target.glob("*.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True)


def test_lola_chain_rejects_changed_science(tmp_path, lola_source, payload):
    target = tmp_path / "chain"
    ops.prepare_chain(lola_source, target, target)
    _complete_chain_segment(target, "first", payload, 2)
    ops.prepare_chain_stage(target / "plan.json", "second")
    config = target / "second/train.yaml"
    cfg = OmegaConf.load(config)
    cfg.optimizer.learning_rate = 2e-4
    OmegaConf.save(cfg, config)
    _complete_chain_segment(target, "second", payload, 4)
    with pytest.raises(ValueError, match="Scientific settings changed"):
        ops.prepare_chain_stage(target / "plan.json", "third")


@pytest.mark.skipif(os.name != "posix", reason="Slurm submission requires POSIX Bash")
@pytest.mark.parametrize("fail_at", [None, 2])
def test_chain_submission_records_dependencies_and_refuses_duplicates(
    tmp_path, lola_source, fail_at
):
    target = tmp_path / "chain"
    ops.prepare_chain(lola_source, target, target)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "sbatch.log"
    stub = binaries / "sbatch"
    stub.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SBATCH_TEST_LOG"\n'
        'count=$(wc -l < "$SBATCH_TEST_LOG")\n'
        'if [ "$count" -eq "$SBATCH_FAIL_AT" ]; then exit 1; fi\n'
        'echo "$((9000 + count))"\n'
    )
    stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        SBATCH_TEST_LOG=str(log),
        SBATCH_FAIL_AT=str(fail_at or 0),
    )
    command = ["bash", str(target / "queue.sh"), "test-account", "test-partition"]
    result = subprocess.run(command, env=env, check=False)
    assert result.returncode == (1 if fail_at else 0)
    calls = log.read_text().splitlines()
    assert len(calls) == (fail_at or 4)
    assert "--dependency" not in calls[0]
    for index, call in enumerate(calls[1:], start=1):
        assert f"--dependency=afterok:{9000 + index}" in call
        assert "--kill-on-invalid-dep=yes" in call
    assert (target / "submitted-jobs.tsv").read_text().splitlines() == [
        f"{stage}\t{9001 + index}"
        for index, stage in enumerate(ops.STAGES[: (fail_at - 1) if fail_at else 4])
    ]
    assert subprocess.run(command, env=env, check=False).returncode != 0
    assert len(log.read_text().splitlines()) == (fail_at or 4)


@pytest.fixture
def restored(original):
    model = L.LightningModule()
    model.register_parameter(
        "weight", torch.nn.Parameter(original["state_dict"]["weight"].clone())
    )
    ema = EMACallback(decay=0.9)
    ema.load_state_dict(original["callbacks"]["EMACallback"])
    optimizer = copy.deepcopy(original["optimizer_states"][0])
    scheduler = copy.deepcopy(original["lr_schedulers"][0])
    timer = Timer(duration="00:23:30:00", interval="epoch")
    trainer = SimpleNamespace(
        world_size=1,
        global_step=10,
        max_epochs=4096,
        optimizers=[
            SimpleNamespace(
                state_dict=lambda: optimizer, param_groups=optimizer["param_groups"]
            )
        ],
        lr_scheduler_configs=[
            SimpleNamespace(scheduler=SimpleNamespace(state_dict=lambda: scheduler))
        ],
        callbacks=[ema, timer],
    )
    ema.on_fit_start(cast(L.Trainer, trainer), model)
    return trainer, model, optimizer, scheduler, ema, timer


def test_restored_weights_optimizer_scheduler_and_ema_match(
    tmp_path, restored, checkpoint_file
):
    trainer, model, *_ = restored
    guard = ops.ResumeGuard(
        tmp_path / "verified.json",
        expected_step=10,
        checkpoint=str(checkpoint_file),
        world_size=1,
    )
    assert guard.verify(trainer, model)["full_state_resume"]


@pytest.mark.parametrize(
    "damage", ["weights", "optimizer", "scheduler", "ema", "timer", "step", "ranks"]
)
def test_guard_detects_wrong_restored_state(
    tmp_path, restored, damage, checkpoint_file
):
    trainer, model, optimizer, scheduler, ema, timer = restored
    if damage == "weights":
        model.weight.data.add_(1)
    elif damage == "optimizer":
        optimizer["param_groups"][0]["lr"] = 1e-4
    elif damage == "scheduler":
        scheduler["last_epoch"] = 0
    elif damage == "ema":
        ema.ema_model.module.weight.data.add_(1)
    elif damage == "timer":
        timer._offset = 100.0
    elif damage == "step":
        trainer.global_step = 0
    else:
        trainer.world_size = 2
    guard = ops.ResumeGuard(
        tmp_path / "verified.json",
        expected_step=10,
        checkpoint=str(checkpoint_file),
        world_size=1,
    )
    messages = {
        "weights": "Restored model",
        "optimizer": "Optimizer state",
        "scheduler": "Scheduler state",
        "ema": "Callback state",
        "timer": "Allocation timer",
        "step": "global step",
        "ranks": "world size",
    }
    with pytest.raises(ValueError, match=messages[damage]):
        guard.verify(trainer, model)


class TinyModel(OptimizerMixin, L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer_config = {
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "betas": [0.9, 0.99],
            "weight_decay": 0.0,
            "scheduler": "cosine",
            "scheduler_interval": "epoch",
            "cosine_epochs": 4096,
            "warmup": 0,
        }

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return ((self.weight * batch[0] - 0.25) ** 2).mean()

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self.log("val_loss", float(self.current_epoch + 1), batch_size=1)


class StopAtEpoch(L.Callback):
    def __init__(self, completed):
        self.completed = completed

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: ARG002
        if trainer.current_epoch + 1 == self.completed:
            trainer.should_stop = True


@pytest.mark.parametrize("segments", [2, 4])
def test_guard_runs_after_actual_lightning_full_state_restore(tmp_path, segments):
    checkpoints = tmp_path / "checkpoints"
    source = tmp_path / "first-final.ckpt"
    loader = DataLoader(TensorDataset(torch.arange(1.0, 65.0)), batch_size=1)

    def trainer_for(completed, resume):
        guard = ops.ResumeGuard(
            tmp_path / ("resumed.json" if resume else "initial.json"),
            expected_step=64 * (completed - 1),
            checkpoint=str(source) if resume else None,
            world_size=1,
        )
        callbacks = [
            ModelCheckpoint(
                dirpath=checkpoints,
                every_n_epochs=1,
                save_top_k=-1,
                filename="snapshot-{epoch}",
                save_last=True,
            ),
            ModelCheckpoint(
                dirpath=checkpoints,
                monitor="val_loss",
                save_top_k=1,
                filename="best-{epoch}",
            ),
            EMACallback(decay=0.999),
            Timer(duration="00:13:30:00", interval="epoch"),
            guard,
            TrainingTimerCallback(),
            StopAtEpoch(completed),
        ]
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=4096,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            default_root_dir=tmp_path,
        )
        _attach_reset_timer_callback(trainer, enabled=resume)
        return trainer

    first = trainer_for(1, False)
    first.fit(TinyModel(), loader, loader)
    for completed in range(2, segments + 1):
        next(
            cb for cb in getattr(first, "callbacks", []) if isinstance(cb, Timer)
        )._offset = 13.5 * 3600
        first.save_checkpoint(source)
        assert ops.checkpoint_summary(source)["global_step"] == (completed - 1) * 64
        resumed = trainer_for(completed, True)
        resumed.fit(TinyModel(), loader, loader, ckpt_path=source)
        assert resumed.global_step == completed * 64
        verified = json.loads((tmp_path / "resumed.json").read_text())
        assert verified["status"] == "passed"
        assert verified["global_step"] == (completed - 1) * 64
        assert verified["full_state_resume"]
        first = resumed


def test_run_entrypoint_creates_work_directory_before_training(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"experiment_name": "entrypoint-check"}), config)
    work = tmp_path / "new/run"
    written = []

    def fake_training(cfg, *, work_dir):
        written.append(save_resolved_config(cfg, work_dir))

    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setattr(ops, "pin_local_cuda_device", lambda: None)
    monkeypatch.setattr(ops, "run_epd_training", fake_training)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rb_manual", "run", "--config", str(config), "--work-dir", str(work)],
    )
    ops.main()
    assert written == [work / "resolved_config.yaml"]
    assert written[0].is_file()
