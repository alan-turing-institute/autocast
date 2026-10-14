"""CPU-only full-state resume checks with a changed loader length."""

import copy
import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning as L
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader, TensorDataset

from autocast.callbacks.ema import EMACallback

ROOT = Path(__file__).resolve().parents[2]
HELPERS = runpy.run_path(str(ROOT / "scripts/swe64s/ae_diversity.py"))
historical_lr = HELPERS["historical_lr"]
normalize_checkpoint = HELPERS["normalize_checkpoint"]
HistoricalAELearningRate = HELPERS["HistoricalAELearningRate"]


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


def test_exact_historical_lr_sequence():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 0.5 * (1 + math.cos(math.pi * epoch / 512))
    )
    for epoch in range(78):
        rate = optimizer.param_groups[0]["lr"]
        for step in range(epoch * 1004, (epoch + 1) * 1004):
            assert historical_lr(step) == rate
        optimizer.step()
        scheduler.step()
    assert historical_lr(29116) == 9.921050461934646e-6
    assert historical_lr(30119) == historical_lr(29116)
    assert historical_lr(30120) != historical_lr(30119)


class TinyAE(L.LightningModule):
    def __init__(self, *, scheduler):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.scheduler = scheduler
        self.validation_steps = []
        self.optimizer_before_resume = None
        self.ema_before_resume = None
        self.rates = []
        self.batch_indices = []

    def training_step(self, batch, batch_idx):
        self.batch_indices.append(batch_idx)
        return (self.layer(batch[0]) - batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self.validation_steps.append(self.global_step)
        self.log("val_loss", (self.layer(batch[0]) - batch[0]).square().mean())

    def on_train_start(self):
        trainer = cast(Trainer, self.trainer)
        self.optimizer_before_resume = copy.deepcopy(trainer.optimizers[0].state_dict())
        callbacks = getattr(trainer, "callbacks", [])
        ema = next(cb for cb in callbacks if isinstance(cb, EMACallback))
        self.ema_before_resume = copy.deepcopy(ema.state_dict())

    def on_before_optimizer_step(self, optimizer):
        self.rates.append((self.global_step, optimizer.param_groups[0]["lr"]))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        if not self.scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 0.5 * (1 + math.cos(math.pi * epoch / 8))
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def trainer(tmp_path, callbacks, **kwargs):
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
        callbacks=callbacks,
        **kwargs,
    )


def test_thinned_shape_probe_preserves_initialization_rng():
    # setup_datamodule consumes one shuffled probe batch before constructing
    # the model. Different dataset lengths must not change the global RNG
    # state at model initialization; RandomSampler uses a private generator.
    states, models = [], []
    for examples in (64200, 64000):
        torch.manual_seed(42)
        loader = DataLoader(
            TensorDataset(torch.zeros(examples, 1)),
            batch_size=16,
            shuffle=True,
            num_workers=0,
        )
        next(iter(loader))
        states.append(torch.get_rng_state())
        models.append(TinyAE(scheduler=False).state_dict())
    assert_equal(states[0], states[1])
    assert_equal(models[0], models[1])


def test_fresh_fit_starts_empty_matches_lr_and_validates_endpoint(tmp_path):
    torch.manual_seed(42)
    reference_initial = TinyAE(scheduler=True)
    torch.manual_seed(42)
    fresh = TinyAE(scheduler=False)
    assert_equal(fresh.state_dict(), reference_initial.state_dict())
    control = HistoricalAELearningRate(
        original_steps_per_epoch=3,
        cosine_epochs=8,
        base_lr=0.01,
        expected_resume_step=0,
        stop_step=11,
        expected_world_size=1,
        expected_batch_size=1,
    )
    ema = EMACallback(decay=0.9)
    fit = trainer(
        tmp_path,
        [control, ema],
        max_epochs=8,
        max_steps=11,
        enable_checkpointing=False,
        check_val_every_n_epoch=None,
        val_check_interval=3,
    )
    loader = DataLoader(TensorDataset(torch.arange(8.0).reshape(4, 2)), batch_size=1)
    validation = DataLoader(TensorDataset(torch.ones(1, 2)), batch_size=1)
    fit.fit(fresh, loader, validation)
    assert fit.global_step == 11
    assert fresh.optimizer_before_resume is not None
    assert fresh.ema_before_resume is not None
    assert fresh.optimizer_before_resume["state"] == {}
    assert int(fresh.ema_before_resume["ema_model_state"]["n_averaged"]) == 0
    assert fresh.batch_indices == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
    assert fresh.validation_steps == [3, 6, 9, 11]
    assert ema.ema_model is not None
    assert int(ema.ema_model.n_averaged) == 11
    assert [step for step, _ in fresh.rates] == list(range(11))
    for step, rate in fresh.rates:
        assert rate == historical_lr(
            step, original_steps_per_epoch=3, cosine_epochs=8, base_lr=0.01
        )


def test_full_resume_preserves_state_and_validates_partial_endpoint(tmp_path):
    torch.manual_seed(42)
    old_loader = DataLoader(
        TensorDataset(torch.arange(6.0).reshape(3, 2)), batch_size=1
    )
    validation = DataLoader(TensorDataset(torch.ones(1, 2)), batch_size=1)
    saved = ModelCheckpoint(dirpath=tmp_path / "original", save_last=True)
    original = TinyAE(scheduler=True)
    first = trainer(tmp_path, [EMACallback(decay=0.9), saved], max_epochs=2)
    first.fit(original, old_loader, validation)
    source = torch.load(saved.last_model_path, weights_only=False)
    assert source["global_step"] == 6
    before = copy.deepcopy(source)
    audit = normalize_checkpoint(
        source,
        original_steps_per_epoch=3,
        cosine_epochs=8,
        base_lr=0.01,
        expected_step=6,
    )
    for key in ("state_dict", "ema_state_dict", "optimizer_states"):
        assert_equal(source[key], before[key])
    assert_equal(source["callbacks"]["EMACallback"], before["callbacks"]["EMACallback"])
    assert source["lr_schedulers"] == []
    assert source["epoch"] == 2
    assert audit["completed_epochs"] == 2
    assert (
        source["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]["completed"]
        == 0
    )
    derived = tmp_path / "derived.ckpt"
    torch.save(source, derived)
    # Four new batches per epoch, versus three originally. Five updates finish
    # partway through the second new epoch, so the endpoint needs forced eval.
    new_loader = DataLoader(
        TensorDataset(torch.arange(8.0).reshape(4, 2)), batch_size=1
    )
    control = HistoricalAELearningRate(
        original_steps_per_epoch=3,
        cosine_epochs=8,
        base_lr=0.01,
        expected_resume_step=6,
        stop_step=11,
        expected_world_size=1,
        expected_batch_size=1,
    )
    resumed_ema = EMACallback(decay=0.9)
    resumed = TinyAE(scheduler=False)
    second = trainer(
        tmp_path,
        [control, resumed_ema],
        max_epochs=8,
        max_steps=11,
        enable_checkpointing=False,
        check_val_every_n_epoch=None,
        val_check_interval=3,
    )
    second.fit(resumed, new_loader, validation, ckpt_path=derived)
    assert second.global_step == 11
    assert resumed.batch_indices == [0, 1, 2, 3, 0]
    assert resumed.validation_steps == [9, 11]
    assert_equal(resumed.optimizer_before_resume, before["optimizer_states"][0])
    assert_equal(resumed.ema_before_resume, before["callbacks"]["EMACallback"])
    assert resumed_ema.ema_model is not None
    assert int(resumed_ema.ema_model.n_averaged) == 11
    assert [step for step, _ in resumed.rates] == list(range(6, 11))
    for step, rate in resumed.rates:
        assert rate == historical_lr(
            step, original_steps_per_epoch=3, cosine_epochs=8, base_lr=0.01
        )
    assert second.optimizers[0].param_groups[0]["lr"] == historical_lr(
        11, original_steps_per_epoch=3, cosine_epochs=8, base_lr=0.01
    )
    final = tmp_path / "final.ckpt"
    second.save_checkpoint(final)
    state = torch.load(final, weights_only=False)
    callback_state = state["callbacks"][control.state_key]
    another = HistoricalAELearningRate(
        original_steps_per_epoch=3,
        cosine_epochs=8,
        base_lr=0.01,
        expected_resume_step=6,
        stop_step=13,
        expected_world_size=1,
        expected_batch_size=1,
    )
    another.load_state_dict(callback_state)
    assert another.last_completed_step == 11


def psgd_checkpoint():
    current = {"ready": 1004, "completed": 1004}
    total = {"ready": 29116, "completed": 29116}
    batches = {
        "total": {**total, "started": 29116, "processed": 29116},
        "current": {**current, "started": 1004, "processed": 1004},
        "is_last_batch": True,
    }
    epochs = {"ready": 29, "started": 29, "processed": 29, "completed": 28}
    lr = historical_lr(29116)
    checkpoint = {
        "epoch": 28,
        "global_step": 29116,
        "state_dict": {"weight": torch.arange(4.0)},
        "ema_state_dict": {"weight": torch.arange(4.0) / 2},
        "optimizer_states": [
            {
                "state": {
                    0: {
                        "step": torch.tensor(29116),
                        "momentum": torch.arange(4.0),
                        "Q": [torch.eye(2)],
                    }
                },
                "param_groups": [
                    {
                        "params": [0],
                        "lr": lr,
                        "initial_lr": 1e-5,
                        "base_lr": 9.926388211944707e-6,
                        "prev_lr": 9.926388211944707e-6,
                    }
                ],
            }
        ],
        "lr_schedulers": [
            {"last_epoch": 29, "_step_count": 30, "base_lrs": [1e-5], "_last_lr": [lr]}
        ],
        "callbacks": {
            "EMACallback": {
                "ema_model_state": {
                    "n_averaged": torch.tensor(29116),
                    "module.weight": torch.arange(4.0) / 2,
                }
            },
            "Timer": {"time_elapsed": {"train": 500}},
            "ModelCheckpoint": {"best_model_path": "/old.ckpt"},
        },
        "loops": {
            "fit_loop": {
                "state_dict": {"combined_loader": ["old state"]},
                "epoch_progress": {
                    "total": copy.deepcopy(epochs),
                    "current": copy.deepcopy(epochs),
                },
                "epoch_loop.batch_progress": batches,
                "epoch_loop.scheduler_progress": {
                    "total": {"ready": 29, "completed": 29},
                    "current": {"ready": 1, "completed": 1},
                },
                "epoch_loop.automatic_optimization.optim_progress": {
                    "optimizer": {
                        "step": {"total": total, "current": current},
                        "zero_grad": copy.deepcopy(batches),
                    }
                },
                "epoch_loop.state_dict": {"_batches_that_stepped": 29116},
                "epoch_loop.val_loop.batch_progress": copy.deepcopy(batches),
            }
        },
    }
    return checkpoint


def test_psgd_optimizer_tensors_are_preserved():
    checkpoint = psgd_checkpoint()
    original = copy.deepcopy(checkpoint)
    normalize_checkpoint(checkpoint)
    for key in ("state_dict", "ema_state_dict", "optimizer_states"):
        assert_equal(checkpoint[key], original[key])
    assert_equal(
        checkpoint["callbacks"]["EMACallback"], original["callbacks"]["EMACallback"]
    )
    assert list(checkpoint["callbacks"]) == ["EMACallback"]
    assert (
        checkpoint["loops"]["fit_loop"][
            "epoch_loop.automatic_optimization.optim_progress"
        ]["optimizer"]["step"]["total"]["completed"]
        == 29116
    )
    with pytest.raises(ValueError, match="full-state"):
        normalize_checkpoint(checkpoint)


def test_callback_rejects_wrong_global_step():
    callback = HistoricalAELearningRate()
    with pytest.raises(ValueError, match="Expected restored global step"):
        callback.on_train_start(SimpleNamespace(global_step=0), None)


@pytest.mark.parametrize("destination", ["checkpoint", "manifest"])
def test_prepare_rejects_dangling_destination_links(tmp_path, destination):
    target = tmp_path / "derived.ckpt"
    link = (
        target
        if destination == "checkpoint"
        else target.with_suffix(".provenance.json")
    )
    redirected = tmp_path / "unintended.ckpt"
    link.symlink_to(redirected)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        HELPERS["prepare_checkpoint"](
            tmp_path / "source.ckpt", target, tmp_path / "config.yaml"
        )
    assert link.is_symlink()
    assert not redirected.exists()


def test_prepare_checkpoint_writes_manifest_and_preserves_source(tmp_path):
    source, target = tmp_path / "original.ckpt", tmp_path / "derived.ckpt"
    config = tmp_path / "resolved.yaml"
    config.write_text(
        "optimizer:\n  optimizer: psgd\n  scheduler: cosine\n"
        "  scheduler_interval: epoch\n  cosine_epochs: 512\n"
        "  learning_rate: 0.00001\n  warmup: 0\n  min_lr_ratio: 0.0\n"
    )
    original = psgd_checkpoint()
    torch.save(original, source)
    source_hash = HELPERS["sha256"](source)
    record = HELPERS["prepare_checkpoint"](
        source, target, config, expected_sha256=source_hash
    )
    assert HELPERS["sha256"](source) == source_hash
    manifest = json.loads(target.with_suffix(".provenance.json").read_text())
    assert manifest == record
    assert record["derived_checkpoint_sha256"] == HELPERS["sha256"](target)
    assert record["audit"]["global_step"] == 29116
    derived = torch.load(target, weights_only=False)
    for key in ("state_dict", "ema_state_dict", "optimizer_states"):
        assert_equal(derived[key], original[key])
    assert_equal(torch.load(source, weights_only=False), original)
