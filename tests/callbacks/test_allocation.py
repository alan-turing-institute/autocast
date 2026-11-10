"""Check collective epoch-boundary stopping without changing the fit horizon."""

from types import SimpleNamespace
from typing import cast

import lightning as L
import pytest

from autocast.callbacks.allocation import AllocationDeadline

UNUSED_TRAINER = cast(L.Trainer, None)
UNUSED_MODEL = cast(L.LightningModule, None)


def test_allocation_stop_is_synchronized_at_epoch_boundary(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("autocast.callbacks.allocation.time.time", lambda: clock[0])
    monkeypatch.setenv("AUTOCAST_ALLOCATION_END_TIME", "2000")
    decisions = []

    def broadcast(value):
        decisions.append(value)
        return value

    trainer = SimpleNamespace(
        should_stop=False, strategy=SimpleNamespace(broadcast=broadcast), max_epochs=798
    )
    callback = AllocationDeadline(reserve_seconds=100)
    callback.on_fit_start(cast(L.Trainer, trainer), UNUSED_MODEL)
    callback.on_train_epoch_end(cast(L.Trainer, trainer), UNUSED_MODEL)
    assert not trainer.should_stop
    clock[0] = 1900
    callback.on_train_epoch_end(cast(L.Trainer, trainer), UNUSED_MODEL)
    assert trainer.should_stop
    assert decisions == [False, True]
    assert trainer.max_epochs == 798
    assert callback.state_dict() == {}


def test_resumed_allocation_reads_a_new_deadline(monkeypatch):
    monkeypatch.setattr("autocast.callbacks.allocation.time.time", lambda: 1000.0)
    callback = AllocationDeadline(100)
    monkeypatch.setenv("AUTOCAST_ALLOCATION_END_TIME", "2000")
    callback.on_fit_start(UNUSED_TRAINER, UNUSED_MODEL)
    assert callback.deadline == 2000
    monkeypatch.setenv("AUTOCAST_ALLOCATION_END_TIME", "3000")
    callback.on_fit_start(UNUSED_TRAINER, UNUSED_MODEL)
    assert callback.deadline == 3000


def test_invalid_allocation_deadlines_fail_before_training(monkeypatch):
    monkeypatch.delenv("AUTOCAST_ALLOCATION_END_TIME", raising=False)
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    with pytest.raises(ValueError, match="ALLOCATION_END_TIME"):
        AllocationDeadline().on_fit_start(UNUSED_TRAINER, UNUSED_MODEL)
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1")
    with pytest.raises(RuntimeError, match="Insufficient"):
        AllocationDeadline().on_fit_start(UNUSED_TRAINER, UNUSED_MODEL)
