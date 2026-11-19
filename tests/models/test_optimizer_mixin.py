"""Unit tests for OptimizerMixin warmup resolution."""

from __future__ import annotations

import math
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.multiprocessing.spawn import ProcessRaisedException, spawn

from autocast.models.optimizer_mixin import (
    ALLOW_COSINE_HORIZON_CHANGE_ENV_VAR,
    PRODIGY_D0_KEY,
    HorizonAwareLambdaLR,
    OptimizerMixin,
    WarmupReduceLROnPlateau,
)


class _ToyMixinUser(OptimizerMixin):
    """Minimal nn.Module that exposes OptimizerMixin helpers for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.optimizer_config = None


@pytest.fixture
def mixin() -> _ToyMixinUser:
    return _ToyMixinUser()


class _ShapeLoss(nn.Module):
    """Stand-in loss owning a single global shape scalar (like raw_nu)."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_nu = nn.Parameter(torch.zeros(()))

    def shape_parameters(self) -> list[nn.Parameter]:
        return [self.raw_nu]


class _ShapeMixinUser(OptimizerMixin):
    """Mixin user with a bulk network plus a shape-parameter-owning submodule."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.loss_func = _ShapeLoss()
        self.optimizer_config = None


class TestResolveWarmup:
    """Tests for _resolve_warmup: fractional vs absolute warmup parsing."""

    def test_zero_returns_zero(self, mixin: _ToyMixinUser) -> None:
        assert mixin._resolve_warmup({"warmup": 0}, horizon=1000) == 0

    def test_fractional_within_unit_interval_uses_horizon(
        self, mixin: _ToyMixinUser
    ) -> None:
        # 0.05 * 1000 = 50
        assert mixin._resolve_warmup({"warmup": 0.05}, horizon=1000) == 50

    def test_fractional_rounds_down(self, mixin: _ToyMixinUser) -> None:
        # 0.05 * 137 = 6.85 -> int() -> 6
        assert mixin._resolve_warmup({"warmup": 0.05}, horizon=137) == 6

    def test_absolute_int_passes_through(self, mixin: _ToyMixinUser) -> None:
        assert mixin._resolve_warmup({"warmup": 500}, horizon=1000) == 500

    def test_float_equal_to_one_is_absolute(self, mixin: _ToyMixinUser) -> None:
        # 1.0 is excluded from the fractional range so int(1.0) = 1 step.
        assert mixin._resolve_warmup({"warmup": 1.0}, horizon=1000) == 1

    def test_float_above_one_is_absolute(self, mixin: _ToyMixinUser) -> None:
        # 100.7 -> int(100.7) = 100 absolute steps (not 100 * horizon).
        assert mixin._resolve_warmup({"warmup": 100.7}, horizon=1000) == 100

    def test_float_at_zero_is_zero(self, mixin: _ToyMixinUser) -> None:
        assert mixin._resolve_warmup({"warmup": 0.0}, horizon=1000) == 0

    def test_negative_clamps_to_zero(self, mixin: _ToyMixinUser) -> None:
        assert mixin._resolve_warmup({"warmup": -5}, horizon=1000) == 0
        assert mixin._resolve_warmup({"warmup": -0.5}, horizon=1000) == 0

    def test_missing_key_defaults_to_zero(self, mixin: _ToyMixinUser) -> None:
        assert mixin._resolve_warmup({}, horizon=1000) == 0

    def test_nan_warmup_raises(self, mixin: _ToyMixinUser) -> None:
        with pytest.raises(ValueError, match="warmup must be finite"):
            mixin._resolve_warmup({"warmup": float("nan")}, horizon=1000)

    def test_positive_inf_warmup_raises(self, mixin: _ToyMixinUser) -> None:
        with pytest.raises(ValueError, match="warmup must be finite"):
            mixin._resolve_warmup({"warmup": float("inf")}, horizon=1000)

    def test_negative_inf_warmup_raises(self, mixin: _ToyMixinUser) -> None:
        with pytest.raises(ValueError, match="warmup must be finite"):
            mixin._resolve_warmup({"warmup": float("-inf")}, horizon=1000)


class TestGradClipGuard:
    """_create_optimizer rejects the removed optimizer.grad_clip field."""

    def test_grad_clip_value_raises(self, mixin: _ToyMixinUser) -> None:
        # A real value would be silently ignored (clipping is now wired through
        # trainer.gradient_clip_val), so fail loud and tell the caller to migrate.
        cfg = cast(
            "Any",
            {"optimizer": "adamw", "learning_rate": 1e-4, "grad_clip": 2.0},
        )
        with pytest.raises(ValueError, match="grad_clip"):
            mixin._create_optimizer(cfg)

    def test_grad_clip_null_is_allowed(self, mixin: _ToyMixinUser) -> None:
        # null meant "no clip"; it is benign, so it must not raise.
        cfg = cast(
            "Any",
            {"optimizer": "adamw", "learning_rate": 1e-4, "grad_clip": None},
        )
        assert isinstance(mixin._create_optimizer(cfg), torch.optim.AdamW)

    def test_absent_grad_clip_is_allowed(self, mixin: _ToyMixinUser) -> None:
        cfg = cast("Any", {"optimizer": "adamw", "learning_rate": 1e-4})
        assert isinstance(mixin._create_optimizer(cfg), torch.optim.AdamW)


class TestCosineSchedulerWithWarmup:
    """Integration: cosine scheduler uses the resolved warmup correctly."""

    def _build_scheduler(
        self, *, warmup: float | int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        m = _ToyMixinUser()
        m.optimizer_config = cast(
            "Any",
            {
                "optimizer": "adamw",
                "learning_rate": 1.0,  # get_last_lr surfaces the lambda value
                "weight_decay": 0.0,
                "warmup": warmup,
                "scheduler": "cosine",
                "scheduler_interval": "step",
                "cosine_steps": 1000,
            },
        )
        result = m.configure_optimizers()
        assert isinstance(result, dict)
        result_any = cast("dict[str, Any]", result)
        sched = result_any["lr_scheduler"]["scheduler"]
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)
        return sched

    def test_zero_warmup_starts_at_one(self) -> None:
        # cosine(0) = (1 + cos(0))/2 = 1.0; no warmup factor.
        sched = self._build_scheduler(warmup=0)
        assert sched.get_last_lr()[0] == pytest.approx(1.0)

    def test_fractional_warmup_50_steps_starts_at_1_over_51(self) -> None:
        # warmup=0.05 with horizon=1000 -> 50 warmup steps.
        # At step 0: warm = min(1, 1/51), cosine = 1.0 -> lr = 1/51.
        sched = self._build_scheduler(warmup=0.05)
        assert sched.get_last_lr()[0] == pytest.approx(1.0 / 51.0, rel=1e-4)

    def test_absolute_warmup_matches_fractional_equivalent(self) -> None:
        # warmup=50 (absolute) and warmup=0.05 (50/1000) yield identical schedules.
        sched_abs = self._build_scheduler(warmup=50)
        sched_frac = self._build_scheduler(warmup=0.05)
        for _ in range(75):
            sched_abs.optimizer.step()
            sched_abs.step()
            sched_frac.optimizer.step()
            sched_frac.step()
        assert sched_abs.get_last_lr()[0] == pytest.approx(
            sched_frac.get_last_lr()[0], rel=1e-6
        )

    def test_lr_climbs_through_warmup_and_peaks_after(self) -> None:
        # warmup=50, horizon=1000. lr should grow ~linearly through step 50 then
        # follow cosine decay (still ~1.0 at step 50, before significant decay).
        sched = self._build_scheduler(warmup=50)
        lrs = [sched.get_last_lr()[0]]
        for _ in range(60):
            sched.optimizer.step()
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        # Step 0: ~1/51. Step 50: ~51/51 * cos(50/1000 * pi) ~ 0.997. Strict climb.
        assert lrs[0] < lrs[25] < lrs[50]
        assert lrs[50] == pytest.approx(0.997, rel=1e-2)


class _FakeTrainer:
    """Minimal stand-in exposing the callbacks the time-cosine path reads."""

    def __init__(self, timer: Timer | None):
        self.callbacks = [] if timer is None else [timer]


def _check_distributed_time_schedule(rank: int, rendezvous: str) -> None:
    """Verify real broadcasts keep LR and updates equal despite clock skew."""
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(7)
        model = _ToyMixinUser()
        timer = Timer(duration="00:01:00:00")
        trainer = _FakeTrainer(timer)
        trainer.strategy = DDPStrategy(  # type: ignore[attr-defined]
            parallel_devices=[torch.device("cpu"), torch.device("cpu")]
        )
        model.trainer = trainer  # type: ignore[attr-defined]
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = model._create_scheduler(
            optimizer, {"scheduler": "cosine", "scheduler_interval": "time"}
        )
        for elapsed in [0, 900, 1800, 3600]:
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            # Deliberately skew rank one's clock. Both must follow rank zero.
            timer._offset = elapsed + rank * 300
            scheduler.step()
            expected = 0.5 * (1 + math.cos(math.pi * elapsed / 3600))
            assert scheduler.get_last_lr()[0] == pytest.approx(expected)
        values = torch.cat([p.detach().flatten() for p in model.parameters()])
        peers = [torch.empty_like(values) for _ in range(2)]
        torch.distributed.all_gather(peers, values)
        assert torch.equal(peers[0], peers[1])
    finally:
        torch.distributed.destroy_process_group()


class TestTimeCosineSchedule:
    """Wall-clock cosine: LR annealed over trainer.max_time via Lightning's Timer."""

    def _build(
        self, *, warmup: float | int = 0.0, min_lr_ratio: float = 0.0
    ) -> tuple[torch.optim.lr_scheduler.LambdaLR, Timer]:
        timer = Timer(duration="00:01:00:00")  # 3600 s budget
        m = _ToyMixinUser()
        m.trainer = _FakeTrainer(timer)  # type: ignore[attr-defined]
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        cfg = cast(
            "Any",
            {
                "scheduler": "cosine",
                "scheduler_interval": "time",
                "min_lr_ratio": min_lr_ratio,
                "warmup": warmup,
            },
        )
        sched = cast(WarmupReduceLROnPlateau, m._create_scheduler(opt, cfg))
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)
        return sched, timer

    def _lr_at(
        self,
        sched: torch.optim.lr_scheduler.LambdaLR,
        timer: Timer,
        elapsed_s: int,
    ) -> float:
        # start_time is None (training not "started"), so time_elapsed() == _offset.
        timer._offset = elapsed_s
        sched.optimizer.step()
        sched.step()
        return float(sched.get_last_lr()[0])

    def test_zero_floor_warns_that_the_tail_cannot_show_convergence(self) -> None:
        """A zero floor is allowed but must never be silent.

        Annealing to exactly zero flattens the loss tail whether or not the run
        converged, so the run's own logs cannot answer "did this converge?".
        That silence invalidated verdicts in three separate workstreams. The
        default value stays 0.0 (changing it would break reproduction of every
        earlier run -- see tests/scripts/test_training.py); the warning is the
        prevention.
        """
        with pytest.warns(UserWarning, match="CANNOT be used to judge convergence"):
            self._build(min_lr_ratio=0.0)

    def test_non_zero_floor_does_not_warn(self) -> None:
        """The warning must fire on the trap only, or it becomes noise."""
        warnings.simplefilter("error", UserWarning)
        try:
            self._build(min_lr_ratio=0.05)
        finally:
            warnings.resetwarnings()

    def test_zero_elapsed_is_full_lr(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sched, timer = self._build()
        assert self._lr_at(sched, timer, 0) == pytest.approx(1.0)

    def test_half_budget_is_cosine_half(self) -> None:
        sched, timer = self._build()
        # progress 0.5 -> 0.5 * (1 + cos(pi/2)) = 0.5
        assert self._lr_at(sched, timer, 1800) == pytest.approx(0.5)

    def test_full_budget_hits_floor(self) -> None:
        sched, timer = self._build(min_lr_ratio=0.1)
        # progress 1.0 -> cosine 0 -> min_lr_ratio
        assert self._lr_at(sched, timer, 3600) == pytest.approx(0.1)

    def test_past_budget_clamps_to_floor(self) -> None:
        sched, timer = self._build()
        assert self._lr_at(sched, timer, 7200) == pytest.approx(0.0)

    def test_warmup_is_fraction_of_budget(self) -> None:
        sched, timer = self._build(warmup=0.1)
        # progress 0.05, within warmup 0.1 -> warm factor 0.5
        expected = 0.5 * 0.5 * (1.0 + math.cos(math.pi * 0.05))
        assert self._lr_at(sched, timer, 180) == pytest.approx(expected, rel=1e-4)

    def test_warmup_completes_at_fraction(self) -> None:
        sched, timer = self._build(warmup=0.1)
        # progress == warmup -> full cosine, no warm factor
        expected = 0.5 * (1.0 + math.cos(math.pi * 0.1))
        assert self._lr_at(sched, timer, 360) == pytest.approx(expected, rel=1e-6)

    def test_requires_time_budget(self) -> None:
        m = _ToyMixinUser()
        m.trainer = _FakeTrainer(None)  # type: ignore[attr-defined]  # no Timer
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        cfg = cast("Any", {"scheduler": "cosine", "scheduler_interval": "time"})
        with pytest.raises(ValueError, match="max_time"):
            m._create_scheduler(opt, cfg)

    def test_rejects_restarts_in_time_mode(self) -> None:
        m = _ToyMixinUser()
        m.trainer = _FakeTrainer(Timer(duration="00:01:00:00"))  # type: ignore[attr-defined]
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        cfg = cast(
            "Any",
            {"scheduler": "cosine_with_restarts", "scheduler_interval": "time"},
        )
        with pytest.raises(ValueError, match="cosine_with_restarts"):
            m._create_scheduler(opt, cfg)

    def test_rejects_absolute_warmup_in_time_mode(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            self._build(warmup=100)

    @pytest.mark.skipif(
        not torch.distributed.is_gloo_available(), reason="Gloo is unavailable"
    )
    def test_distributed_clocks_produce_identical_updates(self, tmp_path: Path) -> None:
        try:
            spawn(
                _check_distributed_time_schedule,
                args=((tmp_path / "gloo-rendezvous").as_uri(),),
                nprocs=2,
                join=True,
            )
        except ProcessRaisedException as exc:
            # Some Windows builds advertise Gloo but cannot initialize it.
            if sys.platform == "win32" and (
                "RuntimeError: makeDeviceForHostname(): unsupported gloo device"
                in str(exc)
            ):
                pytest.skip("This Windows PyTorch build cannot initialize Gloo")
            raise


class TestShapeParameterGroup:
    """_create_optimizer routes declared shape parameters to a dedicated group."""

    def _cfg(self, **extra: Any) -> Any:
        base: dict[str, Any] = {
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
        }
        base.update(extra)
        return cast("Any", base)

    def _shape_group(
        self, opt: torch.optim.Optimizer, target: nn.Parameter
    ) -> dict[str, Any]:
        return next(
            g for g in opt.param_groups if any(p is target for p in g["params"])
        )

    def test_no_shape_params_single_group(self) -> None:
        # A model with no shape_parameters() keeps the byte-identical single group.
        m = _ToyMixinUser()
        opt = m._create_optimizer(self._cfg())
        assert len(opt.param_groups) == 1

    def test_shape_param_gets_own_group(self) -> None:
        m = _ShapeMixinUser()
        opt = m._create_optimizer(self._cfg(nu_lr_mult=100.0))
        assert len(opt.param_groups) == 2
        shape = self._shape_group(opt, m.loss_func.raw_nu)
        rest = next(g for g in opt.param_groups if g is not shape)
        # The shape group carries exactly raw_nu, at lr * mult, no weight decay.
        assert shape["lr"] == pytest.approx(1e-4 * 100.0)
        assert shape["weight_decay"] == 0.0
        assert len(shape["params"]) == 1
        # The bulk keeps the base LR and the configured decay, without raw_nu.
        assert rest["lr"] == pytest.approx(1e-4)
        assert rest["weight_decay"] == pytest.approx(0.01)
        assert all(p is not m.loss_func.raw_nu for p in rest["params"])

    def test_default_mult_is_unit_but_decay_still_excluded(self) -> None:
        # Absent nu_lr_mult the shape LR equals the base LR; only decay differs.
        m = _ShapeMixinUser()
        opt = m._create_optimizer(self._cfg())
        shape = self._shape_group(opt, m.loss_func.raw_nu)
        assert shape["lr"] == pytest.approx(1e-4)
        assert shape["weight_decay"] == 0.0

    def test_nonpositive_mult_raises(self) -> None:
        m = _ShapeMixinUser()
        with pytest.raises(ValueError, match="nu_lr_mult must be positive"):
            m._create_optimizer(self._cfg(nu_lr_mult=0.0))

    def test_scheduler_scales_shape_group_by_mult(self) -> None:
        # The single cosine lambda multiplies every group, so the shape group
        # rides the same cosine shape scaled by nu_lr_mult throughout training.
        m = _ShapeMixinUser()
        m.optimizer_config = self._cfg(
            learning_rate=1.0,
            nu_lr_mult=50.0,
            scheduler="cosine",
            scheduler_interval="step",
            cosine_steps=1000,
            warmup=0,
        )
        result = m.configure_optimizers()
        result_any = cast("dict[str, Any]", result)
        sched = result_any["lr_scheduler"]["scheduler"]
        opt = sched.optimizer
        groups = opt.param_groups
        shape_idx = next(
            i
            for i, g in enumerate(groups)
            if any(p is m.loss_func.raw_nu for p in g["params"])
        )
        rest_idx = 1 - shape_idx
        # Step 0 (cosine = 1): bulk at lr=1.0, shape at lr=50.0.
        last = sched.get_last_lr()
        assert last[rest_idx] == pytest.approx(1.0)
        assert last[shape_idx] == pytest.approx(50.0)
        # The ratio stays exactly nu_lr_mult as the cosine decays.
        for _ in range(123):
            opt.step()
            sched.step()
        last = sched.get_last_lr()
        assert last[shape_idx] == pytest.approx(50.0 * last[rest_idx])


class TestProdigy:
    """An optimizer that sets its own step size (2026-11-15, after D208 diverged)."""

    _CFG: ClassVar[dict[str, Any]] = {
        "optimizer": "prodigy",
        "learning_rate": 1.0,
        "betas": [0.9, 0.99],
        "scheduler": "plateau",
        "min_lr_ratio": 0.01,
    }

    def test_it_sets_its_own_step_and_the_plateau_cuts_the_multiplier(
        self, mixin: _ToyMixinUser
    ) -> None:
        """Caught by: building Adam for prodigy, or a floor not in multiplier units."""
        optimizer = mixin._create_optimizer(cast("Any", self._CFG))
        group = optimizer.param_groups[0]
        start = group["d"]
        for _ in range(20):
            optimizer.zero_grad()
            mixin.lin(torch.ones(3, 4)).pow(2).sum().backward()
            optimizer.step()
        assert group["d"] > start  # the estimate moved off its starting value
        scheduler = cast(
            WarmupReduceLROnPlateau, mixin._create_scheduler(optimizer, self._CFG)
        )
        assert scheduler.min_lrs == [pytest.approx(0.01)]
        # D350: a warm start sets prodigy_d0 to the source's d; absent, Prodigy's d0.
        assert start == pytest.approx(1e-6)
        inherited = mixin._create_optimizer(
            cast("Any", {**self._CFG, PRODIGY_D0_KEY: 0.00524})
        )
        assert inherited.param_groups[0]["d"] == pytest.approx(0.00524)

    def test_an_unknown_name_is_still_refused_by_name(
        self, mixin: _ToyMixinUser
    ) -> None:
        cfg = cast("Any", {"optimizer": "lion", "learning_rate": 1e-4})
        with pytest.raises(ValueError, match="Unsupported optimizer: lion"):
            mixin._create_optimizer(cfg)


class TestPlateauSchedulerConfig:
    """Every ReduceLROnPlateau knob must be reachable from config.

    They were hardcoded (patience=10, torch's factor=0.1, no floor) until
    2026-08-05, which left the decay schedule unrecorded in any run's config.
    """

    _CFG: ClassVar[dict[str, Any]] = {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "scheduler": "plateau",
        "plateau_patience": 25,
        "plateau_factor": 0.3,
        "plateau_cooldown": 5,
        "plateau_threshold": 1e-3,
        "plateau_threshold_mode": "abs",
        "min_lr_ratio": 0.01,
    }

    def _build(self, mixin: _ToyMixinUser, **overrides: Any) -> WarmupReduceLROnPlateau:
        cfg = {**self._CFG, **overrides}
        optimizer = torch.optim.AdamW(mixin.parameters(), lr=cfg["learning_rate"])
        # _create_scheduler is declared as the LRScheduler base; these tests
        # exercise the plateau branch specifically, so name the concrete class
        # rather than widening the factory's own annotation.
        return cast(WarmupReduceLROnPlateau, mixin._create_scheduler(optimizer, cfg))

    def test_every_value_reaches_the_scheduler(self, mixin: _ToyMixinUser) -> None:
        sched = self._build(mixin)
        assert sched.patience == 25
        assert sched.factor == pytest.approx(0.3)
        assert sched.cooldown == 5
        assert sched.threshold == pytest.approx(1e-3)
        assert sched.threshold_mode == "abs"
        # min_lr is a RATIO of the base LR, matching the cosine branch's units.
        assert sched.min_lrs[0] == pytest.approx(1e-3 * 0.01)

    def test_the_monitored_quantity_is_configurable(self, mixin: _ToyMixinUser) -> None:
        """``plateau_monitor`` sat in this module's allowed-keys list, wired to
        NOTHING, while the branch hard-coded ``val_loss``.

        A config naming it therefore passed every check the repo has and then
        decayed the rate against a quantity nobody had asked for. It matters
        for the distillation restarts, which select on ``val_rmse``.

        Mutation this catches: restoring the hard-coded ``"val_loss"``.
        """
        optimizer = torch.optim.AdamW(mixin.parameters(), lr=1e-3)
        config = cast(
            dict[str, Any],
            mixin._build_scheduler_config(
                optimizer, {**self._CFG, "plateau_monitor": "val_rmse"}
            ),
        )

        assert config["lr_scheduler"]["monitor"] == "val_rmse"

    def test_the_monitor_defaults_to_val_loss(self, mixin: _ToyMixinUser) -> None:
        """Every existing plateau config must be bit-identical: no key, no
        change. Mutation this catches: a different default, which would silently
        re-point every run that never named a monitor."""
        optimizer = torch.optim.AdamW(mixin.parameters(), lr=1e-3)
        config = cast(
            dict[str, Any], mixin._build_scheduler_config(optimizer, self._CFG)
        )

        assert config["lr_scheduler"]["monitor"] == "val_loss"
        assert config["lr_scheduler"]["interval"] == "epoch"

    @pytest.mark.parametrize("interval", ["time", "step"])
    def test_a_non_epoch_interval_is_refused_rather_than_mapped(
        self, mixin: _ToyMixinUser, interval: str
    ) -> None:
        """A plateau schedule decays on a MONITORED METRIC, so only ``epoch``
        means anything for it.

        ``time`` means "step a wall-clock lambda", and a plateau schedule has
        none. ⛔ **``step`` is refused for the same sentence, and that is the
        2026-09-05 fix (C3 Gate-2 MUST-3).** Until this branch honoured
        ``scheduler_interval`` at all it hard-coded ``epoch``, so ``step`` was
        unreachable; honouring it made ``step`` reachable and it does precisely
        what ``time`` was refused for -- polls a metric that only exists once an
        epoch, once per batch, so ``plateau_patience`` counts batches while
        meaning epochs.

        Nothing scored moves: ``campaign2_plateau.yaml`` is the only config
        selecting the plateau schedule and it sets no ``scheduler_interval``.

        Mutation this catches (verified present, seen red): narrowing the guard
        back to ``interval == "time"``, which lets ``step`` through.
        """
        optimizer = torch.optim.AdamW(mixin.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="meaningless for ReduceLROnPlateau"):
            mixin._build_scheduler_config(
                optimizer, {**self._CFG, "scheduler_interval": interval}
            )

    def test_the_epoch_interval_is_still_accepted(self, mixin: _ToyMixinUser) -> None:
        """Negative control: the default must still build, or the guard is too
        wide and every existing plateau config breaks."""
        optimizer = torch.optim.AdamW(mixin.parameters(), lr=1e-3)
        built = cast(
            dict[str, Any],
            mixin._build_scheduler_config(
                optimizer, {**self._CFG, "scheduler_interval": "epoch"}
            ),
        )
        assert built["lr_scheduler"]["interval"] == "epoch"

    def test_defaults_preserve_the_previous_behaviour(
        self, mixin: _ToyMixinUser
    ) -> None:
        optimizer = torch.optim.AdamW(mixin.parameters(), lr=1e-3)
        sched = cast(
            WarmupReduceLROnPlateau,
            mixin._create_scheduler(
                optimizer,
                {"optimizer": "adamw", "learning_rate": 1e-3, "scheduler": "plateau"},
            ),
        )
        assert sched.patience == 10
        assert sched.factor == pytest.approx(0.1)

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("plateau_factor", 1.0, "plateau_factor must be in"),
            ("plateau_factor", 0.0, "plateau_factor must be in"),
            ("plateau_cooldown", -1, "plateau_cooldown must be"),
            ("plateau_threshold", -1e-3, "plateau_threshold must be"),
            ("plateau_threshold_mode", "sideways", "plateau_threshold_mode must be"),
            ("plateau_patience", 0, "plateau_patience must be"),
        ],
    )
    def test_rejects_out_of_range_values(
        self, mixin: _ToyMixinUser, key: str, value: Any, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            self._build(mixin, **{key: value})


class TestPlateauWarmup:
    """The plateau branch must honour `warmup`; it silently ignored it.

    D1 rules the campaign schedule as "constant LR AFTER WARMUP", but warmup
    was implemented only on the cosine branch, so every cell of a 105-cell
    campaign trained with none and nothing raised. See
    `CAMPAIGN_TRIAGE_2026-08-08` §3.
    """

    _CFG: ClassVar[dict[str, Any]] = {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "scheduler": "plateau",
        "plateau_patience": 2,
        "plateau_factor": 0.3,
        "plateau_cooldown": 0,
        "plateau_threshold": 1e-4,
        "plateau_threshold_mode": "abs",
        "min_lr_ratio": 0.01,
    }

    def _build(
        self, mixin: _ToyMixinUser, **overrides: Any
    ) -> tuple[torch.optim.Optimizer, WarmupReduceLROnPlateau]:
        cfg = {**self._CFG, **overrides}
        optimizer = torch.optim.AdamW(mixin.parameters(), lr=cfg["learning_rate"])
        return optimizer, cast(
            WarmupReduceLROnPlateau, mixin._create_scheduler(optimizer, cfg)
        )

    @staticmethod
    def _lr(optimizer: torch.optim.Optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def test_absent_warmup_leaves_the_rate_at_base(self, mixin: _ToyMixinUser) -> None:
        optimizer, _sched = self._build(mixin)
        assert self._lr(optimizer) == pytest.approx(1e-3)

    def test_warmup_starts_the_rate_below_base(self, mixin: _ToyMixinUser) -> None:
        optimizer, _sched = self._build(mixin, warmup=4)
        # First epoch runs at base/(warmup+1), matching the cosine branch's ramp.
        assert self._lr(optimizer) == pytest.approx(1e-3 / 5)

    def test_rate_climbs_linearly_and_reaches_base_after_warmup(
        self, mixin: _ToyMixinUser
    ) -> None:
        optimizer, sched = self._build(mixin, warmup=4)
        seen = [self._lr(optimizer)]
        for _ in range(4):
            sched.step(1.0)
            seen.append(self._lr(optimizer))
        assert seen == pytest.approx([2e-4, 4e-4, 6e-4, 8e-4, 1e-3])
        assert not sched.in_warmup

    def test_decay_cannot_fire_during_warmup(self, mixin: _ToyMixinUser) -> None:
        """A rising loss through the ramp must not trigger a reduction.

        The ramp is exactly when the loss is least informative, which is the
        reason a warmup exists at all.
        """
        optimizer, sched = self._build(mixin, warmup=3)
        for value in (5.0, 6.0, 7.0):
            sched.step(value)
        # Back at base, not base * factor.
        assert self._lr(optimizer) == pytest.approx(1e-3)

    def test_decay_resumes_normally_once_warmed_up(self, mixin: _ToyMixinUser) -> None:
        optimizer, sched = self._build(mixin, warmup=2)
        for _ in range(2):
            sched.step(1.0)
        assert self._lr(optimizer) == pytest.approx(1e-3)
        # patience=2: the first call sets the best, then bad epochs accumulate.
        for value in (1.0, 2.0, 2.0, 2.0):
            sched.step(value)
        assert self._lr(optimizer) == pytest.approx(1e-3 * 0.3)

    def test_fractional_warmup_is_rejected_with_its_reason(
        self, mixin: _ToyMixinUser
    ) -> None:
        with pytest.raises(ValueError, match="whole number of"):
            self._build(mixin, warmup=0.5)

    def test_negative_warmup_is_rejected(self, mixin: _ToyMixinUser) -> None:
        with pytest.raises(ValueError, match="warmup must be >= 0"):
            self._build(mixin, warmup=-1)

    def test_non_finite_warmup_is_rejected(self, mixin: _ToyMixinUser) -> None:
        with pytest.raises(ValueError, match="warmup must be finite"):
            self._build(mixin, warmup=math.inf)

    def test_still_looks_like_a_plateau_scheduler_to_lightning(
        self, mixin: _ToyMixinUser
    ) -> None:
        """`configure_optimizers` routes on isinstance, so the subclass must pass."""
        _optimizer, sched = self._build(mixin, warmup=2)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)


class TestPlateauRelativeThresholdSignGuard:
    """A relative threshold cannot schedule a negative objective.

    torch's `rel` bar is `best * (1 - threshold)`, which for a negative best
    sits ABOVE it, so every epoch counts as an improvement and the rate never
    decays. All five `l96c`/`lowrank` cells of the campaign finished at their
    base rate having never cut once, and then failed the convergence gate's
    learning-rate leg for that configuration reason. See
    `CAMPAIGN_TRIAGE_2026-08-08` §4.
    """

    _CFG: ClassVar[dict[str, Any]] = {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "scheduler": "plateau",
        "plateau_patience": 5,
        "plateau_factor": 0.3,
        "plateau_cooldown": 5,
        "plateau_threshold": 0.0072,
        "min_lr_ratio": 0.01,
    }

    def _build(self, mixin: _ToyMixinUser, **overrides: Any) -> WarmupReduceLROnPlateau:
        cfg = {**self._CFG, **overrides}
        optimizer = torch.optim.AdamW(mixin.parameters(), lr=cfg["learning_rate"])
        # _create_scheduler is declared as the LRScheduler base; these tests
        # exercise the plateau branch specifically, so name the concrete class
        # rather than widening the factory's own annotation.
        return cast(WarmupReduceLROnPlateau, mixin._create_scheduler(optimizer, cfg))

    def test_negative_value_under_rel_mode_raises(self, mixin: _ToyMixinUser) -> None:
        sched = self._build(mixin, plateau_threshold_mode="rel")
        with pytest.raises(ValueError, match="cannot schedule a NEGATIVE"):
            sched.step(-4.58)

    def test_the_error_names_the_fix(self, mixin: _ToyMixinUser) -> None:
        sched = self._build(mixin, plateau_threshold_mode="rel")
        with pytest.raises(ValueError, match="plateau_threshold_mode=abs"):
            sched.step(-4.58)

    def test_positive_value_under_rel_mode_is_unaffected(
        self, mixin: _ToyMixinUser
    ) -> None:
        sched = self._build(mixin, plateau_threshold_mode="rel")
        sched.step(1.0)
        assert sched.best == pytest.approx(1.0)

    def test_abs_mode_schedules_a_negative_objective_normally(
        self, mixin: _ToyMixinUser
    ) -> None:
        """The prescribed fix must actually work on the failing values."""
        sched = self._build(mixin, plateau_threshold_mode="abs", plateau_threshold=1e-3)
        optimizer = sched.optimizer
        base_lr = self._CFG["learning_rate"]
        sched.step(-4.58)
        # A flat, genuinely non-improving tail must now cut the rate.
        for _ in range(12):
            sched.step(-4.58)
        assert float(optimizer.param_groups[0]["lr"]) < base_lr

    def test_zero_is_not_treated_as_negative(self, mixin: _ToyMixinUser) -> None:
        sched = self._build(mixin, plateau_threshold_mode="rel")
        sched.step(0.0)
        assert sched.best == pytest.approx(0.0)


class TestRelAbsThresholdMode:
    """The sign-safe relative bar that keeps a negative objective schedulable.

    torch's ``rel`` bar is ``best * (1 - threshold)``, which for a NEGATIVE best
    sits ABOVE it -- so every epoch counts as an improvement and the rate never
    cuts. That is what stopped all five ``l96c``/``lowrank`` cells decaying.
    ``rel_abs`` writes the intended semantics directly.
    """

    @staticmethod
    def _sched(threshold_mode, threshold=0.1, mode="min"):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=1.0)
        return WarmupReduceLROnPlateau(
            opt, mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )

    def test_matches_torch_rel_when_best_is_positive(self):
        # A strict generalisation: identical wherever rel was already valid.
        rel = self._sched("rel")
        rel_abs = self._sched("rel_abs")
        best = 1.0
        for a in (0.85, 0.89, 0.90, 0.91, 1.0, 1.2):
            assert rel_abs.is_better(a, best) == rel.is_better(a, best), a

    def test_rel_is_broken_on_a_negative_best(self):
        # Pins the defect: a value WORSE than best still reads as an improvement.
        rel = self._sched("rel")
        assert rel.is_better(-0.95, -1.0), (
            "torch's rel bar on a negative best sits above it -- if this ever "
            "stops being true, rel_abs may no longer be needed"
        )

    def test_rel_abs_is_correct_on_a_negative_best(self):
        rel_abs = self._sched("rel_abs")
        best = -1.0
        # bar = -1.0 - 0.1*1.0 = -1.1; only a genuine improvement clears it.
        assert rel_abs.is_better(-1.2, best)
        assert not rel_abs.is_better(-1.05, best)
        assert not rel_abs.is_better(-0.95, best)
        assert not rel_abs.is_better(best, best)

    def test_rel_abs_handles_a_maximised_metric(self):
        rel_abs = self._sched("rel_abs", mode="max")
        assert rel_abs.is_better(-0.8, -1.0)
        assert not rel_abs.is_better(-0.95, -1.0)

    def test_rel_abs_is_accepted_by_the_config_path(self):
        m = _ToyMixinUser()
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=1e-4)
        cfg = {
            "learning_rate": 1e-4,
            "scheduler": "plateau",
            "plateau_threshold_mode": "rel_abs",
            "plateau_threshold": 0.0072,
        }
        sched = cast(WarmupReduceLROnPlateau, m._create_scheduler(opt, cfg))
        assert sched.threshold_mode == "rel_abs"
        # And it does NOT trip the negative-objective guard, which is the point.
        sched.step(-4.58)

    def test_rel_abs_survives_a_save_load_round_trip(self):
        # torch's load_state_dict restores threshold_mode from the state dict and
        # THEN validates it against {"rel", "abs"}, so a rel_abs checkpoint was
        # unresumable: ValueError: threshold mode rel_abs is unknown!. Found by
        # an lr_find smoke test on 2026-08-15, which resumes internally.
        sched = self._sched("rel_abs", threshold=1e-4)
        sched.step(0.05)
        sched.step(0.04)
        state = sched.state_dict()
        assert state["threshold_mode"] == "rel_abs"

        restored = self._sched("rel_abs", threshold=1e-4)
        restored.load_state_dict(state)
        assert restored.threshold_mode == "rel_abs"
        assert restored.best == pytest.approx(0.04)
        # And the restored scheduler still applies OUR bar, not torch's.
        assert restored.is_better(-1.2, -1.0)
        assert not restored.is_better(-0.95, -1.0)

    def test_rel_abs_does_not_treat_a_real_negative_infinity_as_unset(self):
        # Found in review of the sentinel fix. In `min` mode the sentinel is
        # +inf, so a `best` of -inf is a REAL best (the metric genuinely got
        # there) that nothing can improve on. Testing `not isfinite(best)`
        # instead of `best == mode_worse` would wrongly call it unset and accept
        # any finite value as an improvement.
        sched = self._sched("rel_abs", threshold=1e-4, mode="min")
        assert sched.mode_worse == float("inf")
        assert not sched.is_better(0.5, float("-inf"))

    def test_rel_abs_improves_on_the_initial_infinite_best(self):
        # A fresh scheduler's `best` is torch's mode_worse sentinel, +inf. The
        # relative bar `inf - threshold*abs(inf)` is nan and every comparison
        # against nan is False, so before the fix NOTHING beat the sentinel.
        sched = self._sched("rel_abs")
        assert sched.best == float("inf")
        assert sched.is_better(0.035, sched.best)
        # A nan metric is still not an improvement.
        assert not sched.is_better(float("nan"), float("inf"))

    def test_rel_abs_does_not_decay_on_a_strictly_improving_curve(self):
        # The observable defect: with `best` stuck at inf the bad-epoch counter
        # runs free, so the rate cut on a fixed patience+cooldown timer while
        # the metric was still improving every single epoch. Measured on a real
        # 132-epoch run before the fix: cuts at epochs 16/37/58/79, 21 apart.
        sched = self._sched("rel_abs", threshold=1e-4)
        sched.patience = 15
        sched.cooldown = 5
        sched.factor = 0.3
        base_lr = sched.optimizer.param_groups[0]["lr"]
        for epoch in range(25):
            sched.step(0.05 - 0.001 * epoch)
        assert sched.best == pytest.approx(0.026)
        assert sched.optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

    def test_rel_abs_still_decays_on_a_genuinely_flat_curve(self):
        # The other half: the rule must still fire when nothing improves.
        sched = self._sched("rel_abs", threshold=1e-4)
        sched.patience = 3
        sched.cooldown = 0
        sched.factor = 0.3
        base_lr = sched.optimizer.param_groups[0]["lr"]
        for _ in range(6):
            sched.step(0.05)
        assert sched.optimizer.param_groups[0]["lr"] < base_lr

    def test_both_comparison_hook_names_agree(self):
        # torch <2.10 calls `is_better`; >=2.10 calls the private `_is_better`.
        # Whichever name this torch reaches for must land on the same rule.
        sched = self._sched("rel_abs", threshold=1e-4)
        for a, best in [(0.035, math.inf), (0.05, 0.04), (-1.2, -1.0)]:
            assert sched._is_better(a, best) == sched.is_better(a, best)

    def test_step_reads_the_metric_whichever_hook_torch_calls(self):
        """The regression test for the defect `_is_better` existed to fix.

        Asserting parity of the two names cannot catch this: the defect was that
        torch's ``step`` bypassed our override entirely, so ``best`` never left
        the ``inf`` sentinel and the rate cut on a blind patience+cooldown timer
        regardless of the curve. This drives the REAL ``step`` path, so it fails
        on any torch whose chosen hook name we have not overridden -- which is
        what makes it version-agnostic rather than a restatement of the fix.

        Measured pre-fix on job 6029506 (Isambard, torch 2.10.0+cu126): 136
        epochs, ``best: inf`` in the final checkpoint, cuts at epochs
        16/37/58/79, exactly 21 = patience + cooldown + 1 apart.
        """
        sched = self._sched("rel_abs", threshold=1e-4)
        sched.patience = 15
        sched.cooldown = 5
        base_lr = sched.optimizer.param_groups[0]["lr"]
        # A steadily improving curve, run well past patience + cooldown + 1 = 21.
        for i in range(40):
            sched.step(1.20 - 0.001 * i)
        assert math.isfinite(sched.best), (
            f"`best` is still the {sched.mode_worse} sentinel after 40 improving "
            "epochs -- torch's step is not reaching our comparison hook"
        )
        assert sched.best == pytest.approx(1.20 - 0.001 * 39)
        assert sched.optimizer.param_groups[0]["lr"] == pytest.approx(base_lr), (
            "the rate cut while the metric was still improving -- a timer, not a "
            "plateau rule"
        )


class TestRejectsIgnoredSchedulerKeys:
    """A knob the selected schedule cannot read must stop the run, not train.

    Both cases below are real. The `warmup`-on-plateau case ran a 105-cell
    campaign with no warmup (2026-08-08). The scheduler=null case is how the
    constant-rate arm was launched on 2026-08-15 -- `optimizer=campaign2_plateau
    optimizer.scheduler=null` -- carrying five plateau knobs that could not act.
    """

    def _cfg(self, **overrides: Any) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        }
        cfg.update(overrides)
        return cfg

    def test_rejects_plateau_knobs_when_the_scheduler_is_removed(
        self, mixin: _ToyMixinUser
    ) -> None:
        mixin.optimizer_config = self._cfg(
            scheduler=None, plateau_patience=15, plateau_factor=0.3
        )
        with pytest.raises(ValueError, match="will never be read"):
            mixin.configure_optimizers()

    def test_names_the_offending_key_and_its_owner(self, mixin: _ToyMixinUser) -> None:
        mixin.optimizer_config = self._cfg(scheduler="cosine", plateau_patience=15)
        with pytest.raises(ValueError, match="'plateau_patience'") as excinfo:
            mixin.configure_optimizers()
        assert "read only by scheduler=plateau" in str(excinfo.value)

    def test_rejects_cosine_horizon_on_a_plateau_run(
        self, mixin: _ToyMixinUser
    ) -> None:
        mixin.optimizer_config = self._cfg(scheduler="plateau", cosine_epochs=100)
        with pytest.raises(ValueError, match="'cosine_epochs'"):
            mixin.configure_optimizers()

    def test_accepts_a_config_whose_keys_the_schedule_reads(
        self, mixin: _ToyMixinUser
    ) -> None:
        mixin.optimizer_config = self._cfg(
            scheduler="plateau",
            warmup=0,
            plateau_patience=15,
            plateau_factor=0.3,
            plateau_cooldown=5,
            plateau_threshold=1e-4,
            plateau_threshold_mode="rel_abs",
            min_lr_ratio=0.01,
        )
        assert mixin.configure_optimizers() is not None

    def test_a_null_key_is_unset_not_ignored(self, mixin: _ToyMixinUser) -> None:
        """Declaring a key as null is how a config says "not using this"."""
        mixin.optimizer_config = self._cfg(scheduler=None, plateau_patience=None)
        assert mixin.configure_optimizers() is not None

    def test_optimizer_level_keys_are_never_judged(self, mixin: _ToyMixinUser) -> None:
        """Only keys some schedule owns are in scope -- not learning_rate etc."""
        mixin.optimizer_config = self._cfg(scheduler=None, betas=[0.9, 0.99])
        assert mixin.configure_optimizers() is not None

    def test_an_unknown_scheduler_name_is_named_with_the_valid_set(
        self, mixin: _ToyMixinUser
    ) -> None:
        mixin.optimizer_config = self._cfg(scheduler="cosin")
        with pytest.raises(ValueError, match="Unsupported scheduler: cosin"):
            mixin.configure_optimizers()


class TestThresholdAgainstNoise:
    """The improvement bar must clear the metric's own epoch-to-epoch scatter.

    Otherwise noise resets the bad-epoch counter, the rate decays late or never,
    and the convergence gate blames the schedule for a threshold problem.
    Measured on the real arm 2026-08-15: `val_loss` sd 0.001047 vs `val/fit` sd
    0.000028 (37x), against a bar of ~3.6e-6 at a best near 0.036.
    """

    @staticmethod
    def _sched(threshold: float, mode: str = "rel_abs") -> WarmupReduceLROnPlateau:
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-4)
        return WarmupReduceLROnPlateau(
            optimizer,
            warmup_epochs=0,
            mode="min",
            patience=15,
            cooldown=5,
            factor=0.3,
            threshold=threshold,
            threshold_mode=mode,
            min_lr=1e-6,
        )

    @staticmethod
    def _noisy_curve(n: int, level: float, noise: float) -> list[float]:
        """Deterministic alternating scatter of half-amplitude `noise`."""
        return [level + (noise / 2 if i % 2 else -noise / 2) for i in range(n)]

    @staticmethod
    def _drive(sched: WarmupReduceLROnPlateau, curve: list[float]) -> None:
        """Step the whole curve through, so `pytest.warns` wraps ONE call."""
        for value in curve:
            sched.step(value)

    def test_warns_on_the_measured_val_loss_case(self) -> None:
        sched = self._sched(threshold=1e-4)
        with pytest.warns(UserWarning, match="BELOW the monitored metric"):
            self._drive(sched, self._noisy_curve(20, level=0.036, noise=0.001047))

    def test_quiet_metric_with_an_adequate_bar_does_not_warn(self) -> None:
        sched = self._sched(threshold=0.05)  # bar ~= 0.0018 at best ~0.036
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            for value in self._noisy_curve(20, level=0.036, noise=0.000028):
                sched.step(value)

    def test_warns_at_most_once(self) -> None:
        sched = self._sched(threshold=1e-4)
        with pytest.warns(UserWarning, match="BELOW the monitored metric") as record:
            self._drive(sched, self._noisy_curve(60, level=0.036, noise=0.001047))
        noise_warnings = [
            w for w in record if "BELOW the monitored metric" in str(w.message)
        ]
        assert len(noise_warnings) == 1

    def test_silent_before_the_window_is_full(self) -> None:
        """Too few epochs is NOT a pass -- it is not enough data to judge."""
        sched = self._sched(threshold=1e-4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            for value in self._noisy_curve(5, level=0.036, noise=0.001047):
                sched.step(value)

    def test_a_steadily_improving_curve_is_not_called_noisy(self) -> None:
        """Trend is not scatter.

        A run marching down by a constant amount each epoch is the healthiest
        case there is; reading its step size as noise would warn on exactly the
        runs that need no warning. Caught by this suite's own
        `rel_abs`-improving-curve test when the first implementation used the
        median absolute successive difference.
        """
        sched = self._sched(threshold=1e-4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            for epoch in range(20):
                sched.step(0.05 - 0.001 * epoch)

    def test_scatter_riding_on_a_trend_is_still_caught(self) -> None:
        """The other half: a noisy curve that is ALSO improving must warn."""
        sched = self._sched(threshold=1e-4)
        curve = [
            0.05 - 0.0002 * epoch + (0.0005 if epoch % 2 else -0.0005)
            for epoch in range(20)
        ]
        with pytest.warns(UserWarning, match="BELOW the monitored metric"):
            self._drive(sched, curve)


#: The horizon a cosine run was trained under, and the longer one an extension
#: would resolve to. Named so the assertions compare against a value, not a
#: literal repeated in five places.
_TRAINED_HORIZON = 240
_EXTENDED_HORIZON = 720


class TestCosineHorizonPersistence:
    """The cosine horizon must ride in the checkpoint and refuse a silent change.

    Before this, the horizon lived only in a closure that `LambdaLR.state_dict`
    deliberately drops, so resuming under a different `trainer.max_epochs`
    re-planned the anneal with nothing on disk recording it: measured
    2026-08-15, the rate climbed from its 5% floor back to 75.7% of base.
    """

    @staticmethod
    def _sched(horizon: int) -> HorizonAwareLambdaLR:
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-4)
        return HorizonAwareLambdaLR(
            optimizer,
            lr_lambda=lambda t: 0.5 * (1.0 + math.cos(math.pi * t / horizon)),
            cosine_horizon=horizon,
            horizon_units="epochs",
        )

    def test_the_horizon_reaches_the_state_dict(self) -> None:
        state = self._sched(_TRAINED_HORIZON).state_dict()
        assert state["cosine_horizon"] == _TRAINED_HORIZON

    def test_refuses_a_resume_under_a_different_horizon(self, monkeypatch) -> None:
        monkeypatch.delenv(ALLOW_COSINE_HORIZON_CHANGE_ENV_VAR, raising=False)
        saved = self._sched(_TRAINED_HORIZON).state_dict()
        with pytest.raises(ValueError, match="REFUSING TO RESUME"):
            self._sched(_EXTENDED_HORIZON).load_state_dict(saved)

    def test_the_error_names_both_horizons_and_the_fix(self, monkeypatch) -> None:
        monkeypatch.delenv(ALLOW_COSINE_HORIZON_CHANGE_ENV_VAR, raising=False)
        saved = self._sched(_TRAINED_HORIZON).state_dict()
        with pytest.raises(ValueError, match="REFUSING TO RESUME") as excinfo:
            self._sched(_EXTENDED_HORIZON).load_state_dict(saved)
        message = str(excinfo.value)
        assert "240 epochs" in message
        assert "720 epochs" in message
        assert "optimizer.cosine_epochs=240" in message

    def test_the_same_horizon_resumes_silently(self, monkeypatch) -> None:
        monkeypatch.delenv(ALLOW_COSINE_HORIZON_CHANGE_ENV_VAR, raising=False)
        saved = self._sched(_TRAINED_HORIZON).state_dict()
        resumed = self._sched(_TRAINED_HORIZON)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            resumed.load_state_dict(saved)
        assert resumed.cosine_horizon == _TRAINED_HORIZON

    def test_opting_in_extends_and_keeps_the_new_horizon(self, monkeypatch) -> None:
        """Deliberate extension must WORK, and must say so."""
        monkeypatch.setenv(ALLOW_COSINE_HORIZON_CHANGE_ENV_VAR, "1")
        saved = self._sched(_TRAINED_HORIZON).state_dict()
        resumed = self._sched(_EXTENDED_HORIZON)
        with pytest.warns(UserWarning, match="Cosine horizon changed on resume"):
            resumed.load_state_dict(saved)
        # The NEW horizon must survive: restoring the old one would undo the
        # extension the operator explicitly asked for.
        assert resumed.cosine_horizon == _EXTENDED_HORIZON


class TestSchedulerStateStaysCheckpointable:
    """New scheduler state must not make checkpoints unloadable.

    `torch.load` defaults to `weights_only=True` since PyTorch 2.6, which
    accepts only plain types. `ReduceLROnPlateau.state_dict` returns EVERY
    attribute except the optimizer, so any exotic object added to the class
    silently rides into the checkpoint and breaks the standard load path.
    Measured on torch 2.8 while adding the threshold-vs-noise check: a `deque`
    attribute produced `UnpicklingError: Weights only load failed`.
    """

    @staticmethod
    def _fresh() -> WarmupReduceLROnPlateau:
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-4)
        return WarmupReduceLROnPlateau(
            optimizer,
            warmup_epochs=0,
            mode="min",
            patience=3,
            cooldown=0,
            factor=0.3,
            threshold=1e-4,
            threshold_mode="rel_abs",
            min_lr=1e-6,
        )

    @classmethod
    def _stepped(cls) -> WarmupReduceLROnPlateau:
        sched = cls._fresh()
        for value in (0.05, 0.049, 0.048):
            sched.step(value)
        return sched

    def test_state_dict_survives_a_weights_only_round_trip(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "sched.ckpt"
        torch.save({"lr_schedulers": [self._stepped().state_dict()]}, path)
        # The assertion IS that this does not raise.
        torch.load(path, map_location="cpu", weights_only=True)

    def test_the_noise_window_is_not_in_the_checkpoint(self) -> None:
        state = self._stepped().state_dict()
        assert "_recent" not in state
        assert "_noise_warned" not in state

    def test_the_real_scheduler_state_is_still_there(self) -> None:
        """Dropping the diagnostic must not drop what resume depends on."""
        state = self._stepped().state_dict()
        for key in ("best", "num_bad_epochs", "cooldown_counter", "threshold_mode"):
            assert key in state, key

    def test_a_resumed_scheduler_rebuilds_its_window(self) -> None:
        """Restoring must leave the window empty and WORKING, not clobbered.

        `load_state_dict` does `self.__dict__.update(state_dict)`, so a state
        dict that no longer carries `_recent` simply leaves the fresh deque in
        place -- which is the intent. The resumed run refills it over the next
        `_NOISE_WINDOW_EPOCHS` epochs.
        """
        saved = self._stepped().state_dict()
        resumed = self._fresh()
        resumed.load_state_dict(saved)
        assert len(resumed._recent) == 0, "the window must not be restored"
        assert resumed.num_bad_epochs == self._stepped().num_bad_epochs, (
            "the real scheduler state must still be restored"
        )
        resumed.step(0.047)
        assert len(resumed._recent) == 1, "and the window must still work"


class TestSchedulerLrPropertyCompatibility:
    """Compatibility with Lightning's Tuner.lr_find (which reads scheduler.lr[0])."""

    def test_warmup_reduce_lr_on_plateau_exposes_lr_property(self) -> None:
        model = nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = WarmupReduceLROnPlateau(opt, warmup_epochs=2)
        assert hasattr(sched, "lr")
        assert isinstance(sched.lr, list)
        assert sched.lr == [opt.param_groups[0]["lr"]]

    def test_horizon_aware_lambda_lr_exposes_lr_property(self) -> None:
        model = nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        sched = HorizonAwareLambdaLR(
            opt, lr_lambda=lambda _step: 1.0, cosine_horizon=100, horizon_units="epochs"
        )
        assert hasattr(sched, "lr")
        assert isinstance(sched.lr, list)
        assert sched.lr == [opt.param_groups[0]["lr"]]


class _HeadModule(nn.Module):
    """Stand-in distribution head that declares itself never-decayed."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(3, 2)

    def no_weight_decay_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())


class _DecayScopeUser(OptimizerMixin):
    """A weight matrix and a bias in the body, plus a declared head."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.dist = _HeadModule()
        self.optimizer_config = None


class TestWeightDecayScope:
    """``weight_decay_scope=matrices`` decays weight matrices only."""

    @staticmethod
    def _cfg(**extra: Any) -> Any:
        base: dict[str, Any] = {
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 3.0,
        }
        base.update(extra)
        return cast("Any", base)

    @staticmethod
    def _group_of(opt: torch.optim.Optimizer, param: torch.Tensor) -> dict[str, Any]:
        return next(g for g in opt.param_groups if any(p is param for p in g["params"]))

    def test_default_scope_is_the_old_single_group(self) -> None:
        """Absent the key, every parameter decays exactly as before."""
        opt = _DecayScopeUser()._create_optimizer(self._cfg())
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(3.0)

    def test_matrices_scope_exempts_biases_and_declared_parameters(self) -> None:
        """Caught by: dropping the ndim rule or the module hook."""
        m = _DecayScopeUser()
        opt = m._create_optimizer(self._cfg(weight_decay_scope="matrices"))
        assert self._group_of(opt, m.lin.weight)["weight_decay"] == pytest.approx(3.0)
        for exempt in (m.lin.bias, m.dist.head.weight, m.dist.head.bias):
            assert self._group_of(opt, exempt)["weight_decay"] == 0.0
        decayed = [
            p for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]
        ]
        assert len(decayed) == 1
        assert decayed[0] is m.lin.weight

    def test_zero_decay_keeps_the_single_group(self) -> None:
        opt = _DecayScopeUser()._create_optimizer(
            self._cfg(weight_decay=0.0, weight_decay_scope="matrices")
        )
        assert len(opt.param_groups) == 1

    def test_unknown_scope_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'everything' is not a valid"):
            _DecayScopeUser()._create_optimizer(
                self._cfg(weight_decay_scope="everything")
            )
