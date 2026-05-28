"""Unit tests for OptimizerMixin warmup resolution."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from autocast.models.optimizer_mixin import OptimizerMixin


class _ToyMixinUser(OptimizerMixin):
    """Minimal nn.Module that exposes OptimizerMixin helpers for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.optimizer_config = None


@pytest.fixture
def mixin() -> _ToyMixinUser:
    return _ToyMixinUser()


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
