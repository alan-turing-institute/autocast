"""Optimizer configuration mixin for Lightning modules."""

import math
from functools import partial
from typing import Any

import heavyball
import torch
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig, OmegaConf
from torch import nn


class OptimizerMixin(nn.Module):
    """Mixin class providing optimizer configuration for Lightning modules.

    Inherits from nn.Module to ensure parameters() method is available.
    Requires the class to have:
        - self.optimizer_config: DictConfig | dict[str, Any] | None
        - self.trainer: Lightning Trainer instance (optional, for scheduler)
    """

    # Type hints for attributes expected from the concrete class
    optimizer_config: DictConfig | dict[str, Any] | None

    @staticmethod
    def _precond_prob_schedule(
        n: int,
        *,
        max_prob: float = 1.0,
        min_prob: float = 0.01,
        decay: float = 0.999,
        flat_start: int = 0,
    ) -> float:
        """Preconditioner update probability schedule.

        Implemented to align with LOLA:
        https://github.com/francois-rozet/lola/blob/21a4354b327e6e5ee06da5075ba3bd1dd88c61f1/lola/optim.py
        """
        return max(min_prob, max_prob * decay ** max(n - flat_start, 0))

    def _get_scheduler_interval(self, cfg: dict[str, Any]) -> str:
        """Return scheduler interval ('epoch', 'step', or 'time')."""
        interval = str(cfg.get("scheduler_interval", "epoch")).lower()
        if interval not in {"epoch", "step", "time"}:
            msg = (
                "scheduler_interval must be 'epoch', 'step', or 'time'. "
                f"Got: {interval}"
            )
            raise ValueError(msg)
        return interval

    def _as_positive_int(self, value: Any, field_name: str) -> int:
        """Convert value to a positive integer."""
        int_value = int(value)
        if int_value <= 0:
            msg = f"{field_name} must be a positive integer. Got: {value}"
            raise ValueError(msg)
        return int_value

    def _resolve_cosine_horizon(self, cfg: dict[str, Any], interval: str) -> int:
        """Resolve the cosine horizon used in the scheduler lambda."""
        explicit_key = "cosine_steps" if interval == "step" else "cosine_epochs"
        explicit_value = cfg.get(explicit_key)
        if explicit_value is not None:
            return self._as_positive_int(explicit_value, explicit_key)

        trainer = getattr(self, "trainer", None)
        if interval == "step" and trainer is not None:
            estimated_steps = getattr(trainer, "estimated_stepping_batches", None)
            if estimated_steps is not None:
                try:
                    estimated_int = int(estimated_steps)
                except (TypeError, ValueError):
                    estimated_int = 0
                if estimated_int > 0:
                    return estimated_int

        if trainer is not None and trainer.max_epochs is not None:
            return self._as_positive_int(trainer.max_epochs, "trainer.max_epochs")

        fallback = cfg.get("cosine_t_max", 1)
        return self._as_positive_int(fallback, "cosine_t_max")

    def _create_optimizer(
        self, cfg: DictConfig | dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        if not cfg.get("optimizer"):
            msg = "Optimizer name is required in optimizer_config."
            raise ValueError(msg)
        if cfg.get("learning_rate") is None:
            msg = "learning_rate is required in optimizer_config."
            raise ValueError(msg)
        if cfg.get("grad_clip") is not None:
            msg = (
                "optimizer.grad_clip is no longer read; gradient clipping is "
                "wired through trainer.gradient_clip_val. Move the value to "
                "trainer.gradient_clip_val (or set it to null to disable)."
            )
            raise ValueError(msg)
        optimizer_name = str(cfg.get("optimizer")).lower()
        lr = cfg.get("learning_rate")
        if not isinstance(lr, (float, int)):
            msg = "learning_rate must be a number in optimizer_config."
            raise TypeError(msg)
        lr = float(lr)
        weight_decay = cfg.get("weight_decay", 0.0)

        if optimizer_name == "adamw":
            betas = cfg.get("betas", [0.9, 0.999])
            return torch.optim.AdamW(
                self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
            )
        if optimizer_name == "adam":
            betas = cfg.get("betas", [0.9, 0.999])
            return torch.optim.Adam(
                self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
            )
        if optimizer_name == "sgd":
            momentum = cfg.get("momentum", 0.9)
            return torch.optim.SGD(
                self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        if optimizer_name == "psgd":
            return self._create_psgd(cfg, lr=lr, weight_decay=float(weight_decay))

        msg = f"Unsupported optimizer: {optimizer_name}"
        raise ValueError(msg)

    def _create_psgd(
        self, cfg: DictConfig | dict[str, Any], *, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        betas = cfg.get("betas", [0.9])
        beta = float(next(iter(betas)))

        precondition_frequency = float(cfg.get("precondition_frequency", 16))
        precondition_frequency_decay = float(
            cfg.get("precondition_frequency_decay", 0.999)
        )
        precondition_size = int(cfg.get("precondition_size", 4096))
        merge_dims = bool(cfg.get("merge_dims", False))

        preconditioner_update_probability = cfg.get("preconditioner_update_probability")
        if preconditioner_update_probability is None:
            preconditioner_update_probability = partial(
                self._precond_prob_schedule,
                min_prob=1.0 / max(precondition_frequency, 1.0),
                decay=precondition_frequency_decay,
            )

        return heavyball.ForeachCachedDelayedPSGDKron(
            self.parameters(),
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=precondition_size,
            merge_dims=merge_dims,
        )

    def _resolve_warmup(self, cfg: dict[str, Any], horizon: int) -> int:
        """Resolve warmup steps/epochs from config.

        A float in (0, 1) is treated as a fraction of the cosine horizon
        (e.g. 0.05 -> 5% of horizon). Anything else is interpreted as an
        absolute count. Negative values clamp to 0. NaN / inf are rejected.
        """
        warmup_raw = cfg.get("warmup", 0)
        if isinstance(warmup_raw, float) and not math.isfinite(warmup_raw):
            msg = f"warmup must be finite. Got: {warmup_raw}"
            raise ValueError(msg)
        if isinstance(warmup_raw, float) and 0.0 < warmup_raw < 1.0:
            warmup = int(warmup_raw * horizon)
        else:
            warmup = int(warmup_raw)
        return max(warmup, 0)

    def _resolve_time_warmup(self, cfg: dict[str, Any]) -> float:
        """Resolve warmup as a fraction of the wall-clock budget.

        With ``scheduler_interval='time'`` warmup is a fraction of
        ``trainer.max_time`` (a float in ``[0, 1)``); absolute step/epoch
        counts are meaningless against a wall-clock budget and are rejected.
        """
        warmup_cfg = cfg.get("warmup", 0)
        warmup_raw = 0.0 if warmup_cfg is None else float(warmup_cfg)
        if not math.isfinite(warmup_raw):
            msg = f"warmup must be finite. Got: {warmup_raw}"
            raise ValueError(msg)
        if not 0.0 <= warmup_raw < 1.0:
            msg = (
                "With scheduler_interval='time', warmup must be a fraction in "
                f"[0, 1) of the wall-clock budget. Got: {warmup_raw}"
            )
            raise ValueError(msg)
        return warmup_raw

    def _find_training_timer(self) -> Timer | None:
        """Return Lightning's Timer callback if one is installed."""
        trainer = getattr(self, "trainer", None)
        callbacks = getattr(trainer, "callbacks", []) if trainer is not None else []
        return next((cb for cb in callbacks if isinstance(cb, Timer)), None)

    def _require_time_budget(self) -> None:
        """Ensure a wall-clock budget exists for time-based scheduling.

        ``trainer.max_time`` installs Lightning's ``Timer`` with a duration
        (``max_time`` itself is not exposed as a trainer attribute). A Timer
        without a duration, or no Timer at all, means there is no budget to
        anneal over, so fail loud rather than silently never decaying.
        """
        timer = self._find_training_timer()
        if timer is None or timer.time_remaining() is None:
            msg = (
                "scheduler_interval='time' requires a wall-clock budget; set "
                "trainer.max_time (which installs Lightning's Timer)."
            )
            raise ValueError(msg)

    def _training_time_progress(self) -> float:
        """Fraction in [0, 1] of the wall-clock budget elapsed.

        Reads Lightning's ``Timer`` callback, whose elapsed offset is restored
        from checkpoints (and cleared by ``reset_resume_time_budget``). The
        schedule therefore tracks the same clock that enforces ``max_time``: a
        single-job run anneals over the job; a full-state-resumed run anneals
        over the cumulative training time. Returns 0.0 (schedule start) if no
        Timer / budget is available, which only happens before training begins.
        """
        timer = self._find_training_timer()
        if timer is None:
            return 0.0
        remaining = timer.time_remaining()
        if remaining is None:
            return 0.0
        elapsed = timer.time_elapsed()
        budget = elapsed + remaining
        if budget <= 0.0:
            return 0.0
        return min(max(elapsed / budget, 0.0), 1.0)

    def _create_time_cosine(
        self,
        optimizer: torch.optim.Optimizer,
        cfg: dict[str, Any],
        min_lr_ratio: float,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Cosine LR annealed over wall-clock time (``trainer.max_time``).

        The lambda ignores the step counter and instead reads elapsed training
        time from Lightning's ``Timer`` each step, so the LR reaches
        ``min_lr_ratio`` exactly as the budget is exhausted regardless of
        throughput. Stepped per optimizer step (see ``configure_optimizers``).
        """
        warmup_frac = self._resolve_time_warmup(cfg)

        def lr_lambda(_step: int) -> float:
            progress = self._training_time_progress()
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scaled = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
            if warmup_frac > 0.0 and progress < warmup_frac:
                return (progress / warmup_frac) * scaled
            return scaled

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer, cfg: dict[str, Any]
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler from config."""
        scheduler_name = str(cfg.get("scheduler", "")).lower()
        scheduler_interval = self._get_scheduler_interval(cfg)

        min_lr_ratio = float(cfg.get("min_lr_ratio", 0.0))
        if not 0.0 <= min_lr_ratio <= 1.0:
            msg = f"min_lr_ratio must be in [0, 1]. Got: {min_lr_ratio}"
            raise ValueError(msg)

        if scheduler_name in {"cosine", "cosine_with_restarts"}:
            if scheduler_interval == "time":
                if scheduler_name == "cosine_with_restarts":
                    msg = (
                        "cosine_with_restarts is not supported with "
                        "scheduler_interval='time'."
                    )
                    raise ValueError(msg)
                self._require_time_budget()
                return self._create_time_cosine(optimizer, cfg, min_lr_ratio)

            horizon = self._resolve_cosine_horizon(cfg, scheduler_interval)
            warmup = self._resolve_warmup(cfg, horizon)
            use_restarts = scheduler_name == "cosine_with_restarts"

            def cosine_lambda(t: int) -> float:
                phase_t = t % horizon if use_restarts else t
                cosine = 0.5 * (
                    1.0 + math.cos(math.pi * float(phase_t) / float(horizon))
                )
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            if warmup > 0:

                def lr_lambda(t: int) -> float:
                    warm = min(1.0, float(t + 1) / float(warmup + 1))
                    return warm * cosine_lambda(t)

            else:
                lr_lambda = cosine_lambda

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        if scheduler_name == "step":
            step_size = cfg.get("step_size", 30)
            gamma = cfg.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        if scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=10
            )
        msg = f"Unsupported scheduler: {scheduler_name}"
        raise ValueError(msg)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers for training."""
        if self.optimizer_config is None:
            msg = "optimizer_config is required for training."
            raise ValueError(msg)

        # Accept both plain dict and Hydra DictConfig
        cfg_any: Any = self.optimizer_config
        if not isinstance(cfg_any, dict):
            cfg_any = OmegaConf.to_container(cfg_any, resolve=True)
        if not isinstance(cfg_any, dict):
            msg = (
                "optimizer_config must be a mapping (dict-like). "
                f"Got: {type(cfg_any).__name__}"
            )
            raise TypeError(msg)
        cfg = cfg_any
        if not cfg:
            msg = "optimizer_config cannot be empty."
            raise ValueError(msg)

        optimizer = self._create_optimizer(cfg)
        scheduler_name = cfg.get("scheduler", None)

        # Return optimizer only if no scheduler
        if scheduler_name is None:
            return optimizer

        scheduler = self._create_scheduler(optimizer, cfg)
        scheduler_interval = self._get_scheduler_interval(cfg)

        # ReduceLROnPlateau needs special handling
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                },
            }

        # Time-based cosine steps per optimizer step (the lambda reads the
        # wall-clock; Lightning's lr_scheduler only accepts "step"/"epoch").
        lightning_interval = (
            "step" if scheduler_interval == "time" else scheduler_interval
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": lightning_interval,
                "frequency": 1,
            },
        }
