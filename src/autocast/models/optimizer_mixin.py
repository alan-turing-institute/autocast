"""Optimizer configuration mixin for Lightning modules."""

from typing import Any

import heavyball
import torch
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
        betas = cfg.get("betas", None)
        if betas is not None:
            beta = float(next(iter(betas)))
        else:
            beta = float(cfg.get("beta", 0.9))

        precondition_frequency = cfg.get("precondition_frequency", None)
        if precondition_frequency is not None:
            precondition_frequency = float(precondition_frequency)
            preconditioner_update_probability = 1.0 / max(precondition_frequency, 1.0)
        else:
            preconditioner_update_probability = cfg.get(
                "preconditioner_update_probability", None
            )

        precondition_frequency_decay = cfg.get("precondition_frequency_decay", None)
        if precondition_frequency_decay is not None:
            # HeavyBall uses a stochastic update schedule when enabled; we keep this
            # value in config for parity with LOLA but do not currently map it.
            pass

        precondition_size = cfg.get("precondition_size", None)
        if precondition_size is not None:
            max_size_triangular = int(precondition_size)
        else:
            max_size_triangular = int(cfg.get("max_size_triangular", 2048))

        min_ndim_triangular = int(cfg.get("min_ndim_triangular", 2))
        memory_save_mode = cfg.get("memory_save_mode", None)
        momentum_into_precond_update = bool(
            cfg.get("momentum_into_precond_update", True)
        )
        warmup_steps = int(cfg.get("warmup", 1))
        merge_dims = bool(cfg.get("merge_dims", False))
        split = bool(cfg.get("split", False))
        store_triu_as_line = bool(cfg.get("store_triu_as_line", True))
        foreach = bool(cfg.get("foreach", True))
        q_dtype = str(cfg.get("q_dtype", "float32"))
        stochastic_schedule = bool(cfg.get("stochastic_schedule", True))
        storage_dtype = str(cfg.get("storage_dtype", "float32"))
        precond_init_scale = float(cfg.get("precond_init_scale", 1.0))
        precond_lr = float(cfg.get("precond_lr", 0.1))

        return heavyball.ForeachPSGDKron(
            self.parameters(),
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            warmup_steps=warmup_steps,
            merge_dims=merge_dims,
            split=split,
            store_triu_as_line=store_triu_as_line,
            foreach=foreach,
            q_dtype=q_dtype,
            stochastic_schedule=stochastic_schedule,
            storage_dtype=storage_dtype,
            precond_init_scale=precond_init_scale,
            precond_lr=precond_lr,
        )

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer, cfg: dict[str, Any]
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler from config."""
        scheduler_name = str(cfg.get("scheduler", "")).lower()

        if scheduler_name == "cosine":
            max_epochs = 1
            trainer = getattr(self, "trainer", None)
            if trainer is not None and trainer.max_epochs is not None:
                max_epochs = int(trainer.max_epochs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=0
            )
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

        # ReduceLROnPlateau needs special handling
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
