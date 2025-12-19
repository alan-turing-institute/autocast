"""Optimizer configuration mixin for Lightning modules."""

from typing import Any

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import OmegaConf
from torch import nn


class OptimizerMixin(nn.Module):
    """Mixin class providing optimizer configuration for Lightning modules.

    Inherits from nn.Module to ensure parameters() method is available.
    Requires the class to have:
        - self.learning_rate: float
        - self.optimizer_config: dict[str, Any] | None
        - self.trainer: Lightning Trainer instance (optional, for scheduler)
    """

    # Type hints for attributes expected from the concrete class
    learning_rate: float
    optimizer_config: dict[str, Any] | None

    def _create_optimizer(self, cfg: dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        optimizer_name = str(cfg.get("optimizer", "adam")).lower()
        lr = cfg.get("learning_rate", self.learning_rate)
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
        msg = f"Unsupported optimizer: {optimizer_name}"
        raise ValueError(msg)

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
        # Backwards compatibility: if no optimizer_config, use simple Adam
        if self.optimizer_config is None:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
