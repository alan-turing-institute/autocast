from collections.abc import Sequence  # noqa: EXE002
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import OmegaConf
from torch import nn
from torchmetrics import Metric

from autocast.decoders import Decoder
from autocast.encoders import Encoder
from autocast.metrics.utils import MetricsMixin
from autocast.types import Batch, Tensor, TensorBNC, TensorBTSC


class EncoderDecoder(L.LightningModule, MetricsMixin):
    """Encoder-Decoder Model."""

    encoder: Encoder
    decoder: Decoder
    loss_func: nn.Module | None
    learning_rate: float = 1e-3

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        loss_func: nn.Module | None = None,
        learning_rate: float = 1e-3,
        optimizer_config: dict[str, Any] | None = None,
        train_metrics: Sequence[Metric] | None = [],
        val_metrics: Sequence[Metric] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.optimizer_config = optimizer_config
        self.train_metrics = self._build_metrics(train_metrics, "train_")
        self.val_metrics = self._build_metrics(val_metrics, "val_")
        self.test_metrics = self._build_metrics(test_metrics, "test_")

    def forward(self, batch: Batch) -> TensorBTSC:
        return self.decoder(self.encoder(batch))

    def forward_with_latent(self, batch: Batch) -> tuple[TensorBTSC, TensorBNC]:
        encoded = self.encode(batch)
        decoded = self.decode(encoded)
        return decoded, encoded

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        if self.loss_func is None:
            msg = "Loss function not defined for EncoderDecoder model."
            raise ValueError(msg)
        x = self(batch)
        y_pred = self.decoder(x)
        y_true = batch.output_fields
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        self._update_and_log_metrics(
            self, self.train_metrics, y_pred, y_true, batch.input_fields.shape[0]
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        if self.loss_func is None:
            msg = "Loss function not defined for EncoderDecoder model."
            raise ValueError(msg)
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        self._update_and_log_metrics(
            self, self.val_metrics, y_pred, y_true, batch.input_fields.shape[0]
        )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        if self.loss_func is None:
            msg = "Loss function not defined for EncoderDecoder model."
            raise ValueError(msg)
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "test_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        self._update_and_log_metrics(
            self, self.test_metrics, y_pred, y_true, batch.input_fields.shape[0]
        )
        return loss

    def predict_step(self, batch: Batch, batch_idx: int) -> TensorBTSC:  # noqa: ARG002
        return self(batch)

    def encode(self, batch: Batch) -> TensorBNC:
        return self.encoder.encode(batch)

    def decode(self, z: TensorBNC) -> TensorBTSC:
        return self.decoder.decode(z)

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
            if (
                getattr(self, "trainer", None) is not None
                and self.trainer.max_epochs is not None
            ):
                max_epochs = int(self.trainer.max_epochs)
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


class VAE(EncoderDecoder):
    """Variational Autoencoder Model."""

    def forward(self, batch: Batch) -> Tensor:
        mu, log_var = self.encoder(batch)
        z = self.reparametrize(mu, log_var)
        x = self.decoder(z)
        return x  # noqa: RET504

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
