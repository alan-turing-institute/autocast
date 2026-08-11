"""Fit residual standardization statistics before training."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, cast

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.seed import isolate_rng
from torch import nn

from autocast.processors.residual_normalization import ResidualStandardizer
from autocast.types import Batch, EncodedBatch, Tensor

log = logging.getLogger(__name__)


class _ResidualProcessor(Protocol):
    standardizer: ResidualStandardizer | None
    reference: nn.Module

    def target_residual(self, batch: EncodedBatch) -> Tensor: ...


class _RunningMoments:
    """Streaming population moments over batches and spatial dimensions."""

    def __init__(self, standardizer: ResidualStandardizer) -> None:
        self.granularity = standardizer.granularity
        self.count = 0
        self.mean = torch.zeros(standardizer.statistic_shape, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)

    def update(self, residual: Tensor) -> None:
        if self.granularity == "channel":
            values = residual.reshape(-1, residual.shape[-1])
        else:
            values = residual.movedim(1, 0).reshape(
                residual.shape[1], -1, residual.shape[-1]
            )

        batch_count = values.shape[-2]
        if batch_count == 0:
            return
        batch_variance, batch_mean = torch.var_mean(
            values.detach().float(), dim=-2, correction=0
        )
        batch_mean = batch_mean.to(device="cpu", dtype=torch.float64)
        batch_m2 = batch_variance.to(device="cpu", dtype=torch.float64) * batch_count

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean.add_(delta * (batch_count / total_count))
        self.m2.add_(
            batch_m2 + delta.square() * (self.count * batch_count / total_count)
        )
        self.count = total_count

    def compute(self) -> tuple[Tensor, Tensor]:
        if self.count == 0:
            msg = "Cannot fit residual statistics from an empty training loader."
            raise ValueError(msg)
        return self.mean.float(), torch.sqrt(self.m2 / self.count).float()


class ResidualStatisticsCallback(Callback):
    """Fit residual statistics once from the full training split.

    The callback uses a processor's public ``target_residual`` method, so it is
    not tied to flow matching. It accepts ``EncodedBatch`` directly or ``Batch``
    through a frozen latent encoder. Rank zero fits streaming moments, broadcasts
    them to other ranks, and stores them in the standardizer checkpoint buffers.
    """

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Fit statistics unless they were restored from a checkpoint."""
        processor, standardizer = self._configuration(pl_module)
        needs_fit = not standardizer.fitted if trainer.is_global_zero else False
        if not trainer.strategy.broadcast(needs_fit, src=0):
            return

        if trainer.is_global_zero:
            mean, scale, count = self._fit_statistics(
                trainer, pl_module, processor, standardizer
            )
            standardizer.set_statistics(mean=mean, scale=scale)
            log.info("Fitted residual statistics from %d values per statistic.", count)

        mean = trainer.strategy.broadcast(standardizer.mean.detach().clone(), src=0)
        scale = trainer.strategy.broadcast(standardizer.scale.detach().clone(), src=0)
        standardizer.set_statistics(mean=mean, scale=scale)

    @staticmethod
    def _configuration(
        pl_module: L.LightningModule,
    ) -> tuple[_ResidualProcessor, ResidualStandardizer]:
        processor = getattr(pl_module, "processor", None)
        if not callable(getattr(processor, "target_residual", None)):
            msg = "model.processor must expose target_residual(batch)."
            raise TypeError(msg)
        standardizer = getattr(processor, "standardizer", None)
        if not isinstance(standardizer, ResidualStandardizer):
            msg = "model.processor.standardizer must be a ResidualStandardizer."
            raise TypeError(msg)
        reference = getattr(processor, "reference", None)
        if not isinstance(reference, nn.Module):
            msg = "model.processor.reference must be an nn.Module."
            raise TypeError(msg)
        if any(parameter.requires_grad for parameter in reference.parameters()):
            msg = "Residual statistics require a fixed, non-trainable reference."
            raise ValueError(msg)
        return cast(_ResidualProcessor, processor), standardizer

    def _fit_statistics(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        processor: _ResidualProcessor,
        standardizer: ResidualStandardizer,
    ) -> tuple[Tensor, Tensor, int]:
        for attribute in ("noise_injector", "input_noise_injector"):
            if getattr(pl_module, attribute, None) is not None:
                msg = (
                    "Residual statistics do not yet support model input noise; "
                    f"{attribute} must be None."
                )
                raise ValueError(msg)

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            msg = "ResidualStatisticsCallback requires a trainer datamodule."
            raise ValueError(msg)
        encoder = self._frozen_encoder(pl_module)
        modules = [processor.reference, *([encoder] if encoder is not None else [])]
        training_modes = [module.training for module in modules]
        moments = _RunningMoments(standardizer)

        try:
            for module in modules:
                module.eval()
            with isolate_rng(), torch.inference_mode():
                for batch in datamodule.train_dataloader():
                    encoded_batch = self._encode_batch(
                        batch, encoder, standardizer.mean.device
                    )
                    moments.update(processor.target_residual(encoded_batch))
        finally:
            for module, mode in zip(modules, training_modes, strict=True):
                module.train(mode)

        mean, scale = moments.compute()
        return mean, scale, moments.count

    @staticmethod
    def _frozen_encoder(pl_module: L.LightningModule) -> nn.Module | None:
        encoder = getattr(getattr(pl_module, "encoder_decoder", None), "encoder", None)
        if encoder is None:
            return None
        if not getattr(pl_module, "train_in_latent_space", False):
            msg = "Raw Batch statistics require train_in_latent_space=True."
            raise ValueError(msg)
        if any(parameter.requires_grad for parameter in encoder.parameters()):
            msg = "Latent residual statistics require a frozen encoder."
            raise ValueError(msg)
        if not callable(getattr(encoder, "encode_batch", None)):
            msg = "The latent encoder must expose encode_batch(batch)."
            raise TypeError(msg)
        return cast(nn.Module, encoder)

    @staticmethod
    def _encode_batch(
        batch: object,
        encoder: nn.Module | None,
        device: torch.device,
    ) -> EncodedBatch:
        if isinstance(batch, EncodedBatch):
            return batch.to(device)
        if not isinstance(batch, Batch):
            msg = f"Training loader yielded unsupported {type(batch).__name__}."
            raise TypeError(msg)
        if encoder is None:
            msg = "Training loader yielded Batch without a frozen latent encoder."
            raise TypeError(msg)
        encode_batch = cast(
            Callable[[Batch], EncodedBatch], encoder.encode_batch  # type: ignore[attr-defined]
        )
        return encode_batch(batch.to(device))
