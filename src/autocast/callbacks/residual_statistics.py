"""Fit residual standardization statistics before training."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sized
from typing import Protocol, cast

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.data import _update_dataloader
from lightning.pytorch.utilities.seed import isolate_rng
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)

from autocast.processors.residual_normalization import ResidualNormalizer
from autocast.types import Batch, EncodedBatch, Tensor

log = logging.getLogger(__name__)


class _ResidualProcessor(Protocol):
    residual_normalizer: ResidualNormalizer | None
    reference: nn.Module

    def target_residual(self, batch: EncodedBatch) -> Tensor: ...


class _RunningMoments:
    """Streaming population moments over batches and spatial dimensions."""

    def __init__(self, residual_normalizer: ResidualNormalizer) -> None:
        self.granularity = residual_normalizer.granularity
        self.count = 0
        self.mean = torch.zeros(
            residual_normalizer.statistic_shape, dtype=torch.float64
        )
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
        # MPS cannot represent float64; transfer before widening the dtype.
        batch_mean = batch_mean.cpu().double()
        batch_m2 = batch_variance.cpu().double() * batch_count

        self.merge(batch_count, batch_mean, batch_m2)

    def merge(self, count: int, mean: Tensor, m2: Tensor) -> None:
        if count == 0:
            return
        total_count = self.count + count
        delta = mean - self.mean
        self.mean.add_(delta * (count / total_count))
        self.m2.add_(m2 + delta.square() * (self.count * count / total_count))
        self.count = total_count

    def synchronize(self, trainer: L.Trainer, device: torch.device) -> None:
        summary = torch.cat(
            (self.mean.flatten(), self.m2.flatten(), self.mean.new_tensor([self.count]))
        )
        summaries = trainer.strategy.all_gather(summary.to(device)).cpu()
        self.count = 0
        self.mean.zero_()
        self.m2.zero_()
        for summary in summaries:
            mean, m2 = summary[:-1].chunk(2)
            self.merge(
                int(summary[-1].item()), mean.view_as(self.mean), m2.view_as(self.m2)
            )

    def compute(self) -> tuple[Tensor, Tensor]:
        if self.count == 0:
            msg = "Cannot fit residual statistics from an empty training loader."
            raise ValueError(msg)
        return self.mean.float(), torch.sqrt(self.m2 / self.count).float()


class ResidualStatisticsCallback(Callback):
    """Fit residual statistics once from the full training split.

    The callback uses a processor's public ``target_residual`` method, so it is
    not tied to flow matching. It accepts ``EncodedBatch`` directly or ``Batch``
    through a frozen latent encoder. Distributed ranks fit disjoint, unpadded
    dataset shards and combine moments before updating checkpoint buffers.
    """

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Fit statistics unless they were restored from a checkpoint."""
        processor, residual_normalizer, encoder = self._configuration(
            trainer, pl_module
        )
        needs_fit = not residual_normalizer.fitted if trainer.is_global_zero else False
        if not trainer.strategy.broadcast(needs_fit, src=0):
            return

        moments = self._fit_statistics(trainer, processor, residual_normalizer, encoder)
        if trainer.world_size > 1:
            moments.synchronize(trainer, residual_normalizer.mean.device)
        mean, scale = moments.compute()
        residual_normalizer.set_statistics(mean=mean, scale=scale)
        if trainer.is_global_zero:
            log.info(
                "Fitted residual statistics from %d values per statistic.",
                moments.count,
            )

    def _configuration(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> tuple[_ResidualProcessor, ResidualNormalizer, nn.Module | None]:
        processor = getattr(pl_module, "processor", None)
        if not callable(getattr(processor, "target_residual", None)):
            msg = "model.processor must expose target_residual(batch)."
            raise TypeError(msg)
        residual_normalizer = getattr(processor, "residual_normalizer", None)
        if not isinstance(residual_normalizer, ResidualNormalizer):
            msg = "model.processor.residual_normalizer must be a ResidualNormalizer."
            raise TypeError(msg)
        reference = getattr(processor, "reference", None)
        if not isinstance(reference, nn.Module):
            msg = "model.processor.reference must be an nn.Module."
            raise TypeError(msg)
        if any(parameter.requires_grad for parameter in reference.parameters()):
            msg = "Residual statistics require a fixed, non-trainable reference."
            raise ValueError(msg)
        encoder = None
        if not residual_normalizer.fitted:
            for attribute in ("noise_injector", "input_noise_injector"):
                if getattr(pl_module, attribute, None) is not None:
                    msg = (
                        "Residual statistics do not yet support model input noise; "
                        f"{attribute} must be None."
                    )
                    raise ValueError(msg)
            if getattr(trainer, "datamodule", None) is None:
                msg = "ResidualStatisticsCallback requires a trainer datamodule."
                raise ValueError(msg)
            encoder = self._frozen_encoder(pl_module)
        return cast(_ResidualProcessor, processor), residual_normalizer, encoder

    def _fit_statistics(
        self,
        trainer: L.Trainer,
        processor: _ResidualProcessor,
        residual_normalizer: ResidualNormalizer,
        encoder: nn.Module | None,
    ) -> _RunningMoments:
        modules = [processor.reference, *([encoder] if encoder is not None else [])]
        training_modes = [module.training for module in modules]
        moments = _RunningMoments(residual_normalizer)

        try:
            for module in modules:
                module.eval()
            with isolate_rng(), torch.inference_mode():
                datamodule = getattr(trainer, "datamodule", None)
                assert datamodule is not None
                loader = datamodule.train_dataloader()
                if trainer.world_size > 1:
                    if (
                        not isinstance(loader, DataLoader)
                        or isinstance(loader.dataset, IterableDataset)
                        or not isinstance(loader.dataset, Sized)
                        or loader.drop_last
                        or type(loader.batch_sampler) not in (type(None), BatchSampler)
                        # Explicit stock BatchSamplers would lose batching when
                        # Lightning reconstructs a loader with batch_size=None.
                        or (
                            loader.batch_sampler is not None
                            and loader.batch_size is None
                        )
                        or type(loader.sampler)
                        not in (
                            RandomSampler,
                            SequentialSampler,
                            DistributedSampler,
                        )
                        or (
                            isinstance(loader.sampler, RandomSampler)
                            and (
                                loader.sampler.replacement
                                or len(loader.sampler) != len(loader.dataset)
                            )
                        )
                    ):
                        msg = (
                            "Distributed residual fitting requires a finite map-style "
                            "DataLoader with standard samplers, implicit batching "
                            "and drop_last=False; "
                            "provide fixed statistics."
                        )
                        raise ValueError(msg)
                    # Lightning preserves batching/collation/worker options. A range
                    # partitions indices without DistributedSampler's padded duplicates.
                    loader = _update_dataloader(
                        loader,
                        range(
                            trainer.global_rank, len(loader.dataset), trainer.world_size
                        ),
                    )
                for batch in loader:
                    # Match training's input dtype and autocast/default-dtype
                    # context for both the encoder and the reference processor.
                    converted_batch = trainer.precision_plugin.convert_input(batch)
                    with trainer.precision_plugin.forward_context():
                        encoded_batch = self._encode_batch(
                            converted_batch, encoder, residual_normalizer.mean.device
                        )
                        residual = processor.target_residual(encoded_batch)
                    # Accumulate moments in FP32/CPU FP64, outside autocast.
                    moments.update(residual)
        finally:
            for module, mode in zip(modules, training_modes, strict=True):
                module.train(mode)

        return moments

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
            Callable[[Batch], EncodedBatch],
            encoder.encode_batch,  # type: ignore[attr-defined]
        )
        return encode_batch(batch.to(device))
