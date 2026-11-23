"""Opt-in in-place compilation for the spatial ViT training backbone."""

import logging

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from autocast.nn.vit import SpatialViTBackbone
from autocast.processors.azula_vit import AzulaViTProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor

log = logging.getLogger(__name__)


class CompileViTBackbone(Callback):
    """Compile the backbone after checkpoint restoration and EMA setup."""

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        processor = getattr(pl_module, "processor", None)
        if isinstance(processor, AzulaViTProcessor):
            backbone = processor.model
        elif isinstance(processor, FlowMatchingProcessor):
            backbone = processor.flow_matching_model
        else:
            backbone = None

        if not isinstance(backbone, SpatialViTBackbone):
            msg = "compile_backbone=true requires an Azula spatial ViT backbone."
            raise ValueError(msg)
        # Keep the parameter references and state_dict names used by DDP and EMA.
        backbone.compile(backend="inductor")
        if trainer.is_global_zero:
            log.info("Enabled torch.compile for the spatial ViT training backbone.")
