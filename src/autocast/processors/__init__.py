from autocast.processors.azula_vit import AzulaViTProcessor
from autocast.processors.base import Processor
from autocast.processors.conv_coupling import ConvCouplingFlowProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.flow_matching_masked_window import (
    FlowMatchingMaskedWindowProcessor,
)
from autocast.processors.residual_reference import (
    LastFrameReference,
    ProcessorReference,
    ReferenceTrajectory,
)
from autocast.processors.sigma_vae import SigmaVAEProcessor
from autocast.processors.swin_vit import SwinViTProcessor
from autocast.processors.tarflow import TarFlowProcessor
from autocast.processors.unet import UNetProcessor

__all__ = [
    "AzulaViTProcessor",
    "ConvCouplingFlowProcessor",
    "FlowMatchingMaskedWindowProcessor",
    "FlowMatchingProcessor",
    "LastFrameReference",
    "Processor",
    "ProcessorReference",
    "ReferenceTrajectory",
    "SigmaVAEProcessor",
    "SwinViTProcessor",
    "TarFlowProcessor",
    "UNetProcessor",
]
