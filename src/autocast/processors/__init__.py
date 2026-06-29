from autocast.processors.azula_vit import AzulaViTProcessor
from autocast.processors.base import Processor
from autocast.processors.consistency import ConsistencyDistilledProcessor
from autocast.processors.distilled import DistilledProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.swin_vit import SwinViTProcessor
from autocast.processors.unet import UNetProcessor

__all__ = [
    "AzulaViTProcessor",
    "ConsistencyDistilledProcessor",
    "DistilledProcessor",
    "FlowMatchingProcessor",
    "Processor",
    "SwinViTProcessor",
    "UNetProcessor",
]
