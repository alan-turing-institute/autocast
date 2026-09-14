from typing import TYPE_CHECKING

from autocast.processors.azula_vit import (
    AzulaViTProcessor,
    MCDropoutAzulaViTProcessor,
)
from autocast.processors.base import Processor
from autocast.processors.consistency import ConsistencyDistilledProcessor
from autocast.processors.conv_coupling import ConvCouplingFlowProcessor
from autocast.processors.distilled import DistilledProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.flow_matching_masked_window import (
    FlowMatchingMaskedWindowProcessor,
)
from autocast.processors.sigma_vae import SigmaVAEProcessor
from autocast.processors.swin_vit import SwinViTProcessor
from autocast.processors.tarflow import TarFlowProcessor
from autocast.processors.unet import UNetProcessor

# NormalizingFlowProcessor depends on the optional ``zuko`` package. Import it
# lazily so the processors package stays importable when zuko is not installed;
# instantiating the processor without zuko then raises a clear ImportError.
if TYPE_CHECKING:
    from autocast.processors.normalizing_flow import NormalizingFlowProcessor
else:
    try:
        from autocast.processors.normalizing_flow import NormalizingFlowProcessor
    except ImportError as exc:
        # The ``as`` target is cleared at the end of the except block, so bind
        # the error to a name that survives for the placeholder to chain from.
        _zuko_import_error = exc

        class NormalizingFlowProcessor:
            """Placeholder when the optional ``zuko`` dependency is unavailable."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                msg = (
                    "NormalizingFlowProcessor requires the optional 'zuko' "
                    "package; install zuko to use normalizing-flow processors."
                )
                raise ImportError(msg) from _zuko_import_error


__all__ = [
    "AzulaViTProcessor",
    "ConsistencyDistilledProcessor",
    "ConvCouplingFlowProcessor",
    "DistilledProcessor",
    "FlowMatchingMaskedWindowProcessor",
    "FlowMatchingProcessor",
    "MCDropoutAzulaViTProcessor",
    "NormalizingFlowProcessor",
    "Processor",
    "SigmaVAEProcessor",
    "SwinViTProcessor",
    "TarFlowProcessor",
    "UNetProcessor",
]
