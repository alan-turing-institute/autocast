from typing import TYPE_CHECKING

from autocast.processors.azula_vit import AzulaViTProcessor
from autocast.processors.base import Processor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.swin_vit import SwinViTProcessor
from autocast.processors.unet import UNetProcessor

# ZukoFlowProcessor depends on the optional ``zuko`` package. Import it lazily so
# the processors package stays importable when zuko is not installed;
# instantiating the processor without zuko then raises a clear ImportError.
if TYPE_CHECKING:
    from autocast.processors.zuko_flow import ZukoFlowProcessor
else:
    try:
        from autocast.processors.zuko_flow import ZukoFlowProcessor
    except ImportError as exc:
        # The ``as`` target is cleared at the end of the except block, so bind
        # the error to a name that survives for the placeholder to chain from.
        _zuko_import_error = exc

        class ZukoFlowProcessor:
            """Placeholder when the optional ``zuko`` dependency is unavailable."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                msg = (
                    "ZukoFlowProcessor requires the optional 'zuko' package; "
                    "install zuko (the 'flows' extra) to use zuko flow processors."
                )
                raise ImportError(msg) from _zuko_import_error


__all__ = [
    "AzulaViTProcessor",
    "FlowMatchingProcessor",
    "Processor",
    "SwinViTProcessor",
    "UNetProcessor",
    "ZukoFlowProcessor",
]
