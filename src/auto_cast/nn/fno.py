from typing import Any, Protocol, runtime_checkable

import torch
from neuralop.models import FNO
from torch import nn

from auto_cast.processors.base import Processor
from auto_cast.types import Tensor


@runtime_checkable
class _HasGridCache(Protocol):
    _grid: Any | None
    _res: Any | None


class FNOProcessor(Processor):
    """Fourier Neural Operator Module.

    A discrete processor that uses a Fourier Neural Operator (FNO) to learn
    mappings between function spaces for spatiotemporal prediction.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    n_modes: tuple[int, ...]
        Number of Fourier modes to keep in each spatial dimension.
    hidden_channels: int, optional
        Width of the FNO (number of channels in hidden layers). Default is 64.
    n_layers: int, optional
        Number of FNO layers. Default is 4.
    channels: tuple[int, ...], optional
        Which channels from input_fields to use. Default is (0,).
    with_constants: bool, optional
        Whether to include constant fields in input. Default is False.
    with_time: bool, optional
        Whether to include time information. Default is False.
    n_steps_output: int, optional
        Number of output time steps. Default is 1.
    loss_fn: nn.Module, optional
        Loss function. Defaults to MSELoss.
    learning_rate: float, optional
        Learning rate for optimizer. Default is 1e-3.
    **fno_kwargs
        Additional keyword arguments passed to the FNO model.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        hidden_channels: int = 64,
        n_layers: int = 4,
        loss_func: nn.Module | None = None,
        learning_rate: float = 1e-3,
        stride: int = 1,
        max_rollout_steps: int = 10,
        **fno_kwargs: Any,
    ):
        super().__init__()

        self.model = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **fno_kwargs,
        )
        self.loss_func = loss_func or nn.MSELoss()
        self.learning_rate = learning_rate
        self.stride = stride
        self.max_rollout_steps = max_rollout_steps
        self._reset_positional_embedding_cache()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _reset_positional_embedding_cache(self) -> None:
        embedding = getattr(self.model, "positional_embedding", None)
        if isinstance(embedding, _HasGridCache):
            embedding._grid = None
            embedding._res = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        # Invalidate cached grids so they'll be regenerated on the new device/dtype.
        self._reset_positional_embedding_cache()
        return self

    def map(self, x: Tensor) -> Tensor:
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
