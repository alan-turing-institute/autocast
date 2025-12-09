from abc import ABC, abstractmethod
from typing import Any, Generic

from torch import nn

from auto_cast.types import BatchT, Tensor


class Processor(ABC, nn.Module, Generic[BatchT]):
    """Processor Base Class."""

    learning_rate: float

    def __init__(
        self,
        *,
        stride: int = 1,
        teacher_forcing_ratio: float = 0.0,
        max_rollout_steps: int = 1,
        loss_func: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.loss_func = loss_func or nn.MSELoss()
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def loss(self, batch: BatchT) -> Tensor:
        """Compute loss between output and target."""

    @abstractmethod
    def map(self, x: Tensor) -> Tensor:
        '''
        Maps input states to output states.

        Args:
            x (Tensor): Input tensor of shape (B, T_in, ...)
        Returns:
            y (Tensor): Output tensor of shape (B, T_out, ...)
        '''


