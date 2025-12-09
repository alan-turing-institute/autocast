from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from auto_cast.types import Tensor


class Processor(ABC, nn.Module):
    """Processor Base Class."""

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
    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        """Compute loss between output and target."""

    @abstractmethod
    def map(self, x: Tensor) -> Tensor:
        """Map input window of states/times to output window."""


class DiscreteProcessor(Processor, ABC):
    """DiscreteProcessor."""


class FlowBasedGenerativeProcessor(DiscreteProcessor):
    """Flow-based generative processor."""
