from abc import ABC, abstractmethod
from typing import Any, Generic

from torch import nn

from autocast.types import BatchT, Tensor


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
        residual: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize processor configuration.

        Parameters
        ----------
        stride:
            Number of time steps to advance per rollout step.
        teacher_forcing_ratio:
            Probability of using ground-truth windows during rollout.
        max_rollout_steps:
            Maximum number of rollout steps to unroll.
        loss_func:
            Optional loss module; defaults to MSELoss.
        residual:
            If True, models predict residuals w.r.t. the last input state.
        **kwargs:
            Additional attributes to set on the instance.
        """
        super().__init__()
        self.stride = stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.loss_func = loss_func or nn.MSELoss()
        self.residual = residual
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def loss(self, batch: BatchT) -> Tensor:
        """Compute loss between output and target."""

    @abstractmethod
    def map(self, x: Tensor) -> Tensor:
        """
        Map input states to output states.

        Parameters
        ----------
        x:
            Input tensor of shape (B, T_in, ...).

        Returns
        -------
        Tensor
            Output tensor of shape (B, T_out, ...).
        """

    def _apply_residual_output(self, x: Tensor, y_pred: Tensor) -> Tensor:
        """
        Convert model-predicted residuals to absolute outputs when enabled.

        Parameters
        ----------
        x:
            Input tensor of shape (B, T_in, ...).
        y_pred:
            Predicted tensor of shape (B, T_out, ...), residuals or absolutes.

        Returns
        -------
        Tensor
            Absolute outputs of shape (B, T_out, ...).
        """
        if not self.residual:
            return y_pred
        base = x[:, -1:, ...]
        if base.shape[1] != y_pred.shape[1]:
            base = base.expand(x.shape[0], y_pred.shape[1], *base.shape[2:])
        return base + y_pred

    def _residualize_target(self, x: Tensor, target: Tensor) -> Tensor:
        """
        Shift targets into residual space when residual mode is enabled.

        Parameters
        ----------
        x:
            Input tensor of shape (B, T_in, ...).
        target:
            Target tensor of shape (B, T_out, ...).

        Returns
        -------
        Tensor
            Residual targets if residual mode, otherwise the original targets.
        """
        if not self.residual:
            return target
        base = x[:, -1:, ...]
        if base.shape[1] != target.shape[1]:
            base = base.expand(x.shape[0], target.shape[1], *base.shape[2:])
        return target - base
