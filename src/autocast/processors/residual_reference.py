"""Reference trajectories for residual forecasting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext

import torch
from einops import repeat
from torch import nn

from autocast.processors.base import Processor
from autocast.types import Tensor


class ReferenceTrajectory(nn.Module, ABC):
    """Build the trajectory about which a residual is modelled."""

    @abstractmethod
    def forward(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        """Return a reference trajectory conditioned on ``x``."""


class LastFrameReference(ReferenceTrajectory):
    """Repeat selected channels of the latest input frame over the horizon."""

    def __init__(
        self,
        *,
        n_steps_output: int,
        channel_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if n_steps_output < 1:
            msg = "n_steps_output must be positive."
            raise ValueError(msg)
        if channel_indices is not None and not channel_indices:
            msg = "channel_indices cannot be empty."
            raise ValueError(msg)
        self.n_steps_output = n_steps_output
        self.channel_indices = (
            tuple(channel_indices) if channel_indices is not None else None
        )

    def forward(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002
        """Repeat the last frame of a channels-last spatiotemporal input."""
        if x.ndim < 3 or x.shape[1] == 0:
            msg = "x must have shape (B, T, ..., C) with at least one timestep."
            raise ValueError(msg)
        frame = x[:, -1:]
        if self.channel_indices is not None:
            frame = frame[..., list(self.channel_indices)]
        return repeat(frame, "b 1 ... c -> b t ... c", t=self.n_steps_output)


class ProcessorReference(ReferenceTrajectory):
    """Use another processor as a reference trajectory generator."""

    def __init__(self, processor: Processor, *, frozen: bool = True) -> None:
        super().__init__()
        self.processor = processor
        self.frozen = frozen
        if frozen:
            self.processor.requires_grad_(False)
            self.processor.eval()

    def train(self, mode: bool = True) -> ProcessorReference:
        """Keep a frozen reference in evaluation mode."""
        super().train(mode)
        if self.frozen:
            self.processor.eval()
        return self

    def forward(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        """Delegate trajectory generation to the wrapped processor."""
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            return self.processor.map(x, global_cond)
