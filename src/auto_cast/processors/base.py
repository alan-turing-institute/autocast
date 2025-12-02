from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch
from torch import nn

from auto_cast.processors.rollout import RolloutMixin
from auto_cast.types import EncodedBatch, RolloutOutput, Tensor


class Processor(RolloutMixin[EncodedBatch], L.LightningModule):
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

    def forward(self, *args, **kwargs: Any) -> Any:
        """Forward pass through the Processor."""
        msg = "To implement."
        raise NotImplementedError(msg)

    def training_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor:  # noqa: ARG002
        output = self.map(batch.encoded_inputs)
        loss = self.loss_func(output, batch.encoded_output_fields)
        return loss  # noqa: RET504

    @abstractmethod
    def map(self, x: Tensor) -> Tensor:
        """Map input window of states/times to output window."""

    def configure_optimizers(self): ...

    def _clone_batch(self, batch: EncodedBatch) -> EncodedBatch:
        return EncodedBatch(
            encoded_inputs=batch.encoded_inputs.clone(),
            encoded_output_fields=batch.encoded_output_fields.clone(),
            encoded_info={
                key: value.clone() if hasattr(value, "clone") else value
                for key, value in batch.encoded_info.items()
            },
        )

    def _predict(self, batch: EncodedBatch) -> Tensor:
        return self.map(batch.encoded_inputs)

    def _true_slice(self, batch: EncodedBatch, stride: int) -> tuple[Tensor, bool]:
        if batch.encoded_output_fields.shape[1] >= stride:
            return batch.encoded_output_fields[:, :stride, ...], True
        return batch.encoded_output_fields, False

    def _advance_batch(
        self, batch: EncodedBatch, next_inputs: Tensor, stride: int
    ) -> EncodedBatch:
        next_inputs = torch.cat(
            [batch.encoded_inputs[:, stride:, ...], next_inputs[:, :stride, ...]],
            dim=1,
        )
        next_outputs = (
            batch.encoded_output_fields[:, stride:, ...]
            if batch.encoded_output_fields.shape[1] > stride
            else batch.encoded_output_fields[:, 0:0, ...]
        )
        return EncodedBatch(
            encoded_inputs=next_inputs,
            encoded_output_fields=next_outputs,
            encoded_info=batch.encoded_info,
        )


class DiscreteProcessor(Processor, ABC):
    """DiscreteProcessor."""

    @abstractmethod
    def map(self, x: Tensor) -> Tensor:
        ...
        # Map input window of states/times to output window

    def rollout(self, batch: EncodedBatch) -> RolloutOutput:
        ...

        # Use self.map to generate trajectory


class FlowBasedGenerativeProcessor(DiscreteProcessor):
    """Flow-based generative processor."""

    def map(self, x: Tensor) -> Tensor:
        ...
        # Sample generative model    def loss(self, ...):...
        # Flow matc
