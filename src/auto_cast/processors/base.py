from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch
from torch import nn

from auto_cast.types import EncodedBatch, RolloutOutput, Tensor


class Processor(ABC, L.LightningModule):
    """Processor Base Class."""

    teacher_forcing_ratio: float
    stride: int
    max_rollout_steps: int
    loss_func: nn.Module
    learning_rate: float

    def forward(self, *args, **kwargs: Any) -> Any:
        """Forward pass through the Processor."""
        msg = "To implement."
        raise NotImplementedError(msg)

    def training_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor:  # noqa: ARG002
        output = self.map(batch.encoded_inputs)
        loss = self.loss_func(output, batch.encoded_output_fields)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.encoded_inputs.shape[0]
        )
        return loss

    @abstractmethod
    def map(self, x: Tensor) -> Tensor:
        """Map input window of states/times to output window."""

    def validation_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor:  # noqa: ARG002
        output = self.map(batch.encoded_inputs)
        loss = self.loss_func(output, batch.encoded_output_fields)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.encoded_inputs.shape[0]
        )
        return loss

    def configure_optimizers(self):
        """Configure optimizers for training.

        Returns Adam optimizer with learning_rate. Subclasses can override
        to use different optimizers or learning rate schedules.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def rollout(self, batch: EncodedBatch) -> RolloutOutput:
        """Rollout over multiple time steps."""
        pred_outs, gt_outs = [], []
        for _ in range(0, self.max_rollout_steps, self.stride):
            pred_outs.append(self.map(batch.encoded_inputs))
            # TODO: combining teacher forcing logic
            gt_outs.append(
                batch.encoded_output_fields
            )  # This assumes we have output fields
        return torch.stack(pred_outs), torch.stack(gt_outs)


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
