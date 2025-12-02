from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

from auto_cast.types import RolloutOutput, Tensor

BatchT = TypeVar("BatchT")


class RolloutMixin(ABC, Generic[BatchT]):
    """Rollout logic for generic batches."""

    stride: int
    max_rollout_steps: int
    teacher_forcing_ratio: float

    def rollout(self, batch: BatchT) -> RolloutOutput:
        pred_outs: list[Tensor] = []
        true_outs: list[Tensor] = []
        current_batch = self._clone_batch(batch)

        for _ in range(0, self.max_rollout_steps, self.stride):
            output = self._predict(current_batch)
            pred_outs.append(output)

            true_slice, should_record = self._true_slice(current_batch, self.stride)
            if should_record:
                true_outs.append(true_slice)

            rand_val = torch.rand(1, device=output.device).item()
            teacher_force = (
                true_slice.numel() > 0 and rand_val < self.teacher_forcing_ratio
            )
            next_inputs = true_slice if teacher_force else output.detach()

            if next_inputs.shape[1] < self.stride:
                break

            current_batch = self._advance_batch(current_batch, next_inputs, self.stride)

        predictions = torch.stack(pred_outs)
        if true_outs:
            return predictions, torch.stack(true_outs)
        return predictions, None

    @abstractmethod
    def _clone_batch(self, batch: BatchT) -> BatchT: ...

    @abstractmethod
    def _predict(self, batch: BatchT) -> Tensor: ...

    @abstractmethod
    def _true_slice(self, batch: BatchT, stride: int) -> tuple[Tensor, bool]: ...

    @abstractmethod
    def _advance_batch(
        self, batch: BatchT, next_inputs: Tensor, stride: int
    ) -> BatchT: ...
