from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import torch
from einops import rearrange

from auto_cast.types import RolloutOutput, Tensor
from auto_cast.types.batch import BatchT


class RolloutMixin(ABC, Generic[BatchT]):
    """Rollout logic for generic batches."""

    stride: int
    max_rollout_steps: int
    teacher_forcing_ratio: float

    def rollout(
        self, batch: BatchT, free_running_only: bool = False, return_windows=False
    ) -> RolloutOutput:
        """Perform rollout over multiple time steps.

        Parameters
        ----------
        batch: BatchT
            Input batch containing initial data for rollout.
        free_running_only: bool, optional
            If True, disables teacher forcing during rollout. By default False.
        return_windows: bool, optional
            If True, returns the true outputs in windows matching the model's output
            shape. By default False.


        Notes
        -----
        The outputs stack along a new axis after batch representing number of rollout
        windows R. Each window R contains n_steps_output time steps T.
        For example with:
        - batch size B=16
        - rollout windows R=10
        - n_steps_output T=2 per window,
        - spatial dimensions W=16, H=8
        - channels C=2

        The shapes will be:
          (B, R, T, W, H, C) = (16, 10, 2, 16, 8, 2)

        If we do not return windows, we then rearrange to concatenate the windows
        along time:
          (B, T*T, W, H, C) = (16, 20, 16, 8, 2)

        requiring that the stride equals n_steps_output.
        """
        pred_outs: list[Tensor] = []
        true_outs: list[Tensor] = []
        current_batch = self._clone_batch(batch)

        # If free running only, override teacher_forcing_ratio=0.0
        teacher_forcing_ratio = (
            self.teacher_forcing_ratio if not free_running_only else 0.0
        )

        n_steps_output = self._predict(current_batch).shape[1]
        if n_steps_output != self.stride and not return_windows:
            msg = (
                f"Rollout stride ({self.stride}) must equal "
                f"n_steps_output ({n_steps_output}) for correct concatenation."
            )
            raise ValueError(msg)

        for _ in range(self.max_rollout_steps):
            output = self._predict(current_batch)
            pred_outs.append(output)

            true_slice, should_record = self._true_slice(current_batch, self.stride)
            if should_record:
                true_outs.append(true_slice)

            rand_val = torch.rand(1, device=output.device).item()
            teacher_force = true_slice.numel() > 0 and rand_val < teacher_forcing_ratio
            next_inputs = true_slice if teacher_force else output.detach()

            if next_inputs.shape[1] < self.stride:
                break

            current_batch = self._advance_batch(current_batch, next_inputs, self.stride)

        # Construct rollout outputs
        preds = torch.stack(pred_outs, dim=1)  # (B, R, T, spatial, C)
        if not return_windows:
            # Concatenate rollout windows along time axis if not returning windows
            preds = rearrange(preds, "b r t ... -> b (r t) ...")  # (B, T*R, spatial, C)
        if len(true_outs) == 0:
            return preds, None

        trues = torch.stack(true_outs, dim=1)  # (B, R, T, spatial, C)
        if not return_windows:
            trues = rearrange(trues, "b r t ... -> b (r t) ...")  # (B, T*R, spatial, C)
        return preds, trues

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
