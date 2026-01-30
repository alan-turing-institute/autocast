from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import torch
from einops import rearrange
from tqdm import tqdm

from autocast.types import RolloutOutput, Tensor
from autocast.types.batch import BatchT


class RolloutMixin(ABC, Generic[BatchT]):
    """Rollout logic for generic batches."""

    def rollout(
        self,
        batch: BatchT,
        stride: int,
        max_rollout_steps: int = 10,
        teacher_forcing_ratio: float = 0.0,
        free_running_only: bool = False,
        return_windows: bool = False,
        detach: bool = True,
        n_members: (
            int | None
        ) = None,  # noqa: ARG002 not used in default implementation but potentially in overrides
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
        detach: bool, optional
            If True, detaches the output from the graph before feeding it back in
            as input. Set to False for autoregressive loss calculation. By default True.
        n_members: int | None
            Number of ensemble members for ensemble models. By default None.


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
        teacher_forcing_ratio = teacher_forcing_ratio if not free_running_only else 0.0

        n_steps_output = self._predict(current_batch).shape[1]
        if n_steps_output != stride and not return_windows:
            msg = (
                f"Rollout stride ({stride}) must equal "
                f"n_steps_output ({n_steps_output}) for correct concatenation."
            )
            raise ValueError(msg)

        for _ in tqdm(range(max_rollout_steps), desc="Rollout"):
            output = self._predict(current_batch)
            pred_outs.append(self.denormalize_tensor(output))

            true_slice, should_record = self._true_slice(current_batch, stride)
            if should_record:
                true_outs.append(self.denormalize_tensor(true_slice))

            rand_val = torch.rand(1, device=output.device).item()
            teacher_force = true_slice.numel() > 0 and rand_val < teacher_forcing_ratio

            if teacher_force:
                next_inputs = true_slice
            else:
                next_inputs = output.detach() if detach else output

            if next_inputs.shape[1] < stride:
                break

            current_batch = self._advance_batch(current_batch, next_inputs, stride)

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

    @abstractmethod
    def denormalize_tensor(self, tensor: Tensor, delta=False) -> Tensor: ...
