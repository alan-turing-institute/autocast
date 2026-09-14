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

    #: Whether this model can be fed its own predictions back in as inputs.
    #: Autoregressive rollout requires output fields to match input fields
    #: (same channels and spatial resolution). Models whose output fields
    #: differ from their input fields (e.g. downscaling, or asymmetric
    #: multi-modal/conditioned inputs) should set this to False and use
    #: one-shot (non-rollout) training/evaluation instead.
    supports_rollout: bool = True

    def rollout(
        self,
        batch: BatchT,
        stride: int,
        max_rollout_steps: int = 10,
        teacher_forcing_ratio: float = 0.0,
        free_running_only: bool = False,
        return_windows: bool = False,
        detach: bool = True,
        n_members: int | None = None,  # noqa: ARG002
    ) -> RolloutOutput:
        """Perform rollout over multiple time steps.

        Args:
            batch: Input batch containing initial data for rollout.
            stride: Number of steps to advance the batch window each iteration.
            max_rollout_steps: Number of rollout windows to generate.
            teacher_forcing_ratio: Probability of using ground-truth inputs each step.
            free_running_only: If True, disables teacher forcing during rollout.
            return_windows: If True, returns the true outputs in windows matching
                the model's output shape.
            detach: If True, detaches the output from the graph before feeding it
                back as input. Set to False for autoregressive loss calculation.
            n_members: Number of ensemble members for ensemble models.

        Note:
            The outputs stack along a new axis after batch representing
            number of rollout windows R. Each window R contains n_steps_output
            time steps T.
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
        if not self.supports_rollout:
            msg = (
                "This model has supports_rollout=False and cannot perform "
                "autoregressive rollout. This is expected for models whose "
                "output fields differ from their input fields (e.g. spatial "
                "or temporal downscaling, or asymmetric multi-modal/"
                "conditioned inputs). Use one-shot (non-rollout) training/"
                "evaluation instead."
            )
            raise NotImplementedError(msg)

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

            self._check_feedback_compatible(current_batch, next_inputs)
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

    def _check_feedback_compatible(self, batch: BatchT, next_inputs: Tensor) -> None:
        """Verify predictions can be fed back in as inputs, before attempting it.

        Compares everything but the time axis: feeding predictions back means
        concatenating them onto the input window along time, so the channel
        count and spatial resolution must already agree.

        Args:
            batch: The current rollout batch.
            next_inputs: The tensor about to be fed back in as inputs.

        Raises:
            ValueError: If `next_inputs` cannot be concatenated onto the
                batch's input fields.
        """
        current_inputs = self._input_fields(batch)
        expected = tuple(current_inputs.shape[2:])
        actual = tuple(next_inputs.shape[2:])
        if expected == actual:
            return
        msg = (
            "Cannot feed this model's predictions back in as inputs: they have "
            f"shape (channels/spatial) {actual}, but its input fields have "
            f"{expected}. Autoregressive rollout requires the two to match. "
            "Models whose outputs differ from their inputs (downscaling, or "
            "predicting different fields than they consume) should set "
            "supports_rollout=False and use one-shot evaluation instead."
        )
        raise ValueError(msg)

    @abstractmethod
    def _input_fields(self, batch: BatchT) -> Tensor:
        """Return the batch's input fields, the tensor predictions feed back into."""

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
