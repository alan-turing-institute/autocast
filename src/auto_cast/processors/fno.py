from typing import Any

import torch
from neuralop.models import FNO
from torch import nn

from auto_cast.preprocessor import FNOInputPreprocessor
from auto_cast.processors.base import DiscreteProcessor
from auto_cast.types import Batch, RolloutOutput, Tensor


class FNOProcessor(DiscreteProcessor):
    """Fourier Neural Operator Processor.

    A discrete processor that uses a Fourier Neural Operator (FNO) to learn
    mappings between function spaces for spatiotemporal prediction.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    n_modes: tuple[int, ...]
        Number of Fourier modes to keep in each spatial dimension.
    hidden_channels: int, optional
        Width of the FNO (number of channels in hidden layers). Default is 64.
    n_layers: int, optional
        Number of FNO layers. Default is 4.
    channels: tuple[int, ...], optional
        Which channels from input_fields to use. Default is (0,).
    with_constants: bool, optional
        Whether to include constant fields in input. Default is False.
    with_time: bool, optional
        Whether to include time information. Default is False.
    n_steps_output: int, optional
        Number of output time steps. Default is 1.
    loss_fn: nn.Module, optional
        Loss function. Defaults to MSELoss.
    learning_rate: float, optional
        Learning rate for optimizer. Default is 1e-3.
    **fno_kwargs
        Additional keyword arguments passed to the FNO model.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        hidden_channels: int = 64,
        n_layers: int = 4,
        channels: tuple[int, ...] = (0,),
        with_constants: bool = False,
        with_time: bool = False,
        n_steps_output: int = 1,
        loss_fn: nn.Module | None = None,
        learning_rate: float = 1e-3,
        **fno_kwargs: Any,
    ) -> None:
        super().__init__()

        self.model = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **fno_kwargs,
        )
        self.channels = channels
        self.with_constants = with_constants
        self.with_time = with_time
        self.n_steps_output = n_steps_output
        self.loss_func = loss_fn or nn.MSELoss()  # TODO: update with loss
        self.learning_rate = learning_rate
        self.input_preprocessor = FNOInputPreprocessor(
            channels=channels,
            with_constants=with_constants,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the FNO model.

        Parameters
        ----------
        x: Tensor
            Input tensor of shape (B, C, *spatial_dims) or (B, C, T, *spatial_dims).

        Returns
        -------
        Tensor
            Output tensor from the FNO model.

        """
        return self.model(x)

    def _prepare_input(self, batch: Batch) -> Tensor:
        """Prepare input tensor from batch.

        Parameters
        ----------
        batch: Batch
            Input batch containing 'input_fields' and optionally 'constant_fields'.

        Returns
        -------
        Tensor
            Prepared input tensor for the FNO model.

        """
        return self.input_preprocessor(batch)

    def map(self, x: Batch) -> Tensor:
        """Map input batch to output prediction using FNO.

        Parameters
        ----------
        x: Batch
            Input batch dictionary.

        Returns
        -------
        Tensor
            Predicted output fields.

        """
        x_input = self._prepare_input(x)
        y_pred = self.forward(x_input)

        # Handle temporal output if with_time is enabled
        if self.with_time and y_pred.dim() > 3:
            # Assume shape is (B, C, T, *spatial_dims)
            y_pred = y_pred[:, :, : self.n_steps_output, ...]

        return y_pred

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Perform a single training step.

        Parameters
        ----------
        batch: Batch
            Input batch.
        batch_idx: int
            Index of the batch.

        Returns
        -------
        Tensor
            Computed loss value.

        """
        y_pred = self.map(batch)
        y_true = batch["output_fields"][:, list(self.channels), ...]
        loss = self.loss_func(y_pred, y_true)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Perform a single validation step.

        Parameters
        ----------
        batch: Batch
            Input batch.
        batch_idx: int
            Index of the batch.

        Returns
        -------
        Tensor
            Computed loss value.

        """
        y_pred = self.map(batch)
        y_true = batch["output_fields"][:, list(self.channels), ...]
        loss = self.loss_func(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer with configured learning rate.

        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def rollout(self, batch: Batch) -> RolloutOutput:
        """Rollout predictions over multiple time steps.

        Parameters
        ----------
        batch: Batch
            Input batch for rollout.

        Returns
        -------
        RolloutOutput
            Tuple of (predictions, ground_truth) stacked over time steps.

        """
        pred_outs = []
        gt_outs = []

        # Create a working copy of the batch
        current_batch = {k: v.clone() for k, v in batch.items()}

        for _ in range(0, self.max_rollout_steps, self.stride):
            pred = self.map(current_batch)
            pred_outs.append(pred)

            if "output_fields" in current_batch:
                gt_outs.append(
                    current_batch["output_fields"][:, list(self.channels), ...]
                )

            # Update input for next step (autoregressive rollout)
            current_batch["input_fields"] = pred

        predictions = torch.stack(pred_outs)
        if gt_outs:
            return (predictions, torch.stack(gt_outs))
        return (predictions, None)
