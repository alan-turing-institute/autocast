from abc import ABC, abstractmethod
from dataclasses import replace

import torch
from torch import nn

from autocast.types import Batch, EncodedBatch, Tensor, TensorBNC


class Encoder(nn.Module, ABC):
    """Base encoder."""

    encoder_model: nn.Module
    latent_dim: int

    # Constants concatenation options
    with_constant_fields: bool = False
    with_constant_scalars: bool = False

    def preprocess(self, batch: Batch) -> Batch:
        """Optionally transform a batch before encoding.

        Subclasses can override to implement pre-encoding steps that still
        return a fully-populated `Batch` instance. Default is identity.
        """
        return batch

    def _concat_constants(self, x: Tensor, batch: Batch) -> Tensor:
        """Concatenate constant fields and scalars to encoded tensor.

        Parameters
        ----------
        x: Tensor
            Encoded tensor with shape (B, T, *spatial, C).
        batch: Batch
            Original batch containing constant_fields and constant_scalars.

        Returns
        -------
        Tensor
            Tensor with constants concatenated along channel dimension.
        """
        # Get dimensions from encoded tensor
        b, t = x.shape[0], x.shape[1]
        spatial_dims = x.shape[2:-1]
        n_spatial = len(spatial_dims)

        if self.with_constant_fields and batch.constant_fields is not None:
            # constant_fields shape: (B, *spatial, C_fields)
            # Expand to match time dimension: (B, T, *spatial, C_fields)
            constants = batch.constant_fields.unsqueeze(1)
            constants = constants.expand(b, t, *spatial_dims, -1)
            x = torch.cat([x, constants], dim=-1)

        if self.with_constant_scalars and batch.constant_scalars is not None:
            # constant_scalars shape: (B, C_scalars)
            # Expand to match all dimensions: (B, T, *spatial, C_scalars)
            scalars = batch.constant_scalars
            # Add time and spatial dimensions
            for _ in range(1 + n_spatial):  # time + spatial dims
                scalars = scalars.unsqueeze(1)
            scalars = scalars.expand(b, t, *spatial_dims, -1)
            x = torch.cat([x, scalars], dim=-1)

        return x

    @abstractmethod
    def encode(self, batch: Batch) -> TensorBNC:
        """Encode the input tensor into the latent space.

        Parameters
        ----------
        x: Batch
            Input batch to be encoded.

        Returns
        -------
        TensorBNC
            Encoded tensor in the latent space with shape (B, *, C_latent).
        """

    def encode_batch(
        self,
        batch: Batch,
        encoded_info: dict | None = None,
    ) -> EncodedBatch:
        """Encode a full Batch into an EncodedBatch.

        By default, encodes both input_fields and output_fields identically.
        If `with_constant_fields` or `with_constant_scalars` is True, the
        constants are concatenated to the encoded inputs (not outputs) as
        additional channels.

        Subclasses can override to implement different encoding strategies.

        Parameters
        ----------
        batch: Batch
            Input batch to be encoded.
        encoded_info: dict | None
            Optional dictionary of additional encoding information.

        Returns
        -------
        EncodedBatch
            Encoded batch containing encoded inputs and original output fields.
        """
        encoded_inputs = self.encode(batch)

        # Optionally concatenate constants to encoded inputs
        if self.with_constant_fields or self.with_constant_scalars:
            encoded_inputs = self._concat_constants(encoded_inputs, batch)

        # Assign output fields to inputs to be encoded identically in this default impl
        # Create a new batch with output fields as input fields to prevent mutation
        output_batch = replace(batch, input_fields=batch.output_fields.clone())

        encoded_outputs = self.encode(output_batch)

        # Return encoded batch
        return EncodedBatch(
            encoded_inputs=encoded_inputs,
            encoded_output_fields=encoded_outputs,
            label=None,  # TODO: revisit handling of labels if part of API
            encoded_info=encoded_info or {},
        )

    def __call__(self, batch: Batch) -> TensorBNC:
        return self.encode(batch)
