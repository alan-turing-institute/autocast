from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Generic, TypeVar

import torch
from torch import nn

from autocast.types import Batch, EncodedBatch, TensorBNC
from autocast.types.types import Tensor

# Generic batch type variables
BatchT = TypeVar("BatchT")
BatchTEncoded = TypeVar("BatchTEncoded")


class GenericEncoder(nn.Module, ABC, Generic[BatchT, BatchTEncoded]):
    """Base encoder interface."""

    def preprocess(self, batch: BatchT) -> BatchT:
        """Optionally transform a batch before encoding.

        Subclasses can override to implement pre-encoding steps that still
        return a fully-populated `Batch` instance. Default is identity.
        """
        return batch

    @abstractmethod
    def encode(self, batch: BatchT) -> TensorBNC | tuple[TensorBNC, Tensor | None]:
        """Encode the input tensor into the latent space.

        Parameters
        ----------
        batch: BatchT
            Input batch to be encoded.

        Returns
        -------
        TensorBNC | tuple[TensorBNC, Tensor | None]
            Encoded tensor in the latent space with shape (B, *, C_latent) or a tuple of
            (encoded tensor, optional conditioning tensor of shape (B, D)).
        """

    @abstractmethod
    def encode_batch(
        self, batch: BatchT, encoded_info: dict | None = None
    ) -> BatchTEncoded:
        """Encode a full BatchT into a BatchTEncoded.

        Parameters
        ----------
        batch: BatchT
            Input batch to be encoded.
        encoded_info: dict | None
            Optional dictionary of additional encoded information to include.

        Returns
        -------
        BatchTEncoded
            Encoded batch containing encoded inputs and original output fields.
        """

    def forward(self, batch: BatchT) -> TensorBNC | tuple[TensorBNC, Tensor | None]:
        return self.encode(batch)


class _Encoder(GenericEncoder[Batch, EncodedBatch]):
    def encode_batch(
        self, batch: Batch, encoded_info: dict | None = None
    ) -> EncodedBatch:
        """Encode a full Batch into an EncodedBatch.

        By default, encodes both input_fields and output_fields identically.
        Subclasses can override to implement different encoding strategies.

        Parameters
        ----------
        batch: Batch
            Input batch to be encoded.

        Returns
        -------
        EncodedBatch
            Encoded batch containing encoded inputs and original output fields.
        """
        encoded = self.encode(batch)

        def _process_encoded(
            encoded: TensorBNC | tuple[TensorBNC, Tensor | None],
        ) -> tuple[TensorBNC, Tensor | None]:
            return (
                (encoded[0], encoded[1])
                if isinstance(encoded, tuple)
                else (encoded, None)
            )

        encoded_inputs, global_cond = _process_encoded(encoded)

        # Assign output fields to inputs to be encoded identically in this default impl
        # Create a new batch with output fields as input fields to prevent mutation
        output_batch = replace(batch, input_fields=batch.output_fields.clone())

        encoded_outputs, _ = _process_encoded(self.encode(output_batch))

        # Return encoded batch
        return EncodedBatch(
            encoded_inputs=encoded_inputs,
            encoded_output_fields=encoded_outputs,
            global_cond=global_cond,
            encoded_info=encoded_info or {},
        )


class Encoder(_Encoder):
    """Base encoder."""

    encoder_model: nn.Module
    channel_dim: int
    latent_dim: int
    outputs_time_channel_concat: bool = False

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


class EncoderWithCond(Encoder):
    """Encoder that returns encoded tensor and optional conditioning."""

    def encode_cond(self, batch: Batch) -> Tensor | None:
        """Encode global conditioning tensor from the batch.

        Default implementation flattens constant scalars and boundary conditions.
        """
        global_cond = None
        if batch.constant_scalars is not None:
            global_cond = batch.constant_scalars
        if batch.boundary_conditions is not None:
            bc = batch.boundary_conditions
            bc = bc.flatten(start_dim=1)
            if global_cond is None:
                global_cond = bc
            else:
                global_cond = torch.cat([global_cond, bc], dim=1)

        return global_cond

    def encode_with_cond(self, batch: Batch) -> tuple[TensorBNC, Tensor | None]:
        """Encode the input tensor into the latent space.

        Parameters
        ----------
        x: Batch
            Input batch to be encoded.

        Returns
        -------
        tuple[TensorBNC, Tensor | None]
            Encoded tensor in the latent space with shape (B, *, C_latent) with optional
            conditioning tensor of shape (B, D).
        """
        return (self.encode(batch), self.encode_cond(batch))

    def encode_batch(
        self, batch: Batch, encoded_info: dict | None = None
    ) -> EncodedBatch:
        """Encode a full Batch into an EncodedBatch with conditioning.

        Overrides base implementation to ensure encode_with_cond is used to capture
        any provided global conditioning variables.
        """
        encoded_inputs, global_cond = self.encode_with_cond(batch)

        # Create a new batch with output fields as input fields to prevent mutation
        output_batch = replace(batch, input_fields=batch.output_fields.clone())

        encoded_outputs = self.encode(output_batch)
        if isinstance(encoded_outputs, tuple):
            encoded_outputs = encoded_outputs[0]

        return EncodedBatch(
            encoded_inputs=encoded_inputs,
            encoded_output_fields=encoded_outputs,
            global_cond=global_cond,
            encoded_info=encoded_info or {},
        )

    def forward(self, batch: Batch) -> TensorBNC | tuple[TensorBNC, Tensor | None]:
        return self.encode_with_cond(batch)
