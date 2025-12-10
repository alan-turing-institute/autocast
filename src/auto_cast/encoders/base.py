from abc import ABC, abstractmethod
from dataclasses import replace

from torch import nn

from auto_cast.types import Batch, EncodedBatch, TensorBNC


class Encoder(nn.Module, ABC):
    """Base encoder."""

    encoder_model: nn.Module
    latent_dim: int

    def preprocess(self, batch: Batch) -> Batch:
        """Optionally transform a batch before encoding.

        Subclasses can override to implement pre-encoding steps that still
        return a fully-populated `Batch` instance. Default is identity.
        """
        return batch

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
        encoded_inputs = self.encode(batch)

        # Assign output fields to inputs to be encoded identically in this default impl
        # Create a new batch with output fields as input fields to prevent mutation
        output_batch = replace(batch, input_fields=batch.output_fields)

        encoded_outputs = self.encode(output_batch)

        # Return encoded batch
        return EncodedBatch(
            encoded_inputs=encoded_inputs,
            encoded_output_fields=encoded_outputs,
            encoded_info=encoded_info or {},
        )

    def __call__(self, batch: Batch) -> TensorBNC:
        return self.encode(batch)
