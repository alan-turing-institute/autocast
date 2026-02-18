import torch
from einops import rearrange
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


def compute_v(x, y_pos, y_neg, T, N, N_pos):
    """Compute drift vector field V for drifting processor."""
    # x: [N, D]
    # y_pos: [N_pos, D]
    # y_neg: [N_neg, D]
    # T: temperature
    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N, N_neg]
    # ignore self (if y_neg is x)
    dist_neg += torch.eye(N) * 1e6
    # compute logits
    logit_pos = -dist_pos / T
    logit_neg = -dist_neg / T
    # concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)
    # normalize along both dimensions
    A_row = logit.softmax(dim=-1)
    A_col = logit.softmax(dim=-2)
    A = torch.sqrt(A_row * A_col)
    # back to [N, N_pos] and [N, N_neg]
    A_pos, A_neg = torch.split(A, [N_pos], dim=1)
    # compute the weights
    W_pos = A_pos  # [N, N_pos]
    W_neg = A_neg  # [N, N_neg]
    W_pos *= A_neg.sum(dim=1, keepdim=True)
    W_neg *= A_pos.sum(dim=1, keepdim=True)
    drift_pos = W_pos @ y_pos  # [N_x, D]
    drift_neg = W_neg @ y_neg  # [N_x, D]
    V = drift_pos - drift_neg
    return V


class DriftingProcessor(Processor):
    """Processor that wraps a flow-matching generative model."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        n_steps_output: int = 4,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_samples: int = 20,
    ) -> None:
        # Store core hyperparameters and optional prebuilt backbone.
        super().__init__()
        self.generator = backbone

        self.n_steps_output = n_steps_output
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.n_samples = n_samples

    def generator_func(
        self, z: Tensor, x: Tensor, global_cond: Tensor | None = None
    ) -> Tensor:
        """Flow matching vector field.

        The vector field over the tangent space of output states (z).
        conditioned on input states (x) at time (t).

        Args:
            z: Current output states of shape (B, T_out, *spatial, C_out).
            x: Conditioning inputs of shape (B, T_in, *spatial, C_in).
            global_cond: Optional non-spatial conditioning/modulation tensor.

        Returns
        -------
            Time derivative of output states with the same shape as `z`.
        """
        return self.generator(z, t=None, cond=x, global_cond=global_cond)

    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Alias to map for Lightning/PyTorch compatibility."""
        return self.map(x, global_cond)

    def map(
        self, x: Tensor, global_cond: Tensor | None, n_samples: int | None = None
    ) -> Tensor:
        """Map inputs states (x) to output states (z) by passing through the generator.

        Args:
            x: Conditioning inputs of shape (B, T_in, *spatial, C_in).
            global_cond: Optional non-spatial conditioning/modulation tensor.

        Returns
        -------
            Generated outputs of shape (B, T_out, *spatial, C_out).
        """
        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype

        # Initialize noisy sample and scalar time for each batch element.
        spatial_shape = tuple(x.shape[2:-1])
        z_shape = (
            batch_size * n_samples if n_samples is not None else batch_size,
            self.n_steps_output,
            *spatial_shape,
            self.n_channels_in,
        )
        z = torch.randn(z_shape, device=device, dtype=dtype)
        return (
            self.generator_func(z, x, global_cond)
            if n_samples is None
            else rearrange(
                self.generator_func(z, x, global_cond),
                "(b m) ... -> b ... m",
                b=batch_size,
                m=n_samples,
            )
        )

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute flow-matching loss for a batch."""
        batch_n_samples = batch.repeat(self.n_samples)
        target_states = batch.encoded_output_fields
        eps = torch.randn(
            *batch_n_samples.encoded_inputs.shape[:-1], self.n_channels_in
        )
        x = self.map(eps, batch_n_samples.global_cond, self.n_samples)
        y_neg = x
        y_pos = target_states
        v = compute_v(
            x.flatten(start_dim=1),
            y_pos.flatten(start_dim=1),
            y_neg.flatten(start_dim=1),
            T=1.0,
            N=eps.shape[0],
            N_pos=y_pos.shape[0],
        )
        loss = (x - (x + v).detach()).pow(2).mean()
        return loss
