import torch
from einops import rearrange
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


def compute_v(
    x, y_pos, y_neg, T, N, N_pos, return_stats: bool = False
) -> tuple[Tensor, dict[str, float] | None]:
    """Compute drift vector field V for drifting processor."""
    # x: [N, D]
    # y_pos: [N_pos, D]
    # y_neg: [N_neg, D]
    # T: temperature
    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N, N_neg]
    # ignore self (if y_neg is x) - Alg. 2
    dist_neg = dist_neg + torch.eye(N, device=x.device, dtype=x.dtype) * 1e6
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
    N_neg = y_neg.shape[0]
    A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=1)
    # compute the weights
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # [N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # [N, N_neg]
    drift_pos = W_pos @ y_pos  # [N_x, D]
    drift_neg = W_neg @ y_neg  # [N_x, D]
    V = drift_pos - drift_neg
    if not return_stats:
        return V, None
    with torch.no_grad():
        stats = {
            "dist_pos_mean": dist_pos.mean().item(),
            "dist_neg_mean": dist_neg.mean().item(),
            "logit_min": logit.min().item(),
            "logit_max": logit.max().item(),
            "a_pos_sum_mean": A_pos.sum(dim=1).mean().item(),
            "a_neg_sum_mean": A_neg.sum(dim=1).mean().item(),
            "w_pos_mean": W_pos.mean().item(),
            "w_neg_mean": W_neg.mean().item(),
            "v_abs_mean": V.abs().mean().item(),
            "v_norm_mean": V.norm(dim=1).mean().item(),
        }
    return V, stats


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
        temperature: float = 50,
        debug_every: int = 50,
    ) -> None:
        # Store core hyperparameters and optional prebuilt backbone.
        super().__init__()
        self.generator = backbone

        self.n_steps_output = n_steps_output
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.n_samples = n_samples
        self.temperature = temperature
        self.debug_every = debug_every
        self._debug_step = 0

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
        n_gen = batch_size * n_samples if n_samples is not None else batch_size
        z_shape = (
            n_gen,
            self.n_steps_output,
            *spatial_shape,
            self.n_channels_in,
        )
        z = torch.randn(z_shape, device=device, dtype=dtype)
        if n_samples is not None:
            x = x.repeat_interleave(n_samples, dim=0)
            if global_cond is not None:
                global_cond = global_cond.repeat_interleave(n_samples, dim=0)
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
        """Compute drifting loss (Alg. 1): L = E[||x - stopgrad(x + V)||^2]."""
        batch_n_samples = batch.repeat(self.n_samples)
        target_states = batch_n_samples.encoded_output_fields  # y_pos from p_data
        cond = batch_n_samples.encoded_inputs
        global_cond = batch_n_samples.global_cond

        N = cond.shape[0]
        spatial_shape = tuple(cond.shape[2:-1])
        eps = torch.randn(
            N,
            self.n_steps_output,
            *spatial_shape,
            self.n_channels_in,
            device=cond.device,
            dtype=cond.dtype,
        )
        # Alg. 1: x = f(eps), use noise as generator input (not conditioning)
        x = self.generator_func(eps, cond, global_cond)

        y_neg = x  # reuse generated as negatives (Alg. 1)
        y_pos = target_states

        x_flat = x.flatten(start_dim=1)
        y_pos_flat = y_pos.flatten(start_dim=1)
        y_neg_flat = y_neg.flatten(start_dim=1)

        v, stats = compute_v(
            x_flat,
            y_pos_flat,
            y_neg_flat,
            T=self.temperature,  # paper τ ∈ {0.02, 0.05, 0.2}, curr 50 for stability
            N=N,
            N_pos=y_pos_flat.shape[0],
            return_stats=self.training and self.debug_every > 0,
        )
        # Alg. 1: loss = MSE(x, stopgrad(x + V))
        x_drifted = (x_flat + v).detach()  # Alg. 1
        loss = (x_flat - x_drifted).pow(2).mean()  # Alg. 1
        if (
            self.training
            and self.debug_every > 0
            and stats is not None
            and (self._debug_step % self.debug_every == 0)
        ):
            print(
                "[drifting debug] "
                f"step={self._debug_step} "
                f"loss={loss.item():.3e} "
                f"|v|_mean={stats['v_abs_mean']:.3e} "
                f"v_norm_mean={stats['v_norm_mean']:.3e} "
                f"dist_pos={stats['dist_pos_mean']:.3e} "
                f"dist_neg={stats['dist_neg_mean']:.3e} "
                "logit[min,max]="
                f"({stats['logit_min']:.3e},{stats['logit_max']:.3e}) "
                "A[pos,neg]="
                f"({stats['a_pos_sum_mean']:.3e},{stats['a_neg_sum_mean']:.3e}) "
                "W[pos,neg]="
                f"({stats['w_pos_mean']:.3e},{stats['w_neg_mean']:.3e})"
            )
        self._debug_step += 1
        return loss
