import torch
from torch import Tensor, nn


class ConditionalLayerNorm(nn.Module):
    """Conditional Layer Normalization.

    Can behave as standard LayerNorm if n_noise_channels is 0 or None.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        n_noise_channels: int | None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.n_noise_channels = n_noise_channels or 0

        # Get channels for linear layer output from the normalized shape
        n_channels = (
            normalized_shape
            if isinstance(normalized_shape, int)
            else normalized_shape[-1]
        )

        if self.n_noise_channels > 0:
            # Conditional Mode
            self.gamma = nn.Linear(
                self.n_noise_channels,
                n_channels,
                bias=True,  # Include bias to allow scaling around non-zero constant
                device=device,
                dtype=dtype,
            )
            self.beta = nn.Linear(
                self.n_noise_channels,
                n_channels,
                bias=False,  # Since this is additive bias already
                device=device,
                dtype=dtype,
            )
            self.normalizer = nn.LayerNorm(
                normalized_shape,
                eps=eps,
                elementwise_affine=False,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            self.standard_ln = None
        else:
            # Standard LayerNorm fallback
            self.gamma = None
            self.beta = None
            self.standard_ln = nn.LayerNorm(
                normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
                bias=bias,
                device=device,
                dtype=dtype,
            )

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        # Standard LayerNorm if no noise channels configured
        if self.standard_ln is not None:
            return self.standard_ln(x)

        # Conditional LayerNorm requires conditioning tensor
        if x_noise is None:
            msg = "Conditioning tensor required for ConditionalLayerNorm"
            raise ValueError(msg)

        # x: (B, ..., C_channels)
        # x_noise: (B, C_noise)

        # Normalize x
        # - normalized_shape contains dimension sizes
        # - LayerNorm normalizes over the last N dimensions, where N is
        #   len(normalized_shape).

        # Normalize x using ln without affine params (mean/std across normalized_shape)
        x_norm = self.normalizer(x)

        # Generate scale and shift
        assert self.gamma is not None, "init failed to create gamma layer"
        assert self.beta is not None, "init failed to create beta layer"

        # Flatten all non-batch dims into a single conditioning vector per batch item
        x_noise = x_noise.flatten(start_dim=1)  # (B, C_noise)
        if x_noise.shape[-1] != self.n_noise_channels:
            msg = (
                f"Conditioning tensor last dim size {x_noise.shape[-1]} "
                f"does not match n_noise_channels {self.n_noise_channels}"
            )
            raise ValueError(msg)

        gamma = self.gamma(x_noise)  # (B, C_channels)
        beta = self.beta(x_noise)  # (B, C_channels)

        # Apply conditional scale and shift to the normalized input
        ones = torch.ones_like(x_norm)
        scaled = torch.einsum("b c, b ... c-> b ... c", gamma, x_norm)
        shifted = torch.einsum("b c, b ... c-> b ... c", beta, ones)
        return scaled + shifted
