import torch
from einops import rearrange
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
                bias=bias,
                device=device,
                dtype=dtype,
            )
            self.beta = nn.Linear(
                self.n_noise_channels,
                n_channels,
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

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        # Standard LayerNorm if no noise channels configured
        if self.standard_ln is not None:
            return self.standard_ln(x)

        # Conditional LayerNorm requires conditioning tensor
        if cond is None:
            msg = "Conditioning tensor required for ConditionalLayerNorm"
            raise ValueError(msg)

        # x: (B, ..., C_channels)
        # TODO: check cond: (B, C_noise

        # Normalize x
        # - normalized_shape contains dimension sizes
        # - LayerNorm normalizes over the last N dimensions, where N is
        #   len(normalized_shape).

        # Determine dimensions to reduce
        if isinstance(self.normalized_shape, int):
            dims = (-1,)
        else:
            # Count up from the number of dims in normalized_shape from the end
            dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        # Calculate mean, var, and normalized x
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Generate scale and shift
        assert self.gamma is not None, "init failed to create gamma layer"
        assert self.beta is not None, "init failed to create beta layer"

        # Flatten all non-batch dims into a single conditioning vector per batch item
        cond = cond.flatten(start_dim=1)  # (B, C_noise)
        if cond.shape[-1] != self.n_noise_channels:
            msg = (
                f"Conditioning tensor last dim size {cond.shape[-1]} "
                f"does not match n_noise_channels {self.n_noise_channels}"
            )
            raise ValueError(msg)
        gamma = self.gamma(cond)  # (B, C_channels)
        beta = self.beta(cond)  # (B, C_channels)

        # Insert 1s for all dimensions between B and C_channels
        dims_to_add = x.ndim - gamma.ndim
        shape_str = " ".join(["b"] + ["1"] * dims_to_add + ["c"])
        if dims_to_add > 0:
            gamma = rearrange(
                gamma, f"b c -> {shape_str}"
            )  # (B, 1, ..., 1, C_channels)
            beta = rearrange(beta, f"b c -> {shape_str}")  # (B, 1, ..., 1, C_channels)

        return gamma * x_norm + beta
