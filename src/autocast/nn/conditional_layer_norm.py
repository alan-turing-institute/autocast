import torch
from einops import repeat
from torch import Tensor, nn


class ConditionalLayerNorm(nn.Module):
    """Conditional Layer Normalization."""

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        n_noise_channels: int,
        n_channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,  # TODO: check if need non-learnable option  # noqa: ARG002, E501
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape

        self.gamma = nn.Linear(
            n_noise_channels, n_channels, bias=bias, device=device, dtype=dtype
        )
        self.beta = nn.Linear(
            n_noise_channels, n_channels, bias=bias, device=device, dtype=dtype
        )
        self.eps = eps

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # Mean/variance over last dimension
        # TODO: check if we should add ensemble dim here (e.g. B, N, C, E)
        # x: B, N, C
        # mean: B, *[1]*N, C (if reducing over all other dims but keepdim=True)
        # var: B, *[1]*N, C (if reducing over all other dims but keepdim=True)
        _b, *n, _c = x.shape  # TODO: consider where ensemble dim fits in
        n_dim = len(n)
        mean = x.mean(dim=self.normalized_shape, keepdim=True)
        var = x.var(dim=self.normalized_shape, unbiased=False, keepdim=True)
        print("mean / var", mean.shape, var.shape)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        print("x_norm / cond", x_norm.shape, cond.shape)

        # Generate scale and shift
        gamma = self.gamma(cond)  # B, C
        beta = self.beta(cond)  # B, C

        # Expand gamma and beta to match x_norm dimensions
        other_dim_str = " ".join(f"s{i}" for i in range(n_dim))
        other_dim_sizes = {f"s{i}": n[i] for i in range(n_dim)}
        gamma = repeat(gamma, f"b c -> b {other_dim_str} c", **other_dim_sizes)
        beta = repeat(beta, f"b c -> b {other_dim_str} c", **other_dim_sizes)
        print(
            f"Final shapes(gamma, x_norm, beta):{gamma.shape, x_norm.shape, beta.shape}"
        )

        return gamma * x_norm + beta
