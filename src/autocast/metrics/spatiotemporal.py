import numpy as np
import torch
from torchmetrics import Metric

from autocast.types import TensorBTC, TensorBTSC


class BaseMetric(Metric):
    """
    Base class for metrics that operate on spatial tensors.

    Checks input types and shapes and converts to torch.Tensor.

    Args:
        reduce_all: If True, return scalar by averaging over all non-batch dims
        dist_sync_on_step: Synchronize metric state across processes at each forward()
    """

    name: str = "base_metric"

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.reduce_all = reduce_all

        # States shared by all derived metrics
        self.add_state("sum_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        # Internal flag to set shape of sum_score
        self._initialized = False

    def _check_input(
        self,
        y_pred: torch.Tensor | np.ndarray,
        y_true: torch.Tensor | np.ndarray,
    ) -> tuple[TensorBTSC, TensorBTSC]:
        """
        Check types and shapes and converts inputs to torch.Tensor.

        Args:
            y_pred: Predictions of shape (B, T, *S, C)
            y_true: Ground truth of shape (B, T, *S, C)

        Returns
        -------
            Tuple of (y_pred, y_true) as torch.Tensors
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)

        assert isinstance(y_pred, torch.Tensor), (
            f"y_pred must be a torch.Tensor or np.ndarray, got {type(y_pred)}"
        )
        assert isinstance(y_true, torch.Tensor), (
            f"y_true must be a torch.Tensor or np.ndarray, got {type(y_true)}"
        )

        assert y_pred.shape == y_true.shape, (
            f"y_pred and y_true must have the same shape, "
            f"got {y_pred.shape} and {y_true.shape}"
        )

        assert y_pred.ndim >= 4, (
            f"y_pred has {y_pred.ndim} dimensions, should be at least 4, "
            f"following the pattern(B, T, *S, C)"
        )

        return y_pred, y_true

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        """
        Compute metric reduced over spatial dims only.

        Expected input shape: (B, T, *S, C)
        Expected output shape: (B, T, C)

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def update(
        self,
        y_pred: TensorBTSC | np.ndarray,
        y_true: TensorBTSC | np.ndarray,
    ) -> None:
        """
        Update metric state with a batch of predictions and targets.

        Args:
            y_pred: Predictions of shape (B, T, *S, C)
            y_true: Ground truth of shape (B, T, *S, C)
        """
        y_pred, y_true = self._check_input(y_pred, y_true)

        # (B, T, *S, C) -> (B, T, C)
        score_spatial = self.score(y_pred, y_true)

        if score_spatial.ndim != 3:
            raise ValueError(
                f"score must return shape (B, T, C), got {score_spatial.shape}"
            )

        batch_size = score_spatial.shape[0]

        # Sum over batch dimension: (B, T, C) -> (T, C)
        score_summed = torch.sum(score_spatial, dim=0)

        # Lazily set correct shape for sum_score on first batch
        if not self._initialized:
            self.sum_score = torch.zeros_like(score_summed)
            self._initialized = True

        self.sum_score += score_summed
        self.total_samples += batch_size

    def compute(self) -> torch.Tensor:
        """
        Compute final metric value.

        Returns
        -------
            Tensor of shape (T, C) or scalar if reduce_all=True
        """
        if self.total_samples == 0:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)

        score = self.sum_score / self.total_samples

        if self.reduce_all:
            # Average over time and channels
            return score.mean()

        return score

    def reset(self) -> None:
        """Reset metric state and initialization flag."""
        super().reset()
        self._initialized = False

    def _infer_n_spatial_dims(self, tensor: TensorBTSC) -> int:
        """Infer number of spatial dimensions from tensor shape."""
        return tensor.ndim - 3  # Subtract B, T, C


class MSE(BaseMetric):
    """Mean Squared Error over spatial dims."""

    name: str = "mse"

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims)


class MAE(BaseMetric):
    """Mean Absolute Error over spatial dims."""

    name: str = "mae"

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.mean((y_pred - y_true).abs(), dim=spatial_dims)


class NMAE(BaseMetric):
    """Normalized Mean Absolute Error over spatial dims."""

    name: str = "nmae"

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm = torch.mean(torch.abs(y_true), dim=spatial_dims)
        return torch.mean((y_pred - y_true).abs(), dim=spatial_dims) / (norm + self.eps)


class NMSE(BaseMetric):
    """Normalized Mean Squared Error over spatial dims."""

    name: str = "nmse"

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm = torch.mean(y_true**2, dim=spatial_dims)
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims) / (norm + self.eps)


class RMSE(BaseMetric):
    """Root Mean Squared Error over spatial dims."""

    name: str = "rmse"

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=spatial_dims))


class NRMSE(BaseMetric):
    """Normalized Root Mean Squared Error over spatial dims."""

    name: str = "nrmse"

    def __init__(
        self,
        eps: float = 1e-7,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm = torch.mean(y_true**2, dim=spatial_dims)
        return torch.sqrt(
            torch.mean((y_pred - y_true) ** 2, dim=spatial_dims) / (norm + self.eps)
        )


class VMSE(BaseMetric):
    """Variance Scaled Mean Squared Error over spatial dims."""

    name: str = "vmse"

    def __init__(
        self,
        eps: float = 1e-7,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm_var = torch.std(y_true, dim=spatial_dims) ** 2
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims) / (
            norm_var + self.eps
        )


class VRMSE(BaseMetric):
    """Variance-Scaled Root Mean Squared Error over spatial dims.

    Computes VRMSE = RMSE / std(y_true), where std is computed over spatial dims.
    """

    name: str = "vrmse"

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))

        norm_std = torch.std(y_true, dim=spatial_dims)

        return torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=spatial_dims)) / (
            norm_std + self.eps
        )


class LInfinity(BaseMetric):
    """L-Infinity Norm over spatial dims."""

    name: str = "l_infinity"

    def score(
        self,
        y_pred: TensorBTSC,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.max(
            torch.abs(y_pred - y_true).flatten(start_dim=spatial_dims[0], end_dim=-2),
            dim=-2,
        ).values
