import numpy as np
import torch

from autocast.metrics.base import BaseMetric
from autocast.types import Tensor, TensorBTC, TensorBTSC
from autocast.types.types import ArrayLike, TensorBTSCM


class BTSCMetric(BaseMetric[TensorBTSC | TensorBTSCM, TensorBTSC]):
    """
    Base class for metrics that operate on spatial tensors.

    Checks input types and shapes and converts to Tensor.

    Args:
        reduce_all: If True, return scalar by averaging over all non-batch dims
        dist_sync_on_step: Synchronize metric state across processes at each forward()
    """

    def _check_input(
        self, y_pred: ArrayLike, y_true: ArrayLike
    ) -> tuple[TensorBTSC, TensorBTSC]:
        """
        Check types and shapes and converts inputs to Tensor.

        Args:
            y_pred: Predictions of shape (B, T, S, C)
            y_true: Ground truth of shape (B, T, S, C)

        Returns
        -------
            Tuple of (y_pred, y_true) as Tensors
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)

        if not isinstance(y_pred, Tensor):
            raise TypeError(
                f"y_pred must be a Tensor or np.ndarray, got {type(y_pred)}"
            )
        if not isinstance(y_true, Tensor):
            raise TypeError(
                f"y_true must be a Tensor or np.ndarray, got {type(y_true)}"
            )

        if y_pred.ndim == y_true.ndim and y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred and y_true must have the same shape, "
                f"got {y_pred.shape} and {y_true.shape}"
            )

        if y_pred.ndim == y_true.ndim + 1 and y_pred.shape[:-1] != y_true.shape:
            raise ValueError(
                f"y_pred (ensemble) and y_true must have the same shape along "
                f"non-ensemble dims, got {y_pred.shape} and {y_true.shape}"
            )

        if y_pred.ndim < 4:
            raise ValueError(
                f"y_pred has {y_pred.ndim} dimensions, should be at least 4, "
                f"following the pattern(B, T, S, C)"
            )

        # Handle ensemble dimension if present
        if y_pred.ndim == y_true.ndim + 1:
            y_pred = self._ensemble_aggregation(y_pred)

        return y_pred, y_true

    def _ensemble_aggregation(self, y_pred: TensorBTSCM) -> TensorBTSC:
        """Aggregate ensemble dimension.

        Parameters
        ----------
        y_pred
            Predictions with ensemble dim, shape (B, T, C, M)

        Returns
        -------
            Aggregated predictions, shape (B, T, S, C)
        """
        return y_pred.mean(dim=-1)

    def _score(
        self, y_pred: TensorBTSC | TensorBTSCM, y_true: TensorBTSC | TensorBTSCM
    ) -> TensorBTC:
        """
        Compute metric reduced over spatial dims only.

        Expected input shape: (B, T, S, C)
        Expected output shape: (B, T, C)

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class MSE(BTSCMetric):
    """Mean Squared Error over spatial dims."""

    name: str = "mse"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims)


class MAE(BTSCMetric):
    """Mean Absolute Error over spatial dims."""

    name: str = "mae"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.mean((y_pred - y_true).abs(), dim=spatial_dims)


class NMAE(BTSCMetric):
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

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm = torch.mean(torch.abs(y_true), dim=spatial_dims)
        return torch.mean((y_pred - y_true).abs(), dim=spatial_dims) / (norm + self.eps)


class NMSE(BTSCMetric):
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

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm = torch.mean(y_true**2, dim=spatial_dims)
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims) / (norm + self.eps)


class RMSE(BTSCMetric):
    """Root Mean Squared Error over spatial dims."""

    name: str = "rmse"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=spatial_dims))


class NRMSE(BTSCMetric):
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

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm = torch.mean(y_true**2, dim=spatial_dims)
        return torch.sqrt(
            torch.mean((y_pred - y_true) ** 2, dim=spatial_dims) / (norm + self.eps)
        )


class VMSE(BTSCMetric):
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

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm_var = torch.std(y_true, dim=spatial_dims) ** 2
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims) / (
            norm_var + self.eps
        )


class VRMSE(BTSCMetric):
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

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))

        norm_std = torch.std(y_true, dim=spatial_dims)

        return torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=spatial_dims)) / (
            norm_std + self.eps
        )


class LInfinity(BTSCMetric):
    """L-Infinity Norm over spatial dims."""

    name: str = "l_infinity"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.max(
            torch.abs(y_pred - y_true).flatten(start_dim=spatial_dims[0], end_dim=-2),
            dim=-2,
        ).values
