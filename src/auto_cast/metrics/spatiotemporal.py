import numpy as np
import torch

from auto_cast.metrics.base import Metric
from auto_cast.types import TensorBTSPlusC, TensorBTC

class MSE(Metric):
    @staticmethod
    def score(
        y_pred: TensorBTSPlusC,
        y_true: TensorBTSPlusC,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> TensorBTC:
        """
        Mean Squared Error

        Args:
            y_pred: Predicted values tensor.
            y_true: Target values tensor.
            n_spatial_dims: int
                Number of spatial dimensions.

        Returns:
            Mean squared error between y_pred and y_true.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims)
    
class MAE(Metric):
    @staticmethod
    def score(
        y_pred: TensorBTSPlusC,
        y_true: TensorBTSPlusC,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> TensorBTC:
        """
        Mean Absolute Error

        Args:
            y_pred: Predicted values tensor.
            y_true: Target values tensor.
            n_spatial_dims: int
                Number of spatial dimensions.

        Returns:
            Mean absolute error between y_pred and y_true.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        return torch.mean((y_pred - y_true).abs(), dim=spatial_dims)
    

class NMAE(Metric):
    @staticmethod
    def score(
        y_pred: TensorBTSPlusC,
        y_true: TensorBTSPlusC,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> TensorBTC:
        """
        Normalized Mean Absolute Error

        Args:
            y_pred: Predicted values tensor.
            y_true: Target values tensor.
            n_spatial_dims: int
                Number of spatial dimensions.

        Returns:
            Normalized mean absolute error between y_pred and y_true.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        norm = torch.mean(torch.abs(y_true), dim=spatial_dims)
        return torch.mean((y_pred - y_true).abs(), dim=spatial_dims) / (norm + eps)
    
class NMSE(Metric):
    @staticmethod
    def score(
        y_pred: TensorBTSPlusC,
        y_true: TensorBTSPlusC,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> TensorBTC:
        """
        Normalized Mean Squared Error

        Args:
            y_pred: Predicted values tensor.
            y_true: Target values tensor.
            n_spatial_dims: int
                Number of spatial dimensions.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode:
                Mode for computing the normalization factor. Can be 'norm' or 'std'. Default is 'norm'.

        Returns:
            Normalized mean squared error between y_pred and y_true.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        if norm_mode == "norm":
            norm = torch.mean(y_true**2, dim=spatial_dims)
        elif norm_mode == "std":
            norm = torch.std(y_true, dim=spatial_dims) ** 2
        else:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")
        return MSE.score(y_pred, y_true, n_spatial_dims) / (norm + eps)
    
class RMSE(Metric):
    @staticmethod
    def score(
        y_pred: TensorBTSPlusC,
        y_true: TensorBTSPlusC,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> TensorBTC:
        """
        Root Mean Squared Error

        Args:
            y_pred: Predicted values tensor.
            y_true: Target values tensor.
            n_spatial_dims: int
                Number of spatial dimensions.

        Returns:
            Root mean squared error between y_pred and y_true.
        """
        return torch.sqrt(MSE.score(y_pred, y_true, n_spatial_dims))
    
class NRMSE(Metric):
    @staticmethod
    def score(
        y_pred: TensorBTSPlusC,
        y_true: TensorBTSPlusC,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> TensorBTC:
        """
        Normalized Root Mean Squared Error

        Args:
            y_pred: Predicted values tensor.
            y_true: Target values tensor.
            n_spatial_dims: int
                Number of spatial dimensions.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode : Mode for computing the normalization factor. Can be 'norm' or 'std'. Default is 'norm'.

        Returns:
            Normalized root mean squared error between y_pred and y_true.

        """
        return torch.sqrt(NMSE.score(y_pred, y_true, n_spatial_dims, eps=eps, norm_mode=norm_mode))


class VMSE(Metric):
    @staticmethod
    def score(
        y_pred: TensorBTSPlusC,
        y_true: TensorBTSPlusC,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> TensorBTC:
        """
        Variance Scaled Mean Squared Error

        Args:
            y_pred: Predicted values tensor.
            y_true: Target values tensor.
            n_spatial_dims: int
                Number of spatial dimensions.

        Returns:
            Variance mean squared error between y_pred and y_true.
        """
        return NMSE.score(y_pred, y_true, n_spatial_dims, norm_mode="std")


class VRMSE(Metric):
    @staticmethod
    def score(
        y_pred: TensorBTSPlusC,
        y_true: TensorBTSPlusC,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> TensorBTC:
        """
        Root Variance Scaled Mean Squared Error

        Args:
            y_pred: Predicted values tensor.
            y_true: Target values tensor.
            n_spatial_dims: int
                Number of spatial dimensions.

        Returns:
            Root variance mean squared error between y_pred and y_true.
        """
        return NRMSE.score(y_pred, y_true, n_spatial_dims, norm_mode="std")


class LInfinity(Metric):
    @staticmethod
    def score(
        y_pred: TensorBTSPlusC,
        y_true: TensorBTSPlusC,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> TensorBTC:
        """
        L-Infinity Norm

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.

        Returns:
            L-Infinity norm between y_pred and y_true.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        return torch.max(
            torch.abs(y_pred - y_true).flatten(start_dim=spatial_dims[0], end_dim=-2), dim=-2
        ).values