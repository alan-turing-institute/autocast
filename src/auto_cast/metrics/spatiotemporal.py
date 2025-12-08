import numpy as np
import torch

from auto_cast.metrics.base import Metric
from auto_cast.types import TensorBTSPlusC, TensorBTC

class MSE(Metric):
    @staticmethod
    def eval(
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
    def eval(
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
    def eval(
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