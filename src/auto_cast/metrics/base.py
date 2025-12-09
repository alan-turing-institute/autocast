import numpy as np
import torch
from torch import nn

from auto_cast.types import TensorBCTSPlus


class Metric(nn.Module):
    """
    Base class for metrics.

    This class standardizes the input arguments and
    checks the dimensions of the input tensors.

    Args:
        f: function
            Metric function that takes in the following arguments:
            y_pred: torch.Tensor | np.ndarray
                Predicted values tensor.
            y_true: torch.Tensor | np.ndarray
                Target values tensor.
            **kwargs : dict
                Additional arguments for the metric.
    """

    def forward(self, *args, **kwargs):
        assert len(args) >= 2, (
            "At least two arguments required (y_pred, y_true, n_spatial_dims)"
        )
        y_pred, y_true, n_spatial_dims = args[:3]

        # Convert y_pred and y_true to torch.Tensor if they are np.ndarray
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        assert isinstance(y_pred, torch.Tensor), (
            "y_pred must be a torch.Tensor or np.ndarray"
        )
        assert isinstance(y_true, torch.Tensor), (
            "y_true must be a torch.Tensor or np.ndarray"
        )

        # Check dimensions
        assert y_pred.ndim >= n_spatial_dims + 1, (
            "y_pred must have at least n_spatial_dims + 1 dimensions"
        )
        assert y_true.ndim >= n_spatial_dims + 1, (
            "y_true must have at least n_spatial_dims + 1 dimensions"
        )
        return self.score(y_pred, y_true, n_spatial_dims, **kwargs)

    @staticmethod
    def score(
        y_pred: TensorBCTSPlus, y_true: TensorBCTSPlus, n_spatial_dims: int, **kwargs
    ):
        raise NotImplementedError
