import numpy as np
import torch
import torch.nn as nn


class Metric(nn.Module):
    """
    Decorator for metrics that standardizes the input arguments and checks the dimensions of the input tensors.

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
        assert len(args) >= 2, "At least two arguments required (y_pred, y_true)"
        y_pred, y_true = args[:2]

        # Convert x and y to torch.Tensor if they are np.ndarray
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        assert isinstance(y_pred, torch.Tensor), "y_pred must be a torch.Tensor or np.ndarray"
        assert isinstance(y_true, torch.Tensor), "y_true must be a torch.Tensor or np.ndarray"
    
        # Check dimensions
        """TODO: check TheWell"""

        return self.eval(y_pred, y_true, **kwargs)

    @staticmethod
    def eval(self, y_pred, y_true, **kwargs):
        raise NotImplementedError