import torch
import pytest

from auto_cast.metrics.spatiotemporal import MSE, MAE, NMAE, NMSE, NRMSE, RMSE, VRMSE, VMSE
from auto_cast.types import TensorBTSPlusC

@pytest.mark.parametrize(
    "metric",
    (MSE(), MAE(), NMAE(), NMSE(), NRMSE(), RMSE(), VRMSE(), VMSE()),
)

def test_spatiotemporal_metrics(metric):
    y_pred: TensorBTSPlusC = torch.ones((2, 3, 4, 5))
    y_true: TensorBTSPlusC = torch.ones((2, 3, 4, 5))
    n_spatial_dims = 1

    error = metric(y_pred, y_true, n_spatial_dims)
    assert torch.allclose(error.nansum(), torch.tensor(0.0))