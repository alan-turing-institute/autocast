import pytest
import torch

from auto_cast.metrics import ALL_METRICS
from auto_cast.types import TensorBTSC


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_spatiotemporal_metrics(metric):
    y_pred: TensorBTSC = torch.ones((2, 3, 4, 5))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 5))
    n_spatial_dims = 1

    error = metric()(y_pred, y_true, n_spatial_dims)
    assert torch.allclose(error.nansum(), torch.tensor(0.0))
