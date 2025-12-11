import pytest
import torch

from auto_cast.metrics import ALL_METRICS
from auto_cast.types import TensorBTSC


@pytest.mark.parametrize("MetricCls", ALL_METRICS)
def test_spatiotemporal_metrics(MetricCls):
    # shape. (B, T, S, C) with n_spatial_dims = 1
    y_pred: TensorBTSC = torch.ones((2, 3, 4, 5))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 5))
    n_spatial_dims = 1

    # instantiate the metric with n_spatial_dims
    metric = MetricCls(n_spatial_dims=n_spatial_dims)

    # score computes the metric over the spatial dims, returning (B, T, C)
    error = metric.score(y_pred, y_true)

    # for identical tensors, all errors must be zero
    assert torch.allclose(error.nansum(), torch.tensor(0.0))


@pytest.mark.parametrize("MetricCls", ALL_METRICS)
def test_spatiotemporal_metrics_stateful(MetricCls):
    y_pred = torch.ones((2, 3, 4, 5))
    y_true = torch.ones((2, 3, 4, 5))

    metric = MetricCls(n_spatial_dims=1, reduce_all=True)
    metric.update(y_pred, y_true)
    value = metric.compute()

    assert torch.allclose(value, torch.tensor(0.0))
