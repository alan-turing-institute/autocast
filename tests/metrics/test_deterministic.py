import pytest
import torch

from autocast.metrics import ALL_DETERMINISTIC_METRICS
from autocast.metrics.deterministic import (
    PowerSpectrumRMSE,
    PowerSpectrumRMSEHigh,
    PowerSpectrumRMSELow,
    PowerSpectrumRMSEMid,
    PowerSpectrumRMSETail,
    _isotropic_binning,
)
from autocast.types import TensorBTSC


@pytest.mark.parametrize("MetricCls", ALL_DETERMINISTIC_METRICS)
def test_spatiotemporal_metrics(MetricCls):
    # shape. (B, T, S1, S2, C) with n_spatial_dims = 2
    y_pred: TensorBTSC = torch.ones((2, 3, 4, 4, 5))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 4, 5))

    # instantiate the metric with n_spatial_dims
    metric = MetricCls()

    # score computes the metric over the spatial dims, returning (B, T, C)
    error = metric.score(y_pred, y_true)

    # for identical tensors, all errors must be zero
    assert torch.allclose(error.nansum(), torch.tensor(0.0))


@pytest.mark.parametrize("MetricCls", ALL_DETERMINISTIC_METRICS)
def test_spatiotemporal_metrics_stateful(MetricCls):
    y_pred = torch.ones((2, 3, 4, 4, 5))
    y_true = torch.ones((2, 3, 4, 4, 5))

    metric = MetricCls()
    metric.update(y_pred, y_true)
    value = metric.compute()

    assert torch.allclose(value, torch.tensor(0.0))


@pytest.mark.parametrize(
    "MetricCls",
    [
        PowerSpectrumRMSE,
        PowerSpectrumRMSELow,
        PowerSpectrumRMSEMid,
        PowerSpectrumRMSEHigh,
        PowerSpectrumRMSETail,
    ],
)
def test_power_spectrum_rmse_increases_with_spectral_scale(MetricCls):
    torch.manual_seed(0)
    y_true: TensorBTSC = torch.randn((1, 1, 8, 8, 1))
    y_pred: TensorBTSC = 2.0 * y_true

    value = MetricCls()(y_pred, y_true)
    assert torch.all(value > 0)


def test_isotropic_binning_respects_requested_device():
    shape = (8, 8)
    edges_cpu, counts_cpu, indices_cpu = _isotropic_binning(
        shape, device=torch.device("cpu")
    )
    assert edges_cpu.device.type == "cpu"
    assert counts_cpu.device.type == "cpu"
    assert indices_cpu.device.type == "cpu"

    target_device: torch.device | None = None
    if torch.cuda.is_available():
        target_device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        target_device = torch.device("mps")

    if target_device is None:
        return

    # Cross-device call after CPU call should still honor requested device.
    edges_dev, counts_dev, indices_dev = _isotropic_binning(shape, device=target_device)
    assert edges_dev.device.type == target_device.type
    assert counts_dev.device.type == target_device.type
    assert indices_dev.device.type == target_device.type

    # CUDA indices are stable/meaningful; MPS may report mps:0 while target is mps.
    if target_device.type == "cuda":
        assert edges_dev.device.index == target_device.index
        assert counts_dev.device.index == target_device.index
        assert indices_dev.device.index == target_device.index
