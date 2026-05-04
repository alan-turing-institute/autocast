import math

import pytest
import torch

from autocast.metrics import ALL_DETERMINISTIC_METRICS
from autocast.metrics.deterministic import (
    PowerSpectrumCCRMSE,
    PowerSpectrumCCRMSEHigh,
    PowerSpectrumCCRMSELow,
    PowerSpectrumCCRMSEMid,
    PowerSpectrumCCRMSETail,
    PowerSpectrumRMSE,
    PowerSpectrumRMSEHigh,
    PowerSpectrumRMSELow,
    PowerSpectrumRMSEMid,
    PowerSpectrumRMSETail,
    _isotropic_binning,
)
from autocast.types import TensorBTSC


def _lola_isotropic_binning(
    shape: tuple[int, ...],
    bins: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = [torch.fft.fftfreq(s, device=device) for s in shape]
    k2_grid = torch.meshgrid(*(torch.square(k_i) for k_i in k), indexing="ij")

    k2_iso = torch.zeros_like(k2_grid[0])
    for component in k2_grid:
        k2_iso = k2_iso + component
    k_iso = torch.sqrt(k2_iso)

    if bins is None:
        bins = math.floor(math.sqrt(k_iso.ndim) * min(k_iso.shape) / 2)

    edges = torch.linspace(0, k_iso.max(), bins + 1, device=k_iso.device)
    indices = torch.bucketize(k_iso.flatten(), edges)
    counts = torch.bincount(indices, minlength=bins + 1)
    return edges, counts, indices


def _lola_isotropic_power_spectrum(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edges, counts, indices = _lola_isotropic_binning(
        tuple(x.shape[-2:]), device=x.device
    )
    s = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")
    p = torch.abs(s).square().flatten(start_dim=-2)

    p_iso = torch.zeros((*p.shape[:-1], edges.numel()), dtype=x.dtype, device=x.device)
    p_iso = p_iso.scatter_add(dim=-1, index=indices.expand_as(p), src=p)
    p_iso = p_iso / torch.clamp(counts.to(dtype=x.dtype), min=1)
    return p_iso[..., 1:], edges[1:]


def _lola_isotropic_cross_correlation(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = torch.broadcast_tensors(x, y)
    edges, counts, indices = _lola_isotropic_binning(
        tuple(x.shape[-2:]), device=x.device
    )

    sx = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")
    sy = torch.fft.fftn(y, dim=(-2, -1), norm="ortho")
    c = torch.abs(sx * torch.conj(sy)).flatten(start_dim=-2)

    c_iso = torch.zeros((*c.shape[:-1], edges.numel()), dtype=x.dtype, device=x.device)
    c_iso = c_iso.scatter_add(dim=-1, index=indices.expand_as(c), src=c)
    c_iso = c_iso / torch.clamp(counts.to(dtype=x.dtype), min=1)
    return c_iso[..., 1:], edges[1:]


def _lola_eval_reference_bands(
    y_pred: TensorBTSC, y_true: TensorBTSC, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirror Lola eval.py: treat prediction as a sample axis and average over samples.
    u = y_true[0, 0, ..., 0]
    v = y_pred[0, 0, ..., 0].unsqueeze(0)

    p_u, k = _lola_isotropic_power_spectrum(u)
    p_v, _ = _lola_isotropic_power_spectrum(v)
    p_v = p_v.mean(dim=0)

    c_uv, _ = _lola_isotropic_cross_correlation(u, v)
    c_uv = c_uv.mean(dim=0)

    se_p = torch.square(1.0 - (p_v + eps) / (p_u + eps))
    se_c = torch.square(1.0 - (c_uv + eps) / torch.sqrt(p_u * p_v + eps**2))

    bins = torch.logspace(k[0].log2(), -1.0, steps=4, base=2, device=k.device)

    power_bands = []
    cross_bands = []
    for i in range(4):
        mask = (
            torch.logical_and(bins[i] <= k, k <= bins[i + 1]) if i < 3 else bins[i] <= k
        )
        if torch.any(mask):
            power_bands.append(torch.sqrt(torch.mean(se_p[mask])))
            cross_bands.append(torch.sqrt(torch.mean(se_c[mask])))
        else:
            zero = torch.zeros((), dtype=u.dtype, device=u.device)
            power_bands.append(zero)
            cross_bands.append(zero)

    return torch.stack(power_bands), torch.stack(cross_bands)


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


@pytest.mark.parametrize(
    "MetricCls",
    [
        PowerSpectrumCCRMSE,
        PowerSpectrumCCRMSELow,
        PowerSpectrumCCRMSEMid,
        PowerSpectrumCCRMSEHigh,
        PowerSpectrumCCRMSETail,
    ],
)
def test_cross_correlation_rmse_nonzero_for_uncorrelated(MetricCls):
    torch.manual_seed(0)
    y_true: TensorBTSC = torch.randn((1, 1, 8, 8, 1))
    y_pred: TensorBTSC = torch.randn((1, 1, 8, 8, 1))

    value = MetricCls()(y_pred, y_true)
    assert torch.all(value > 0)


def test_cross_correlation_rmse_near_zero_for_identical():
    torch.manual_seed(0)
    y: TensorBTSC = torch.randn((1, 1, 8, 8, 1))
    value = PowerSpectrumCCRMSE()(y, y)
    # eps regularisation means result is near-zero, not exactly zero.
    assert torch.all(value < 1e-5)


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


@pytest.mark.parametrize(
    "MetricCls",
    [
        PowerSpectrumRMSE,
        PowerSpectrumRMSELow,
        PowerSpectrumRMSEMid,
        PowerSpectrumRMSEHigh,
        PowerSpectrumRMSETail,
        PowerSpectrumCCRMSE,
        PowerSpectrumCCRMSELow,
        PowerSpectrumCCRMSEMid,
        PowerSpectrumCCRMSEHigh,
        PowerSpectrumCCRMSETail,
    ],
)
def test_power_spectrum_metrics_device_consistency(MetricCls):
    target_device: torch.device | None = None
    if torch.cuda.is_available():
        target_device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        target_device = torch.device("mps")

    if target_device is None:
        pytest.skip("No GPU/MPS available for device test")

    torch.manual_seed(0)
    y_true: TensorBTSC = torch.randn((1, 1, 8, 8, 1), device=target_device)
    y_pred: TensorBTSC = torch.randn((1, 1, 8, 8, 1), device=target_device)

    # Evaluation should succeed without device mismatch errors
    value = MetricCls()(y_pred, y_true)

    assert value.device.type == target_device.type


def test_power_spectrum_metrics_match_lola_eval_reference():
    torch.manual_seed(0)
    y_true: TensorBTSC = torch.randn((1, 1, 32, 32, 1))
    y_pred: TensorBTSC = y_true + 0.15 * torch.randn_like(y_true)
    eps = 1e-6

    power_ref, cross_ref = _lola_eval_reference_bands(y_pred, y_true, eps=eps)

    assert torch.allclose(PowerSpectrumRMSELow(eps=eps)(y_pred, y_true), power_ref[0])
    assert torch.allclose(PowerSpectrumRMSEMid(eps=eps)(y_pred, y_true), power_ref[1])
    assert torch.allclose(PowerSpectrumRMSEHigh(eps=eps)(y_pred, y_true), power_ref[2])
    assert torch.allclose(PowerSpectrumRMSETail(eps=eps)(y_pred, y_true), power_ref[3])
    assert torch.allclose(
        PowerSpectrumRMSE(eps=eps)(y_pred, y_true), power_ref[:3].mean()
    )

    assert torch.allclose(PowerSpectrumCCRMSELow(eps=eps)(y_pred, y_true), cross_ref[0])
    assert torch.allclose(PowerSpectrumCCRMSEMid(eps=eps)(y_pred, y_true), cross_ref[1])
    assert torch.allclose(
        PowerSpectrumCCRMSEHigh(eps=eps)(y_pred, y_true), cross_ref[2]
    )
    assert torch.allclose(
        PowerSpectrumCCRMSETail(eps=eps)(y_pred, y_true), cross_ref[3]
    )
    assert torch.allclose(
        PowerSpectrumCCRMSE(eps=eps)(y_pred, y_true), cross_ref[:3].mean()
    )
