import numpy as np
import pytest
import torch
from einops import repeat

from autocast.metrics.coverage import (
    Coverage,
    MultiCoverage,
    MultivariateRankHistogram,
    RankHistogram,
)
from autocast.types import TensorBTSC


def test_coverage_perfect():
    y_pred: TensorBTSC = torch.ones((2, 3, 4, 4, 5, 6))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 4, 5))

    metric = Coverage(coverage_level=0.9)
    metric.update(y_pred, y_true)
    value = metric.compute()

    assert torch.allclose(value, torch.tensor(1.0))


def test_coverage_miss():
    y_pred: TensorBTSC = torch.zeros((2, 3, 4, 4, 5, 6))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 4, 5))

    metric = Coverage(coverage_level=0.9)
    metric.update(y_pred, y_true)
    value = metric.compute()

    assert torch.allclose(value, torch.tensor(0.0))


def test_multi_alpha_coverage_dict_keys_and_values():
    y_pred: TensorBTSC = torch.ones((2, 3, 4, 4, 5, 6))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 4, 5))

    levels = [0.5, 0.9]
    metric = MultiCoverage(coverage_levels=levels)
    metric.update(y_pred, y_true)
    results = metric.compute_detailed()

    assert set(results.keys()) == {"coverage_0.5", "coverage_0.9"}
    for value in results.values():
        assert torch.isclose(torch.tensor(value), torch.tensor(1.0))


def test_coverage_partial():
    # Setup: 1 batch, 1 time, 4 spatial points, 1 channel, 10 ensemble members
    # y_pred ensemble values: 0, 1, ..., 9
    # Quantiles for 0.8 coverage (alpha=0.2): low=0.1, high=0.9
    # q_0.1 of 0..9 -> 0.9
    # q_0.9 of 0..9 -> 8.1
    # Interval: [0.9, 8.1]

    members = torch.arange(10).float()
    # Shape (B=1, T=1, S=4, C=1, M=10)
    y_pred = members.view(1, 1, 1, 1, 10).expand(1, 1, 4, 1, 10)

    # y_true values
    # 4.5 -> Inside [0.9, 8.1]
    # 4.5 -> Inside
    # -1.0 -> Outside
    # 11.0 -> Outside
    vals = torch.tensor([4.5, 4.5, -1.0, 11.0])
    y_true = vals.view(1, 1, 4, 1)

    value = Coverage(coverage_level=0.8)(y_pred, y_true)

    # expected coverage: 2 out of 4 -> 0.5
    assert torch.allclose(value, torch.tensor(0.5))


def test_coverage_multichannel_multitime():
    # Setup: 1 batch, 2 time, 1 spatial point, 2 channels, 10 ensemble members
    members = torch.arange(10).float()
    y_pred = repeat(members, "m -> b t s c m", b=1, t=2, s=1, c=2, m=10)

    # Add a single spatial point with different values for the two channels
    y_true = repeat(torch.tensor([5.0, 100.0]), "c -> b t s c", b=1, t=2, s=1)
    value = Coverage(coverage_level=0.8)(y_pred, y_true)

    # expected: average of 1.0 and 0.0 -> 0.5
    assert torch.allclose(value, torch.tensor(0.5))

    # Test case where reduce_all=False, so we get per-channel coverage per unit time
    value = Coverage(coverage_level=0.8, reduce_all=False)(y_pred, y_true)

    # expected: average of 1.0 and 0.0 -> 0.5
    assert torch.allclose(value, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))

    # Add a second spatial point with the opposite values to test spatial averaging
    y_true = repeat(
        torch.tensor([[5.0, 100.0], [0.8, 100.0]]), "c t -> b t s c", b=1, s=1
    )
    value = Coverage(coverage_level=0.8, reduce_all=False)(y_pred, y_true)
    assert torch.allclose(value, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))

    value = Coverage(coverage_level=0.8, reduce_all=True)(y_pred, y_true)
    # expected: average of 1.0, 0.0, 0.0, 0.0 -> 0.25
    assert torch.allclose(value, torch.tensor(0.25))


def test_multicoverage_save_csv_handles_1d_observed_channels(tmp_path):
    metric = MultiCoverage(coverage_levels=[0.5, 0.9])

    metric._save_csv_data(
        save_path=tmp_path / "coverage.png",
        levels=[0.5, 0.9],
        observed_means=[0.6, 0.8],
        observed_channels=np.array([0.6, 0.8]),
    )

    csv_path = tmp_path / "coverage.csv"
    assert csv_path.exists()


def test_rank_histogram_calibrated_is_flat():
    # Truth and members are exchangeable draws from the SAME distribution, so the
    # verification rank is uniform -> a flat histogram -> ~0 TV from uniform.
    torch.manual_seed(0)
    n_members = 15
    samples = torch.randn(1, 1, 20000, 1, n_members + 1)
    y_true = samples[..., 0]
    y_pred = samples[..., 1:]

    metric = RankHistogram()
    metric.update(y_pred, y_true)

    # n_members + 1 = 16 bins.
    assert len(metric.compute_detailed()["bin"]) == n_members + 1
    assert metric.compute().item() < 0.05


def test_rank_histogram_underdispersed_is_u_shaped():
    # Members far too narrow relative to the truth: the observation falls outside
    # the ensemble range, piling ranks at the two extremes (a U-shape).
    torch.manual_seed(0)
    n_members = 15
    y_pred = torch.randn(1, 1, 20000, 1, n_members) * 0.1
    y_true = torch.randn(1, 1, 20000, 1) * 1.0

    metric = RankHistogram()
    metric.update(y_pred, y_true)

    freqs = metric._frequencies()[0]  # (16,)
    assert (freqs[0] + freqs[-1]).item() > 0.5  # extremes dominate
    assert metric.compute().item() > 0.2  # far from flat


def test_rank_histogram_overdispersed_is_dome_shaped():
    # Members far too wide: the truth sits near the ensemble centre, depleting the
    # extreme ranks (a dome / inverted-U).
    torch.manual_seed(0)
    n_members = 15
    y_pred = torch.randn(1, 1, 20000, 1, n_members) * 3.0
    y_true = torch.randn(1, 1, 20000, 1) * 1.0

    metric = RankHistogram()
    metric.update(y_pred, y_true)

    freqs = metric._frequencies()[0]  # (16,)
    uniform = 1.0 / freqs.shape[0]
    assert freqs[0].item() < uniform  # extremes depleted
    assert freqs[-1].item() < uniform
    assert metric.compute().item() > 0.05


def test_rank_histogram_raises_before_update():
    metric = RankHistogram()
    with pytest.raises(RuntimeError):
        metric.compute()


def test_rank_histogram_rejects_shape_mismatch():
    metric = RankHistogram()
    y_pred = torch.randn(2, 3, 4, 4, 5, 6)
    y_true = torch.randn(2, 3, 4, 10, 5)
    with pytest.raises(ValueError):  # noqa: PT011
        metric.update(y_pred, y_true)


def test_rank_histogram_plot_saves_png_and_csv(tmp_path):
    torch.manual_seed(0)
    y_pred = torch.randn(1, 1, 256, 1, 16)
    y_true = torch.randn(1, 1, 256, 1)
    metric = RankHistogram()
    metric.update(y_pred, y_true)

    save_path = tmp_path / "rank_histogram.png"
    fig = metric.plot(save_path=str(save_path))

    assert fig is not None
    assert save_path.exists()
    assert save_path.with_suffix(".csv").exists()


def test_mv_rank_histogram_calibrated_is_flat():
    # Each instance ranks a whole spatial FIELD jointly. Obs and members are
    # exchangeable iid Gaussian fields -> the multivariate rank is uniform.
    torch.manual_seed(0)
    n_members = 8
    fields = torch.randn(2000, 1, 64, 1, n_members + 1)
    y_true = fields[..., 0]
    y_pred = fields[..., 1:]

    metric = MultivariateRankHistogram()
    metric.update(y_pred, y_true)

    assert len(metric.compute_detailed()["bin"]) == n_members + 1
    assert metric.compute().item() < 0.05


def test_mv_rank_histogram_joint_underdispersion_is_u_shaped():
    # Member fields far too tight relative to the observation field -> the obs
    # field sits jointly outside the ensemble, piling ranks at the extremes.
    torch.manual_seed(0)
    y_pred = torch.randn(2000, 1, 64, 1, 8) * 0.3
    y_true = torch.randn(2000, 1, 64, 1) * 1.0

    metric = MultivariateRankHistogram()
    metric.update(y_pred, y_true)

    freqs = metric._frequencies()[0]  # (9,)
    assert (freqs[0] + freqs[-1]).item() > 0.25  # extremes dominate
    assert metric.compute().item() > 0.1


def test_mv_rank_histogram_joint_overdispersion_is_dome_shaped():
    # Member fields far too wide -> obs field sits jointly near the centre,
    # depleting the extreme ranks (a dome).
    torch.manual_seed(0)
    y_pred = torch.randn(2000, 1, 64, 1, 8) * 3.0
    y_true = torch.randn(2000, 1, 64, 1) * 1.0

    metric = MultivariateRankHistogram()
    metric.update(y_pred, y_true)

    freqs = metric._frequencies()[0]  # (9,)
    uniform = 1.0 / freqs.shape[0]
    assert freqs[0].item() < uniform
    assert freqs[-1].item() < uniform
    assert metric.compute().item() > 0.1


def test_mv_rank_histogram_rejects_unimplemented_vector_dims():
    with pytest.raises(NotImplementedError):
        MultivariateRankHistogram(vector_dims="spatial_temporal")


def test_mv_rank_histogram_rejects_shape_mismatch():
    metric = MultivariateRankHistogram()
    y_pred = torch.randn(2, 3, 4, 4, 5, 6)
    y_true = torch.randn(2, 3, 4, 10, 5)
    with pytest.raises(ValueError):  # noqa: PT011
        metric.update(y_pred, y_true)


def test_mv_rank_histogram_accepts_integer_fields():
    # Integer-valued fields must not crash: the integer componentwise ranks are
    # averaged in a float dtype regardless of the input dtype (torch.mean has no
    # integer implementation).
    torch.manual_seed(0)
    y_pred = torch.randint(0, 5, (16, 1, 32, 1, 8))
    y_true = torch.randint(0, 5, (16, 1, 32, 1))
    metric = MultivariateRankHistogram()
    metric.update(y_pred, y_true)
    assert torch.isfinite(metric.compute()).item()
