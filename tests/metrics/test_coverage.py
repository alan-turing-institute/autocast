import torch
from einops import repeat

from autocast.metrics.coverage import Coverage, MultiCoverage
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
