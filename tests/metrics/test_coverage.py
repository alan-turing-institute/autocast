import torch

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
    results = metric.compute()

    assert set(results.keys()) == {"coverage_0.5", "coverage_0.9"}
    for value in results.values():
        assert torch.allclose(value, torch.tensor(1.0))
