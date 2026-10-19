import pytest
import torch

from autocast.processors.residual_normalization import ResidualStandardizer


def test_channel_statistics_broadcast_and_round_trip():
    standardizer = ResidualStandardizer(
        n_steps_output=3,
        n_channels=2,
        granularity="channel",
        mean=(1.0, -2.0),
        scale=(2.0, 4.0),
    )
    standardized = torch.full((2, 3, 4, 5, 2), 3.0)
    residual = standardizer.unstandardize(standardized)

    assert torch.equal(residual[..., 0], torch.full((2, 3, 4, 5), 7.0))
    assert torch.equal(residual[..., 1], torch.full((2, 3, 4, 5), 10.0))
    assert torch.equal(standardizer.standardize(residual), standardized)


def test_lead_channel_statistics_broadcast_and_round_trip():
    standardizer = ResidualStandardizer(
        n_steps_output=3,
        n_channels=1,
        granularity="lead_channel",
        mean=((1.0,), (2.0,), (3.0,)),
        scale=((0.5,), (1.0,), (2.0,)),
    )
    standardized = torch.ones(2, 3, 4, 5, 1)
    residual = standardizer.unstandardize(standardized)

    expected = torch.tensor((1.5, 3.0, 5.0)).view(1, 3, 1, 1, 1)
    assert torch.equal(residual, expected.expand_as(residual))
    assert torch.equal(standardizer.standardize(residual), standardized)


def test_unfitted_standardizer_fails_before_use():
    standardizer = ResidualStandardizer(n_steps_output=4, n_channels=1)

    with pytest.raises(RuntimeError, match="must be fitted"):
        standardizer.standardize(torch.empty(2, 4, 1, 1, 1))


def test_set_statistics_floors_zero_scale_and_persists_state():
    standardizer = ResidualStandardizer(
        n_steps_output=2,
        n_channels=1,
        epsilon=1e-4,
    )
    standardizer.set_statistics(mean=((1.0,), (2.0,)), scale=((0.0,), (3.0,)))
    restored = ResidualStandardizer(n_steps_output=2, n_channels=1)
    restored.load_state_dict(standardizer.state_dict())

    assert restored.fitted
    assert restored.scale[0, 0] == pytest.approx(1e-4)
    assert torch.equal(restored.mean, standardizer.mean)
    assert torch.equal(restored.scale, standardizer.scale)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"granularity": "space"}, "granularity"),
        ({"mean": 0.0}, "both be provided"),
        ({"scale": 1.0}, "both be provided"),
        ({"epsilon": 0.0}, "epsilon"),
    ],
)
def test_standardizer_configuration_is_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ResidualStandardizer(n_steps_output=4, n_channels=1, **kwargs)


def test_statistic_shape_is_validated():
    with pytest.raises(ValueError, match=r"shape \(4, 2\)"):
        ResidualStandardizer(
            n_steps_output=4,
            n_channels=2,
            mean=(0.0, 0.0),
            scale=(1.0, 1.0),
        )


def test_residual_shape_is_validated():
    standardizer = ResidualStandardizer(
        n_steps_output=4,
        n_channels=1,
        mean=0.0,
        scale=1.0,
    )

    with pytest.raises(ValueError, match="expected T=4, C=1"):
        standardizer.standardize(torch.empty(2, 3, 1, 1, 1))


def test_negative_or_nonfinite_statistics_are_rejected():
    standardizer = ResidualStandardizer(n_steps_output=2, n_channels=1)

    with pytest.raises(ValueError, match="non-negative"):
        standardizer.set_statistics(mean=0.0, scale=-1.0)
    with pytest.raises(ValueError, match="finite"):
        standardizer.set_statistics(mean=float("nan"), scale=1.0)
