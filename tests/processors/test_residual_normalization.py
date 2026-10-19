import pytest
import torch

from autocast.processors import residual_normalization
from autocast.processors.residual_normalization import (
    ResidualNormalizer,
    ResidualStandardizer,
)


def test_residual_normalizer_has_explicit_names_and_legacy_aliases():
    assert hasattr(residual_normalization, "ResidualNormalizer")
    normalizer = residual_normalization.ResidualNormalizer(
        n_steps_output=2, n_channels=1, mean=2.0, scale=3.0
    )
    residual = torch.full((1, 2, 3, 1), 8.0)
    normalized = normalizer.normalize(residual)

    assert torch.equal(normalized, torch.full_like(residual, 2.0))
    assert torch.equal(normalizer.denormalize(normalized), residual)
    assert ResidualStandardizer is residual_normalization.ResidualNormalizer
    assert torch.equal(normalizer.standardize(residual), normalized)
    assert torch.equal(normalizer.unstandardize(normalized), residual)
    assert torch.equal(normalizer.denormalize(normalized_residual=normalized), residual)
    assert torch.equal(
        normalizer.unstandardize(standardized_residual=normalized), residual
    )


def test_channel_statistics_broadcast_and_round_trip():
    residual_normalizer = ResidualNormalizer(
        n_steps_output=3,
        n_channels=2,
        granularity="channel",
        mean=(1.0, -2.0),
        scale=(2.0, 4.0),
    )
    standardized = torch.full((2, 3, 4, 5, 2), 3.0)
    residual = residual_normalizer.denormalize(standardized)

    assert torch.equal(residual[..., 0], torch.full((2, 3, 4, 5), 7.0))
    assert torch.equal(residual[..., 1], torch.full((2, 3, 4, 5), 10.0))
    assert torch.equal(residual_normalizer.normalize(residual), standardized)


def test_lead_channel_statistics_broadcast_and_round_trip():
    residual_normalizer = ResidualNormalizer(
        n_steps_output=3,
        n_channels=1,
        granularity="lead_channel",
        mean=((1.0,), (2.0,), (3.0,)),
        scale=((0.5,), (1.0,), (2.0,)),
    )
    standardized = torch.ones(2, 3, 4, 5, 1)
    residual = residual_normalizer.denormalize(standardized)

    expected = torch.tensor((1.5, 3.0, 5.0)).view(1, 3, 1, 1, 1)
    assert torch.equal(residual, expected.expand_as(residual))
    assert torch.equal(residual_normalizer.normalize(residual), standardized)


def test_unfitted_residual_normalizer_fails_before_use():
    residual_normalizer = ResidualNormalizer(n_steps_output=4, n_channels=1)

    with pytest.raises(RuntimeError, match="must be fitted"):
        residual_normalizer.normalize(torch.empty(2, 4, 1, 1, 1))


def test_set_statistics_floors_zero_scale_and_persists_state():
    residual_normalizer = ResidualNormalizer(
        n_steps_output=2,
        n_channels=1,
        epsilon=1e-4,
    )
    residual_normalizer.set_statistics(mean=((1.0,), (2.0,)), scale=((0.0,), (3.0,)))
    restored = ResidualNormalizer(n_steps_output=2, n_channels=1)
    restored.load_state_dict(residual_normalizer.state_dict())

    assert restored.fitted
    assert restored.scale[0, 0] == pytest.approx(1e-4)
    assert torch.equal(restored.mean, residual_normalizer.mean)
    assert torch.equal(restored.scale, residual_normalizer.scale)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"granularity": "space"}, "granularity"),
        ({"mean": 0.0}, "both be provided"),
        ({"scale": 1.0}, "both be provided"),
        ({"epsilon": 0.0}, "epsilon"),
    ],
)
def test_residual_normalizer_configuration_is_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ResidualNormalizer(n_steps_output=4, n_channels=1, **kwargs)


def test_statistic_shape_is_validated():
    with pytest.raises(ValueError, match=r"shape \(4, 2\)"):
        ResidualNormalizer(
            n_steps_output=4,
            n_channels=2,
            mean=(0.0, 0.0),
            scale=(1.0, 1.0),
        )


def test_residual_shape_is_validated():
    residual_normalizer = ResidualNormalizer(
        n_steps_output=4,
        n_channels=1,
        mean=0.0,
        scale=1.0,
    )

    with pytest.raises(ValueError, match="expected T=4, C=1"):
        residual_normalizer.normalize(torch.empty(2, 3, 1, 1, 1))


def test_negative_or_nonfinite_statistics_are_rejected():
    residual_normalizer = ResidualNormalizer(n_steps_output=2, n_channels=1)

    with pytest.raises(ValueError, match="non-negative"):
        residual_normalizer.set_statistics(mean=0.0, scale=-1.0)
    with pytest.raises(ValueError, match="finite"):
        residual_normalizer.set_statistics(mean=float("nan"), scale=1.0)
