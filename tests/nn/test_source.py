import math

import pytest
import torch

from autocast.nn.noise.source import (
    IIDGaussianSource,
    SeparableGaussianSource,
    TemporalOUSource,
    ZeroSource,
)


@pytest.mark.parametrize("source", [ZeroSource(), IIDGaussianSource()])
def test_simple_sources_preserve_shape_device_and_dtype(source):
    reference = torch.empty(2, 4, 8, 8, 3, dtype=torch.float64)

    samples = source.sample_like(reference)

    assert samples.shape == reference.shape
    assert samples.device == reference.device
    assert samples.dtype == reference.dtype


def test_zero_source_is_deterministic():
    reference = torch.empty(2, 4, 3)

    assert torch.count_nonzero(ZeroSource().sample_like(reference)) == 0


def test_temporal_ou_source_has_stationary_target_correlation():
    source = TemporalOUSource(correlation_time=4.0)
    samples = source.sample_like(torch.empty(4096, 6, 1, 1, 1))
    variance = samples.square().mean()
    lag_one = (samples[:, 1:] * samples[:, :-1]).mean() / variance

    assert samples.std(correction=0) == pytest.approx(1.0, abs=0.03)
    assert lag_one == pytest.approx(math.exp(-0.25), abs=0.03)


def test_infinite_ou_correlation_time_shares_noise_across_frames():
    samples = TemporalOUSource(correlation_time=math.inf).sample_like(
        torch.empty(2, 4, 3, 3, 1)
    )

    assert torch.equal(samples, samples[:, :1].expand_as(samples))


def test_separable_gp_has_temporal_and_spatial_correlation():
    source = SeparableGaussianSource(
        temporal_correlation_time=2.0,
        spatial_length_scale=2.0,
    )
    samples = source.sample_like(torch.empty(512, 4, 16, 16, 1))
    variance = samples.square().mean()
    temporal = (samples[:, 1:] * samples[:, :-1]).mean() / variance
    spatial = (samples[:, :, :, 1:] * samples[:, :, :, :-1]).mean() / variance

    assert samples.std(correction=0) == pytest.approx(1.0, abs=0.1)
    assert temporal > 0.5
    assert spatial > 0.5


def test_separable_gp_preserves_low_precision_dtype():
    source = SeparableGaussianSource(
        temporal_correlation_time=2.0,
        spatial_length_scale=2.0,
    )

    samples = source.sample_like(torch.empty(2, 4, 8, 8, 1, dtype=torch.bfloat16))

    assert samples.dtype == torch.bfloat16


def test_separable_gp_applies_channel_correlation():
    source = SeparableGaussianSource(
        temporal_correlation_time=2.0,
        spatial_length_scale=2.0,
        channel_correlation=((1.0, 0.75), (0.75, 1.0)),
    )
    samples = source.sample_like(torch.empty(512, 4, 8, 8, 2))
    observed = (samples[..., 0] * samples[..., 1]).mean() / (
        samples[..., 0].std(correction=0) * samples[..., 1].std(correction=0)
    )

    assert observed == pytest.approx(0.75, abs=0.05)


@pytest.mark.parametrize(
    ("correlation", "message"),
    [
        (((1.0, 0.0),), "square"),
        (((1.0, 0.5), (0.0, 1.0)), "symmetric"),
        (((2.0, 0.0), (0.0, 1.0)), "unit diagonal"),
        (((1.0, 2.0), (2.0, 1.0)), "positive definite"),
    ],
)
def test_separable_gp_validates_channel_correlation(correlation, message):
    with pytest.raises(ValueError, match=message):
        SeparableGaussianSource(
            temporal_correlation_time=2.0,
            spatial_length_scale=2.0,
            channel_correlation=correlation,
        )


def test_separable_gp_validates_reference_channel_count():
    source = SeparableGaussianSource(
        temporal_correlation_time=2.0,
        spatial_length_scale=2.0,
        channel_correlation=((1.0, 0.5), (0.5, 1.0)),
    )

    with pytest.raises(ValueError, match="expected 2, got 1"):
        source.sample_like(torch.empty(2, 4, 8, 8, 1))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: TemporalOUSource(correlation_time=0.0), "correlation_time"),
        (
            lambda: SeparableGaussianSource(
                temporal_correlation_time=1.0,
                spatial_length_scale=0.0,
            ),
            "spatial_length_scale",
        ),
    ],
)
def test_correlated_sources_validate_length_scales(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_correlated_sources_require_spatiotemporal_fields():
    with pytest.raises(ValueError, match=r"\(B, T, Y, X, C\)"):
        TemporalOUSource(correlation_time=1.0).sample_like(torch.empty(2, 4, 3))
