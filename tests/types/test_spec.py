import pytest

from autocast.types import FieldSpec, IOSpec


def test_field_spec_from_batch_shape_splits_time_spatial_and_channels():
    spec = FieldSpec.from_batch_shape((4, 2, 16, 8, 3))

    assert spec.n_steps == 2
    assert spec.spatial_resolution == (16, 8)
    assert spec.n_channels == 3


def test_field_spec_from_batch_shape_handles_a_single_spatial_dim():
    spec = FieldSpec.from_batch_shape((4, 2, 16, 3))

    assert spec.spatial_resolution == (16,)


def test_field_spec_from_batch_shape_rejects_too_few_dims():
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        FieldSpec.from_batch_shape((4, 2))


def test_n_flat_channels_folds_time_into_channels():
    spec = FieldSpec(n_steps=3, n_channels=4, spatial_resolution=(8, 8))

    assert spec.n_flat_channels == 12


def test_io_spec_from_batch_shapes_keeps_the_two_sides_distinct():
    spec = IOSpec.from_batch_shapes((4, 2, 16, 16, 3), (4, 1, 16, 16, 1))

    assert spec.inputs.n_steps == 2
    assert spec.inputs.n_channels == 3
    assert spec.outputs.n_steps == 1
    assert spec.outputs.n_channels == 1


def test_is_autoregressive_true_when_only_step_counts_differ():
    spec = IOSpec.from_batch_shapes((4, 4, 16, 16, 3), (4, 1, 16, 16, 3))

    assert spec.is_autoregressive


def test_is_autoregressive_false_for_differing_channels():
    spec = IOSpec.from_batch_shapes((4, 2, 16, 16, 3), (4, 2, 16, 16, 1))

    assert not spec.is_autoregressive


def test_is_autoregressive_false_for_differing_resolution():
    spec = IOSpec.from_batch_shapes((4, 2, 16, 16, 3), (4, 2, 32, 32, 3))

    assert not spec.is_autoregressive
