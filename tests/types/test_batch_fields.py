"""Tests for the field-driven tensor transforms on the batch dataclasses."""

from dataclasses import dataclass

import pytest
import torch

from autocast.types import Batch, EncodedBatch, EncodedSample, Sample
from autocast.types.batch import TensorFieldsMixin
from autocast.types.collation import collate_batches, collate_encoded_samples
from autocast.types.types import Tensor


@dataclass
class _BatchWithExtraField(TensorFieldsMixin):
    """Stands in for a future batch type that adds a field (e.g. forcing fields)."""

    input_fields: Tensor
    extra_fields: Tensor | None
    label: str


def _make_batch(**overrides) -> Batch:
    defaults = {
        "input_fields": torch.randn(2, 1, 4, 4, 1),
        "output_fields": torch.randn(2, 1, 4, 4, 1),
        "constant_scalars": torch.randn(2, 3),
        "constant_fields": torch.randn(2, 4, 4, 1),
        "boundary_conditions": torch.randn(2, 2),
    }
    return Batch(**{**defaults, **overrides})


# --- map_tensors ---


def test_map_tensors_applies_to_every_tensor_field():
    batch = _make_batch()

    doubled = batch.map_tensors(lambda tensor: tensor * 2)

    assert torch.equal(doubled.input_fields, batch.input_fields * 2)
    assert torch.equal(doubled.output_fields, batch.output_fields * 2)
    assert torch.equal(doubled.constant_scalars, batch.constant_scalars * 2)
    assert torch.equal(doubled.constant_fields, batch.constant_fields * 2)
    assert torch.equal(doubled.boundary_conditions, batch.boundary_conditions * 2)


def test_map_tensors_leaves_the_original_untouched():
    batch = _make_batch()
    original = batch.input_fields.clone()

    batch.map_tensors(lambda tensor: tensor * 2)

    assert torch.equal(batch.input_fields, original)


def test_map_tensors_passes_absent_optional_fields_through():
    batch = _make_batch(constant_scalars=None, boundary_conditions=None)

    mapped = batch.map_tensors(lambda tensor: tensor * 2)

    assert mapped.constant_scalars is None
    assert mapped.boundary_conditions is None


def test_map_tensors_descends_into_mapping_fields():
    batch = EncodedBatch(
        encoded_inputs=torch.randn(2, 4, 4, 1),
        encoded_output_fields=torch.randn(2, 4, 4, 1),
        global_cond=None,
        encoded_info={"mu": torch.randn(2, 3)},
    )

    mapped = batch.map_tensors(lambda tensor: tensor * 2)

    assert torch.equal(mapped.encoded_info["mu"], batch.encoded_info["mu"] * 2)


def test_map_tensors_preserves_the_concrete_type():
    assert isinstance(_make_batch().map_tensors(lambda t: t), Batch)


def test_map_tensors_covers_a_field_the_mixin_does_not_know_about():
    """A field added by a subclass is transformed without touching the mixin."""
    batch = _BatchWithExtraField(
        input_fields=torch.randn(2, 3),
        extra_fields=torch.randn(2, 5),
        label="untouched",
    )

    doubled = batch.map_tensors(lambda tensor: tensor * 2)

    assert torch.equal(doubled.extra_fields, batch.extra_fields * 2)
    assert doubled.label == "untouched"


def test_sample_types_do_not_carry_batch_lifecycle_transforms():
    """`Sample`/`EncodedSample` are pre-collation items, never pinned or moved.

    They stay plain dataclasses; `fields()`-driven collation reads them without
    needing the mixin.
    """
    for cls in (Sample, EncodedSample):
        assert not issubclass(cls, TensorFieldsMixin)
        for method in ("to", "pin_memory", "repeat", "map_tensors"):
            assert not hasattr(cls, method), f"{cls.__name__}.{method}"


# --- repeat ---


@pytest.mark.parametrize("cls", [Batch, EncodedBatch])
def test_repeat_interleaves_the_batch_dimension(cls):
    fields = torch.tensor([[1.0], [2.0]])
    batch = (
        Batch(fields, fields, None, None)
        if cls is Batch
        else EncodedBatch(fields, fields, None, {})
    )

    repeated = batch.repeat(3)

    leading = repeated.input_fields if cls is Batch else repeated.encoded_inputs
    assert leading.squeeze(-1).tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]


def test_repeat_covers_a_field_the_mixin_does_not_know_about():
    batch = _BatchWithExtraField(
        input_fields=torch.zeros(2, 3), extra_fields=torch.zeros(2, 5), label="x"
    )

    assert batch.repeat(3).extra_fields.shape == (6, 5)


# --- pin_memory ---


def test_pin_memory_noops_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch = _make_batch()

    pinned = batch.pin_memory()

    assert pinned.input_fields is batch.input_fields
    assert pinned.constant_scalars is batch.constant_scalars


# --- collation ---


def test_collate_batches_stacks_every_field():
    samples = [
        Sample(
            input_fields=torch.randn(1, 4, 4, 1),
            output_fields=torch.randn(1, 4, 4, 1),
            constant_scalars=torch.randn(3),
            constant_fields=torch.randn(4, 4, 1),
            boundary_conditions=torch.randn(2),
        )
        for _ in range(2)
    ]

    batch = collate_batches(samples)

    assert isinstance(batch, Batch)
    assert batch.input_fields.shape == (2, 1, 4, 4, 1)
    assert batch.constant_scalars.shape == (2, 3)
    assert torch.equal(batch.output_fields[1], samples[1].output_fields)


def test_collate_batches_keeps_a_uniformly_absent_field_as_none():
    samples = [
        Sample(torch.randn(1, 4, 4, 1), torch.randn(1, 4, 4, 1), None, None, None)
        for _ in range(2)
    ]

    batch = collate_batches(samples)

    assert batch.constant_scalars is None
    assert batch.boundary_conditions is None


def test_collate_batches_rejects_an_inconsistently_absent_field():
    samples = [
        Sample(
            torch.randn(1, 4, 4, 1), torch.randn(1, 4, 4, 1), torch.randn(3), None, None
        ),
        Sample(torch.randn(1, 4, 4, 1), torch.randn(1, 4, 4, 1), None, None, None),
    ]

    with pytest.raises(ValueError, match="inconsistently None"):
        collate_batches(samples)


def test_collate_batches_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one sample"):
        collate_batches([])


def test_collate_encoded_samples_merges_encoded_info():
    samples = [
        EncodedSample(
            encoded_inputs=torch.randn(4, 1),
            encoded_output_fields=torch.randn(4, 1),
            global_cond=torch.randn(3),
            encoded_info={"mu": torch.randn(2)},
        )
        for _ in range(2)
    ]

    batch = collate_encoded_samples(samples)

    assert isinstance(batch, EncodedBatch)
    assert batch.encoded_info["mu"].shape == (2, 2)
    assert batch.global_cond.shape == (2, 3)


def test_collate_encoded_samples_drops_keys_missing_from_some_samples():
    samples = [
        EncodedSample(
            torch.randn(4, 1), torch.randn(4, 1), None, {"mu": torch.randn(2)}
        ),
        EncodedSample(torch.randn(4, 1), torch.randn(4, 1), None, {}),
    ]

    batch = collate_encoded_samples(samples)

    assert "mu" not in batch.encoded_info
