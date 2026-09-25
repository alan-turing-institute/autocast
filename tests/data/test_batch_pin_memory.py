import torch

from autocast.types import Batch, EncodedBatch


def test_batch_pin_memory_noops_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch = Batch(
        input_fields=torch.randn(2, 1, 4, 4, 1),
        output_fields=torch.randn(2, 1, 4, 4, 1),
        constant_scalars=torch.randn(2, 1),
        constant_fields=torch.randn(2, 4, 4, 1),
        boundary_conditions=torch.randn(2, 2),
    )

    pinned = batch.pin_memory()

    assert pinned.input_fields is batch.input_fields
    assert pinned.output_fields is batch.output_fields
    assert pinned.constant_scalars is batch.constant_scalars
    assert pinned.constant_fields is batch.constant_fields
    assert pinned.boundary_conditions is batch.boundary_conditions


def test_encoded_batch_pin_memory_noops_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch = EncodedBatch(
        encoded_inputs=torch.randn(2, 1, 4, 4, 1),
        encoded_output_fields=torch.randn(2, 4, 4, 1),
        global_cond=torch.randn(2, 3),
        encoded_info={"x": torch.randn(2, 1)},
    )

    pinned = batch.pin_memory()

    assert pinned.encoded_inputs is batch.encoded_inputs
    assert pinned.encoded_output_fields is batch.encoded_output_fields
    assert pinned.global_cond is batch.global_cond
    assert pinned.encoded_info["x"] is batch.encoded_info["x"]
