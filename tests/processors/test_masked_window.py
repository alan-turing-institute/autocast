"""Masking, ensemble accounting and fixed-context inference contracts."""

import pytest
import torch
from einops import rearrange, repeat
from torch import nn

from autocast.losses.ensemble import AlphaFairCRPSLoss
from autocast.models.processor_ensemble import ProcessorModelEnsemble
from autocast.processors.flow_matching_masked_window import (
    FlowMatchingMaskedWindowProcessor,
)
from autocast.processors.masking import masked_mean, random_context_mask
from autocast.processors.spatiotemporal_vit import (
    SpatiotemporalViTMaskedWindowProcessor,
)
from autocast.types import EncodedBatch


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _batch():
    return EncodedBatch(
        encoded_inputs=torch.full((2, 1, 2, 2, 1), 100.0),
        encoded_output_fields=torch.zeros(2, 4, 2, 2, 1),
        global_cond=torch.tensor([[2.0], [4.0]]),
        encoded_info={},
    )


class _ContextMask:
    """Different forecast lengths to distinguish per-example and global means."""

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, x):
        self.batch_sizes.append(x.shape[0])
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[0, :1] = True
        mask[1, :4] = True
        return mask


class _WindowBackbone(nn.Module):
    conditioning = "mask"
    use_precomputed_modulation = True
    n_steps_input = n_steps_output = 5
    in_channels = cond_channels = 1
    mod_features = 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.seen = []

    def forward(self, x, t, cond, global_cond):
        self.seen.append((x.detach().clone(), t.detach().clone(), cond.clone()))
        value = rearrange(global_cond[:, 0], "b -> b 1 1 1 1")
        return self.scale * torch.where(cond.bool(), 100.0, value).expand_as(x)


def _ensemble_processor(**overrides):
    backbone, context_mask = _WindowBackbone(), _ContextMask()
    kwargs = {
        "backbone": backbone,
        "n_steps_input": 1,
        "n_steps_output": 4,
        "n_members": 3,
        "loss_func": AlphaFairCRPSLoss(reduction="none"),
        "context_mask": context_mask,
        "n_noise_input_channels": 4,
    }
    processor = SpatiotemporalViTMaskedWindowProcessor(**(kwargs | overrides))
    return processor, backbone, context_mask


def test_crps_mask_is_shared_before_member_expansion_and_loss_is_per_example():
    processor, backbone, context_mask = _ensemble_processor()
    # Use the actual ensemble model fallback, not processor.loss in isolation.
    model = ProcessorModelEnsemble(processor, n_members=3, loss_func=None)
    loss = model.loss(_batch())
    assert context_mask.batch_sizes == [2]
    clean, noise, mask = backbone.seen[0]
    clean = rearrange(clean, "(b m) ... -> b m ...", m=3)
    mask = rearrange(mask, "(b m) ... -> b m ...", m=3)
    assert torch.equal(clean[:, 0], clean[:, 1])
    assert torch.equal(mask[:, 0], mask[:, 1])
    assert torch.equal(mask[:, 1], mask[:, 2])
    assert not torch.equal(noise[0], noise[1])
    assert torch.count_nonzero(clean[~mask.bool()]) == 0
    # Unknown scores are 2 and 4 regardless of forecast length: mean = 3.
    # A global unknown-element mean would be 2.4; a full-window mean, 1.2.
    torch.testing.assert_close(loss, torch.tensor(3.0))
    loss.backward()
    torch.testing.assert_close(backbone.scale.grad, torch.tensor(3.0))


def test_crps_unobserved_targets_do_not_enter_the_generator():
    processor, backbone, context_mask = _ensemble_processor()
    batch = _batch()
    mask = context_mask(
        torch.cat((batch.encoded_inputs, batch.encoded_output_fields), dim=1)
    )
    torch.manual_seed(8)
    original_loss = processor.loss(batch)
    original_input = backbone.seen[-1]
    batch.encoded_output_fields[~mask[:, 1:]] = 20
    torch.manual_seed(8)
    changed_loss = processor.loss(batch)
    changed_input = backbone.seen[-1]
    assert not torch.equal(original_loss, changed_loss)
    for original, changed in zip(original_input, changed_input, strict=True):
        torch.testing.assert_close(original, changed)


def test_crps_inference_uses_fixed_context_without_sampling_a_training_mask():
    processor, backbone, context_mask = _ensemble_processor()
    batch = _batch()
    model = ProcessorModelEnsemble(processor, n_members=3, loss_func=None)
    pred = model(batch.encoded_inputs, batch.global_cond)
    assert pred.shape == (*batch.encoded_output_fields.shape, 3)
    assert context_mask.batch_sizes == []
    clean, _, mask = backbone.seen[0]
    assert mask[:, :1].bool().all()
    assert not mask[:, 1:].bool().any()
    assert torch.count_nonzero(clean[:, 1:]) == 0
    torch.testing.assert_close(
        clean[:, :1], repeat(batch.encoded_inputs, "b ... -> (b m) ...", m=3)
    )


def test_crps_requires_unreduced_scores():
    processor, _, _ = _ensemble_processor(loss_func=AlphaFairCRPSLoss())
    with pytest.raises(ValueError, match="unreduced score"):
        processor.loss(_batch())


def test_fm_forecast_loss_ignores_observed_frames_and_balances_examples(monkeypatch):
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    processor = FlowMatchingMaskedWindowProcessor(
        backbone=_WindowBackbone(),
        n_steps_input=1,
        n_steps_output=4,
        n_channels_out=1,
        context_mask=_ContextMask(),
        loss_on_observed=False,
    )
    loss = processor.loss(_batch())
    torch.testing.assert_close(loss, torch.tensor(10.0))
    loss.backward()
    torch.testing.assert_close(
        processor.flow_matching_model.scale.grad, torch.tensor(20.0)
    )
    # Legacy presets still penalize the observed-frame velocity as before.
    processor.loss_on_observed = True
    assert processor.loss(_batch()) > 1000


def test_legacy_pickled_fm_preserves_loss_and_gradients(tmp_path):
    backbone = _WindowBackbone()
    processor = FlowMatchingMaskedWindowProcessor(
        backbone=backbone,
        n_steps_input=1,
        n_steps_output=4,
        n_channels_out=1,
        atleast=1,
    )
    torch.manual_seed(17)
    expected = processor.loss(_batch())
    expected.backward()
    assert backbone.scale.grad is not None
    expected_grad = backbone.scale.grad.clone()
    processor.zero_grad(set_to_none=True)

    # Older full-object checkpoints lack these newly introduced options.
    del processor.context_mask
    del processor.loss_on_observed
    checkpoint = tmp_path / "legacy.pt"
    torch.save(processor, checkpoint)
    restored = torch.load(checkpoint, weights_only=False)

    torch.manual_seed(17)
    actual = restored.loss(_batch())
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(restored.flow_matching_model.scale.grad, expected_grad)


@pytest.mark.parametrize("broadcast", [False, True])
def test_masked_mean_balances_examples_and_masks_gradients(broadcast):
    score = torch.tensor([[2.0, 2.0, 999.0], [4.0, 999.0, 999.0]], requires_grad=True)
    selected = torch.tensor([[True, True, False], [True, False, False]])
    values = repeat(score, "b t -> b t s", s=3) if broadcast else score
    mask = rearrange(selected, "b t -> b t 1") if broadcast else selected
    loss = masked_mean(values, mask)
    torch.testing.assert_close(loss, torch.tensor(3.0))
    loss.backward()
    torch.testing.assert_close(score.grad, torch.tensor([[0.25, 0.25, 0], [0.5, 0, 0]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_masked_mean_accumulates_large_low_precision_windows_safely(dtype):
    score = torch.ones(1, 5, 64, 64, 4, dtype=dtype, requires_grad=True)
    mask = torch.tensor([False, True, True, True, True])[None, :, None, None, None]
    loss = masked_mean(score, mask)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert score.grad is not None
    assert torch.isfinite(score.grad).all()
    assert torch.count_nonzero(score.grad[:, :1]) == 0
    torch.testing.assert_close(
        score.grad[:, 1:], torch.full_like(score[:, 1:], 1 / 65536)
    )


@pytest.mark.parametrize(
    ("rho", "expected_rows"),
    [
        (0.0, ["00001", "00011", "01111", "00001", "00111"]),
        (0.5, ["10000", "00011", "01111", "10000", "00111"]),
        (1.0, ["10000", "11000", "11110", "10000", "11100"]),
    ],
)
def test_shared_sampler_matches_expected_masks(monkeypatch, rho, expected_rows):
    def poisson(rate):
        torch.testing.assert_close(rate, torch.full((5, 1), 2.0))
        # Five-frame windows with atleast=1 wrap these to 1, 2, 4, 1, 3.
        return rate.new_tensor([[0], [1], [3], [4], [6]])

    def uniform(shape, *, device):
        assert shape == (5, 1)
        # The draw equal to rho=0.5 selects a suffix, not a prefix.
        return torch.tensor([[0.1], [0.5], [0.9], [0.2], [0.8]], device=device)

    monkeypatch.setattr(torch, "poisson", poisson)
    monkeypatch.setattr(torch, "rand", uniform)
    x = torch.empty(5, 5, 2, 3, 2)
    actual = random_context_mask(x, lmbda=2.0, rho=rho, atleast=1)
    # Explicit observation masks, independent of the sampler's implementation.
    expected = torch.tensor([[bit == "1" for bit in row] for row in expected_rows])
    torch.testing.assert_close(
        actual, repeat(expected, "b t -> b t h w c", h=2, w=3, c=2)
    )
