"""Training rollouts retain member paths, normalized feedback and gradients."""

from unittest.mock import Mock

import pytest
import torch
from einops import rearrange
from the_well.data.normalization import ZScoreNormalization
from torch import nn

from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.identity import IdentityEncoder
from autocast.metrics.deterministic import MSE
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.processors.base import Processor
from autocast.types import Batch


class _Transition(Processor):
    def __init__(self, stochastic=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.stochastic = stochastic
        self.inputs: list[torch.Tensor] = []
        self.draws: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def map(self, x, global_cond=None):  # noqa: ARG002
        self.inputs.append(x.detach().clone())
        noise = (
            torch.randn_like(x[:, -1:])
            if self.stochastic
            else torch.zeros_like(x[:, -1:])
        )
        self.draws.append(noise)
        output = self.weight * x[:, -1:] + noise
        self.outputs.append(output)
        return output

    def loss(self, batch):  # noqa: ARG002
        raise AssertionError


class _MemberMSE(nn.Module):
    def forward(self, prediction, target):
        if prediction.ndim == target.ndim + 1:
            target = target.unsqueeze(-1)
        return (prediction - target).square().mean()


def _model(members=None, **kwargs):
    cls = EncoderProcessorDecoder
    if members is not None:
        cls = EncoderProcessorDecoderEnsemble
        kwargs["n_members"] = members
    return cls(
        encoder_decoder=EncoderDecoder(IdentityEncoder(1), IdentityDecoder(1)),
        processor=_Transition(kwargs.pop("stochastic", False)),
        loss_func=_MemberMSE(),
        residual_prediction=True,
        residual_use_delta_stats=False,
        **kwargs,
    )


def _transition(model: EncoderProcessorDecoder) -> _Transition:
    assert isinstance(model.processor, _Transition)
    return model.processor


def _batch():
    return Batch(torch.ones(2, 1, 2, 3, 1), torch.zeros(2, 2, 2, 3, 1), None, None)


@pytest.mark.parametrize("members", [None, 3])
def test_one_step_rng_weights_and_loss_equal_original(members):
    batch = _batch()
    batch.output_fields = batch.output_fields[:, :1]
    original = _model(members, stochastic=True)
    control = _model(members, stochastic=True, prediction_steps=1)
    control.load_state_dict(original.state_dict(), strict=True)
    torch.manual_seed(91)
    loss_a, pred_a = original.loss(batch)
    state_a = torch.get_rng_state()
    torch.manual_seed(91)
    loss_b, pred_b = control.loss(batch)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    torch.testing.assert_close(pred_a, pred_b, rtol=0, atol=0)
    assert torch.equal(state_a, torch.get_rng_state())
    assert control.state_dict().keys() == original.state_dict().keys()


def test_two_calls_keep_member_paths_and_redraw_noise_without_probe():
    batch = _batch()
    model = _model(3, stochastic=True, prediction_steps=1, train_rollout_steps=2)
    loss, pred = model.loss(batch)
    assert pred is not None
    assert model.loss_func is not None
    assert len(_transition(model).inputs) == len(_transition(model).draws) == 2
    assert all(x.shape[0] == 6 for x in _transition(model).inputs)
    assert pred.shape == (2, 2, 2, 3, 1, 3)
    first = rearrange(pred[:, :1], "b t ... c m -> (b m) t ... c")
    torch.testing.assert_close(_transition(model).inputs[1], first)
    assert not torch.equal(_transition(model).draws[0], _transition(model).draws[1])
    expected = (
        sum(
            model.loss_func(pred[:, i : i + 1], batch.output_fields[:, i : i + 1])
            for i in range(2)
        )
        / 2
    )
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("detach", "expected_gradient"), [(False, 8.25), (True, 4.875)]
)
def test_two_step_gradients_include_feedback_unless_detached(detach, expected_gradient):
    model = _model(
        prediction_steps=1, train_rollout_steps=2, detach_rollout_feedback=detach
    )
    loss, pred = model.loss(_batch())
    assert pred is not None
    torch.testing.assert_close(pred[:, 0], torch.full_like(pred[:, 0], 1.5))
    torch.testing.assert_close(pred[:, 1], torch.full_like(pred[:, 1], 2.25))
    loss.backward()
    gradient = _transition(model).weight.grad
    assert gradient is not None
    assert gradient.item() == pytest.approx(expected_gradient)


def test_one_step_training_and_two_step_validation_use_same_target_window():
    model = _model(prediction_steps=1, validation_rollout_steps=2)
    model.log = Mock()
    batch = _batch()
    assert model.training_step(batch, 0).item() == pytest.approx(1.5**2)
    assert len(_transition(model).inputs) == 1
    assert model.validation_step(batch, 0).item() == pytest.approx(
        (1.5**2 + 2.25**2) / 2
    )
    assert len(_transition(model).inputs) == 3


def test_test_step_uses_validation_horizon_and_aligned_metric_targets():
    model = _model(prediction_steps=1, validation_rollout_steps=2, test_metrics=[MSE()])
    model.log = Mock()
    model._update_and_log_metrics = Mock()
    expected = (1.5**2 + 2.25**2) / 2
    assert model.test_step(_batch(), 0).item() == pytest.approx(expected)
    assert len(_transition(model).inputs) == 2
    _, _, pred, target, _ = model._update_and_log_metrics.call_args.args
    assert pred.shape == target.shape == (2, 2, 2, 3, 1)


def test_validation_can_score_same_predictions_with_separate_objective():
    class _ZeroLoss(nn.Module):
        def forward(self, prediction, target):  # noqa: ARG002
            return prediction.sum() * 0

    model = _model(prediction_steps=1, validation_loss_func=_ZeroLoss())
    model.log = Mock()
    assert model.training_step(_batch(), 0) > 0
    assert model.validation_step(_batch(), 0) == 0
    assert len(_transition(model).inputs) == 2


@pytest.mark.parametrize("members", [None, 3])
def test_rollout_restores_persistence_with_delta_stats(members):
    norm = ZScoreNormalization(
        stats={
            "mean": {"u": 100.0},
            "std": {"u": 2.0},
            "mean_delta": {"u": 0.0},
            "std_delta": {"u": 0.2},
        },
        core_field_names=["u"],
        core_constant_field_names=[],
    )
    model = _model(members, prediction_steps=1, train_rollout_steps=2, norm=norm)
    model.residual_use_delta_stats = True
    _, pred = model.loss(_batch())
    assert pred is not None
    torch.testing.assert_close(pred[:, 0], torch.full_like(pred[:, 0], 1.05))
    torch.testing.assert_close(pred[:, 1], torch.full_like(pred[:, 1], 1.05**2))


def test_rollout_rejects_short_target_before_running_model():
    model = _model(prediction_steps=1, train_rollout_steps=3)
    with pytest.raises(ValueError, match="Target window"):
        model.loss(_batch())
    assert not _transition(model).inputs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_rollout_steps": 0},
        {"train_rollout_steps": True},
        {"validation_rollout_steps": 1.5},
        {"prediction_steps": 0},
        {"validation_rollout_steps": 2},
        {"prediction_steps": 2, "train_rollout_steps": 2},
        {
            "prediction_steps": 1,
            "train_rollout_steps": 2,
            "train_in_latent_space": True,
        },
    ],
)
def test_invalid_rollout_configuration_rejected(kwargs):
    with pytest.raises(ValueError, match=r"(positive|require)"):
        _model(**kwargs)
