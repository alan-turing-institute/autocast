from unittest.mock import Mock

import pytest
import torch
from einops import rearrange, repeat
from the_well.data.normalization import RMSNormalization, ZScoreNormalization
from torch import nn

from autocast.decoders.channels_last import ChannelsLast
from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.identity import IdentityEncoder
from autocast.encoders.permute_concat import PermuteConcat
from autocast.metrics.deterministic import MSE
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.nn.noise.noise_injector import ConcatenatedNoiseInjector
from autocast.processors.base import Processor
from autocast.processors.fno import FNOProcessor
from autocast.types import Batch

STATE_MEAN, STATE_STD = torch.tensor([10.0, -20.0]), torch.tensor([2.0, 5.0])
DELTA_MEAN, DELTA_STD = torch.tensor([0.5, -1.0]), torch.tensor([0.25, 2.0])


def physical_state(state, use_normalization=True):
    return state * STATE_STD + STATE_MEAN if use_normalization else state


def physical_delta(
    delta, use_normalization=True, use_delta_stats=True, independent_delta_stats=False
):
    if use_delta_stats and (use_normalization or independent_delta_stats):
        return delta * DELTA_STD + DELTA_MEAN
    return delta * STATE_STD if use_normalization else delta


def with_members(tensor, n_members):
    return tensor if n_members is None else repeat(tensor, "... -> ... m", m=n_members)


class DecodedOutputProcessor(Processor):
    def __init__(self, output, member_offsets=False):
        super().__init__()
        self.output = nn.Parameter(output.clone())
        self.member_offsets = member_offsets
        self.last_input = None

    def map(self, x, global_cond=None):  # noqa: ARG002
        self.last_input = x.detach().clone()
        output = repeat(self.output, "t ... -> b t ...", b=x.shape[0])
        if self.member_offsets:
            offset = torch.arange(x.shape[0], device=x.device, dtype=x.dtype)
            output = output + rearrange(offset, "b -> b 1 1 1 1")
        return output

    def loss(self, batch):  # noqa: ARG002
        msg = "Decoded residuals must not enter a latent objective"
        raise AssertionError(msg)


class CaptureLoss(nn.Module):
    def forward(self, prediction, target):
        self.prediction = prediction
        if prediction.ndim == target.ndim + 1:
            target = rearrange(target, "... -> ... 1")
        return (prediction - target).square().mean()


@pytest.fixture
def norm():
    return ZScoreNormalization(
        stats={
            "mean": {"u": 10.0, "v": -20.0},
            "std": {"u": 2.0, "v": 5.0},
            "mean_delta": {"u": 0.5, "v": -1.0},
            "std_delta": {"u": 0.25, "v": 2.0},
        },
        core_field_names=["u", "v"],
        core_constant_field_names=[],
    )


def make_batch(t_input=3, t_output=4):
    # Two batch items on a 2x3 grid with two state fields.
    inputs = torch.arange(24 * t_input, dtype=torch.float32).reshape(
        2, t_input, 2, 3, 2
    )
    outputs = repeat(torch.arange(100.0, 100 + t_output), "t -> 2 t 2 3 2")
    return Batch(inputs, outputs, None, None)


def make_model(output, n_members=None, member_offsets=False, **kwargs):
    model_type = EncoderProcessorDecoder
    if n_members is not None:
        model_type = EncoderProcessorDecoderEnsemble
        kwargs["n_members"] = n_members
    return model_type(
        encoder_decoder=EncoderDecoder(
            encoder=IdentityEncoder(in_channels=2),
            decoder=IdentityDecoder(in_channels=2),
        ),
        processor=DecodedOutputProcessor(output, member_offsets=member_offsets),
        **kwargs,
    )


@pytest.mark.parametrize("n_members", [None, 2])
@pytest.mark.parametrize("use_normalization", [False, True])
@pytest.mark.parametrize("use_delta_stats", [False, True])
def test_default_and_explicit_direct_predictions_preserve_checkpoint(
    norm, n_members, use_normalization, use_delta_stats
):
    batch = make_batch()
    decoded = torch.randn_like(batch.output_fields[0])
    normalizer = norm if use_normalization else None
    default = make_model(decoded, n_members=n_members, norm=normalizer)
    direct = make_model(
        decoded,
        n_members=n_members,
        norm=normalizer,
        residual_prediction=False,
        residual_use_delta_stats=use_delta_stats,
        delta_norm=norm,
    )
    direct.load_state_dict(default.state_dict(), strict=True)
    expected = with_members(repeat(decoded, "... -> b ...", b=2), n_members)
    torch.testing.assert_close(default(batch), expected, rtol=0, atol=0)
    torch.testing.assert_close(direct(batch), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        direct.predict_step(batch, 0),
        with_members(
            physical_state(repeat(decoded, "... -> b ...", b=2), use_normalization),
            n_members,
        ),
    )
    assert direct.state_dict().keys() == default.state_dict().keys()


@pytest.mark.parametrize(("t_input", "t_output"), [(1, 1), (3, 4), (6, 2)])
@pytest.mark.parametrize("use_normalization", [False, True])
@pytest.mark.parametrize("use_delta_stats", [False, True])
@pytest.mark.parametrize("independent_delta_stats", [False, True])
def test_residual_units_and_shared_baseline(
    norm, t_input, t_output, use_normalization, use_delta_stats, independent_delta_stats
):
    batch = make_batch(t_input, t_output)
    residual = batch.output_fields[0] - 100
    model = make_model(
        residual,
        norm=norm if use_normalization else None,
        delta_norm=norm if independent_delta_stats else None,
        residual_prediction=True,
        residual_use_delta_stats=use_delta_stats,
    )
    increment = physical_delta(
        residual, use_normalization, use_delta_stats, independent_delta_stats
    )
    if use_normalization:
        increment = increment / STATE_STD
    expected = batch.input_fields[:, -1:] + increment
    prediction = model(batch)
    # Each lead uses the same final observation, without summing earlier leads.
    torch.testing.assert_close(prediction, expected)
    torch.testing.assert_close(
        model.predict_step(batch, 0), physical_state(expected, use_normalization)
    )
    if use_normalization and use_delta_stats:
        # Zero normalized delta still carries the nonzero physical delta mean.
        assert not torch.equal(prediction[:, 0], batch.input_fields[:, -1])


def test_small_residuals_survive_large_state_means_and_rollout():
    norm = ZScoreNormalization(
        stats={
            name: dict.fromkeys(["u", "v"], value)
            for name, value in {
                "mean": 100_000.0,
                "std": 100.0,
                "mean_delta": 0.0,
                "std_delta": 0.1,
            }.items()
        },
        core_field_names=["u", "v"],
        core_constant_field_names=[],
    )
    batch = Batch(torch.ones(1, 3, 1, 1, 2), torch.zeros(1, 100, 1, 1, 2), None, None)
    model = make_model(
        torch.full((1, 1, 1, 2), 0.01), norm=norm, residual_prediction=True
    )
    # Physical increments of 0.001 round away at 100100 in float32, but the
    # normalized increment of 0.00001 is representable and must feed the rollout.
    expected = torch.full((1, 1, 1, 1, 2), 1.00001)
    torch.testing.assert_close(model(batch), expected, rtol=0, atol=1e-7)
    predictions, _ = model.rollout(
        batch, stride=1, max_rollout_steps=100, free_running_only=True
    )
    torch.testing.assert_close(
        predictions[:, -1:], torch.full_like(expected, 100_100.1), rtol=0, atol=0.004
    )
    assert isinstance(model.processor.last_input, torch.Tensor)
    torch.testing.assert_close(
        model.processor.last_input[:, -1:],
        torch.full_like(expected, 1.00099),
        rtol=0,
        atol=2e-6,
    )


def test_residual_reconstruction_with_rms_normalization():
    norm = RMSNormalization(
        stats={"rms": {"u": 2.0, "v": 5.0}, "rms_delta": {"u": 0.25, "v": 2.0}},
        core_field_names=["u", "v"],
        core_constant_field_names=[],
    )
    batch = make_batch()
    residual = torch.ones_like(batch.output_fields[0])
    model = make_model(residual, norm=norm, residual_prediction=True)
    expected = (batch.input_fields[:, -1:] + DELTA_STD / STATE_STD).expand_as(
        batch.output_fields
    )
    torch.testing.assert_close(model(batch), expected)
    torch.testing.assert_close(model.predict_step(batch, 0), expected * STATE_STD)


@pytest.mark.parametrize("n_members", [2, 3])
@pytest.mark.parametrize("use_normalization", [False, True])
@pytest.mark.parametrize("use_delta_stats", [False, True])
@pytest.mark.parametrize("independent_delta_stats", [False, True])
def test_ensemble_reconstruction_and_physical_predictions(
    norm, n_members, use_normalization, use_delta_stats, independent_delta_stats
):
    batch = make_batch()
    residual = torch.ones_like(batch.output_fields[0])
    model = make_model(
        residual,
        n_members=n_members,
        member_offsets=True,
        norm=norm if use_normalization else None,
        delta_norm=norm if independent_delta_stats else None,
        residual_prediction=True,
        residual_use_delta_stats=use_delta_stats,
    )
    # Calculate channels-last, then move members last; equal C and M is ambiguous.
    offsets = rearrange(torch.arange(2 * n_members), "(b m) -> b m 1 1 1 1", b=2)
    physical = physical_state(
        batch.input_fields[:, None, -1:], use_normalization
    ) + physical_delta(
        residual + offsets, use_normalization, use_delta_stats, independent_delta_stats
    )
    normalized = (physical - STATE_MEAN) / STATE_STD if use_normalization else physical
    torch.testing.assert_close(
        model(batch), rearrange(normalized, "b m ... -> b ... m")
    )
    torch.testing.assert_close(
        model.predict_step(batch, 0), rearrange(physical, "b m ... -> b ... m")
    )


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("n_members", [None, 2])
def test_clean_baseline_and_inputs_survive_noise_and_rollout(in_place, n_members):
    class AddNoise(nn.Module):
        def forward(self, x):
            return x.add_(50) if in_place else x + 50

    batch = make_batch()
    clean = batch.input_fields.clone()
    model = make_model(
        torch.zeros_like(batch.output_fields[0]),
        n_members=n_members,
        residual_prediction=True,
        input_noise_injector=AddNoise(),
    )
    expected = with_members(clean[:, -1:].expand_as(batch.output_fields), n_members)
    expected_noisy = (clean + 50).repeat_interleave(n_members or 1, dim=0)
    torch.testing.assert_close(model(batch), expected)
    torch.testing.assert_close(model.processor.last_input, expected_noisy)
    torch.testing.assert_close(batch.input_fields, clean)
    predictions, _ = model.rollout(
        batch,
        stride=4,
        max_rollout_steps=3,
        free_running_only=True,
        n_members=n_members,
    )
    # The rollout's shape probe and each forward must retain the clean baseline.
    torch.testing.assert_close(predictions, torch.cat([expected] * 3, dim=1))
    torch.testing.assert_close(batch.input_fields, clean)


@pytest.mark.parametrize("n_members", [None, 2])
def test_concatenated_input_noise_is_not_part_of_baseline(norm, n_members):
    batch = make_batch()
    model = make_model(
        torch.zeros_like(batch.output_fields[0]),
        n_members=n_members,
        norm=norm,
        residual_prediction=True,
        input_noise_injector=ConcatenatedNoiseInjector(n_channels=3),
    )
    expected = (batch.input_fields[:, -1:] + DELTA_MEAN / STATE_STD).expand_as(
        batch.output_fields
    )
    torch.testing.assert_close(model(batch), with_members(expected, n_members))
    assert isinstance(model.processor.last_input, torch.Tensor)
    assert model.processor.last_input.shape[-1] == 5


@pytest.mark.parametrize("use_normalization", [False, True])
def test_gradients_reach_decoded_residual_and_last_observation(norm, use_normalization):
    batch = make_batch()
    batch.input_fields.requires_grad_()
    model = make_model(
        torch.ones_like(batch.output_fields[0]),
        norm=norm if use_normalization else None,
        delta_norm=norm,
        residual_prediction=True,
    )
    model(batch).sum().backward()
    assert isinstance(model.processor, DecodedOutputProcessor)
    residual_grad = torch.ones_like(model.processor.output) * 2 * DELTA_STD
    if use_normalization:
        residual_grad = residual_grad / STATE_STD
    torch.testing.assert_close(model.processor.output.grad, residual_grad)
    expected_input_grad = torch.zeros_like(batch.input_fields)
    expected_input_grad[:, -1] = 4
    torch.testing.assert_close(batch.input_fields.grad, expected_input_grad)


def test_residuals_with_channels_first_latents_and_fno(norm):
    batch = make_batch()
    encoder_decoder = EncoderDecoder(
        encoder=PermuteConcat(in_channels=2, n_steps_input=3),
        decoder=ChannelsLast(output_channels=2, time_steps=4),
    )
    processor = FNOProcessor(
        in_channels=6, out_channels=8, n_modes=(2, 2), hidden_channels=4, n_layers=1
    )
    model = EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        norm=norm,
        loss_func=nn.MSELoss(),
        residual_prediction=True,
    )
    encoded = encoder_decoder.encoder.encode(batch)
    assert encoded.shape == (2, 6, 2, 3)
    residual = encoder_decoder.decoder.decode(processor.map(encoded, None))
    expected = batch.input_fields[:, -1:] + physical_delta(residual) / STATE_STD
    torch.testing.assert_close(model(batch), expected)
    loss, _ = model.loss(batch)
    loss.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in processor.parameters()
    )


@pytest.mark.parametrize("spatial_shape", [(5,), (2, 3, 4)])
def test_matching_one_and_three_dimensional_grids(norm, spatial_shape):
    inputs = torch.ones(2, 3, *spatial_shape, 2)
    outputs = torch.zeros(2, 4, *spatial_shape, 2)
    model = make_model(outputs[0], norm=norm, residual_prediction=True)
    expected = torch.ones_like(outputs) + DELTA_MEAN / STATE_STD
    torch.testing.assert_close(model(Batch(inputs, outputs, None, None)), expected)


@pytest.mark.parametrize("stage", ["training", "validation", "test"])
@pytest.mark.parametrize("n_members", [None, 2])
def test_losses_and_metrics_receive_complete_fields(
    norm, monkeypatch, stage, n_members
):
    batch = make_batch()
    loss = CaptureLoss()
    model = make_model(
        torch.zeros_like(batch.output_fields[0]),
        n_members=n_members,
        norm=norm,
        residual_prediction=True,
        loss_func=loss,
        train_metrics=[MSE()],
    )
    capture_metrics = Mock()
    monkeypatch.setattr(model, "_update_and_log_metrics", capture_metrics)
    monkeypatch.setattr(model, "log", Mock())
    getattr(model, f"{stage}_step")(batch, 0)
    expected = (batch.input_fields[:, -1:] + DELTA_MEAN / STATE_STD).expand_as(
        batch.output_fields
    )
    target, expected_metric = batch.output_fields, expected
    if stage != "training":
        target, expected_metric = physical_state(target), physical_state(expected)
    torch.testing.assert_close(loss.prediction, with_members(expected, n_members))
    torch.testing.assert_close(
        capture_metrics.call_args.args[2], with_members(expected_metric, n_members)
    )
    torch.testing.assert_close(capture_metrics.call_args.args[3], target)


@pytest.mark.parametrize("teacher_forced", [False, True])
@pytest.mark.parametrize("use_normalization", [False, True])
@pytest.mark.parametrize("use_delta_stats", [False, True])
@pytest.mark.parametrize("independent_delta_stats", [False, True])
@pytest.mark.parametrize(
    ("n_members", "stride", "return_windows"),
    [(None, 4, False), (2, 4, False), (3, 4, False), (None, 2, True)],
)
def test_rollouts_advance_with_complete_normalized_states(
    norm,
    teacher_forced,
    n_members,
    stride,
    return_windows,
    use_normalization,
    use_delta_stats,
    independent_delta_stats,
):
    batch = make_batch(t_input=6 if return_windows else 3, t_output=3 * stride)
    original_inputs = batch.input_fields.clone()
    residual = batch.output_fields[0, :4] - 99
    model = make_model(
        residual,
        n_members=n_members,
        member_offsets=n_members is not None,
        norm=norm if use_normalization else None,
        delta_norm=norm if independent_delta_stats else None,
        residual_prediction=True,
        residual_use_delta_stats=use_delta_stats,
    )
    predictions, targets = model.rollout(
        batch,
        stride=stride,
        max_rollout_steps=3,
        teacher_forcing_ratio=float(teacher_forced),
        n_members=n_members,
        return_windows=return_windows,
    )
    baseline = with_members(
        physical_state(original_inputs[:, -1:], use_normalization), n_members
    )
    if n_members is not None:
        offsets = rearrange(torch.arange(2 * n_members), "(b m) -> b m 1 1 1 1", b=2)
        increments = rearrange(
            physical_delta(
                residual + offsets,
                use_normalization,
                use_delta_stats,
                independent_delta_stats,
            ),
            "b m ... -> b ... m",
        )
    else:
        increments = physical_delta(
            residual, use_normalization, use_delta_stats, independent_delta_stats
        )[None]
    expected_windows = []
    for window in range(3):
        if teacher_forced and window > 0:
            truth = batch.output_fields[:, window * stride - 1 : window * stride]
            baseline = with_members(physical_state(truth, use_normalization), n_members)
        expected_windows.append(baseline + increments)
        if not teacher_forced:
            # The last fed lead, not necessarily lead four, advances the baseline.
            baseline = baseline + increments[:, stride - 1 : stride]
    expected = (
        torch.stack(expected_windows, dim=1)
        if return_windows
        else torch.cat(expected_windows, dim=1)
    )
    physical_targets = physical_state(batch.output_fields, use_normalization)
    if return_windows:
        physical_targets = rearrange(
            physical_targets, "b (r t) ... -> b r t ...", t=stride
        )
    torch.testing.assert_close(predictions, expected)
    torch.testing.assert_close(targets, physical_targets)
    torch.testing.assert_close(batch.input_fields, original_inputs)


@pytest.mark.parametrize("detach", [False, True])
def test_free_running_rollout_respects_gradient_detachment(norm, detach):
    batch = make_batch(t_input=3, t_output=3)
    batch.input_fields.requires_grad_()
    model = make_model(
        torch.ones_like(batch.output_fields[0, :1]), norm=norm, residual_prediction=True
    )
    predictions, _ = model.rollout(
        batch,
        stride=1,
        max_rollout_steps=3,
        teacher_forcing_ratio=1.0,
        free_running_only=True,
        detach=detach,
    )
    # free_running_only overrides teacher forcing and recurrent outputs are fields.
    baseline = physical_state(batch.input_fields[:, -1:])
    expected = torch.cat(
        [baseline + lead * physical_delta(1) for lead in (1, 2, 3)], dim=1
    )
    torch.testing.assert_close(predictions, expected)
    predictions.sum().backward()
    assert isinstance(model.processor, DecodedOutputProcessor)
    multiplicity = 3 if detach else 6
    expected_grad = (
        torch.ones_like(model.processor.output) * 2 * multiplicity * DELTA_STD
    )
    torch.testing.assert_close(model.processor.output.grad, expected_grad)


@pytest.mark.parametrize("shape", [(4, 2, 3, 3), (4, 3, 3, 2), (4, 6, 2)])
def test_incompatible_decoded_field_or_grid_layout_fails(shape):
    model = make_model(torch.zeros(shape), residual_prediction=True)
    with pytest.raises(ValueError, match=r"(?i)(shape|layout|channel|spatial|grid)"):
        model(make_batch())


def test_latent_training_rejects_residual_reconstruction():
    with pytest.raises(ValueError, match=r"(?i)latent"):
        make_model(
            torch.zeros(4, 2, 3, 2),
            residual_prediction=True,
            train_in_latent_space=True,
        )
