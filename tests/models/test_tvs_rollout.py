"""Tests for time-varying-scalar handling across the rollout machinery."""

import torch
from conftest import CondCaptureProcessor, get_optimizer_config

from autocast.decoders.channels_last import ChannelsLast
from autocast.encoders.permute_concat import PermuteConcat
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.types import Batch


def _build_model(n_steps_input, channels, n_tvs_steps=1):
    encoder = PermuteConcat(
        in_channels=channels, n_steps_input=n_steps_input, n_tvs_steps=n_tvs_steps
    )
    decoder = ChannelsLast(output_channels=channels, time_steps=n_steps_input)
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)
    return EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=CondCaptureProcessor(),
        optimizer_config=get_optimizer_config(),
    )


def test_advance_batch_strides_time_varying_scalars():
    batch_size, n_steps_input, w, h, c, c_tvs = 2, 2, 8, 8, 1, 3
    tvs_len = 10
    model = _build_model(n_steps_input, c)
    batch = Batch(
        input_fields=torch.randn(batch_size, n_steps_input, w, h, c),
        output_fields=torch.randn(batch_size, n_steps_input, w, h, c),
        constant_scalars=None,
        constant_fields=None,
        time_varying_scalars=torch.randn(batch_size, tvs_len, c_tvs),
    )
    next_inputs = torch.randn(batch_size, n_steps_input, w, h, c)

    stride = 2
    advanced = model._advance_batch(batch, next_inputs, stride=stride)
    assert advanced.time_varying_scalars is not None
    assert advanced.time_varying_scalars.shape == (batch_size, tvs_len - stride, c_tvs)
    assert torch.equal(
        advanced.time_varying_scalars, batch.time_varying_scalars[:, stride:, :]
    )


def test_advance_batch_empties_tvs_when_consumed():
    batch_size, n_steps_input, w, h, c, c_tvs = 2, 1, 8, 8, 1, 3
    model = _build_model(n_steps_input, c)
    batch = Batch(
        input_fields=torch.randn(batch_size, n_steps_input, w, h, c),
        output_fields=torch.randn(batch_size, n_steps_input, w, h, c),
        constant_scalars=None,
        constant_fields=None,
        time_varying_scalars=torch.randn(batch_size, 2, c_tvs),
    )
    next_inputs = torch.randn(batch_size, n_steps_input, w, h, c)
    advanced = model._advance_batch(batch, next_inputs, stride=3)
    assert advanced.time_varying_scalars is not None
    assert advanced.time_varying_scalars.shape == (batch_size, 0, c_tvs)


def test_forward_conditions_on_current_tvs_step():
    batch_size, n_steps_input, w, h, c, c_tvs = 2, 3, 8, 8, 2, 4
    model = _build_model(n_steps_input, c)
    batch = Batch(
        input_fields=torch.randn(batch_size, n_steps_input, w, h, c),
        output_fields=torch.randn(batch_size, n_steps_input, w, h, c),
        constant_scalars=None,
        constant_fields=None,
        time_varying_scalars=torch.randn(batch_size, n_steps_input, c_tvs),
    )
    _ = model(batch)
    captured = model.processor.last_global_cond
    assert captured is not None
    # n_tvs_steps=1 -> conditioning is the last input step's scalars.
    assert torch.equal(captured, batch.time_varying_scalars[:, n_steps_input - 1, :])
