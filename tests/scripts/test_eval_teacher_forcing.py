"""Teacher-forcing evaluation validation and window-boundary regression tests."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from autocast.processors.base import Processor
from autocast.scripts.eval.encoder_processor_decoder import (
    _build_encode_once_rollout_predict,
    _render_rollouts,
    _resolve_teacher_forcing_ratio,
)
from autocast.types import Batch, EncodedBatch


@pytest.mark.parametrize("settings", [{}, {"free_running_only": False}])
def test_teacher_forcing_defaults_preserve_free_running(settings):
    cfg = OmegaConf.create(settings)
    assert isinstance(cfg, DictConfig)
    assert _resolve_teacher_forcing_ratio(cfg) == 0.0


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_teacher_forcing_accepts_explicit_valid_ratio(ratio):
    settings = {"free_running_only": False, "teacher_forcing_ratio": ratio}
    assert _resolve_teacher_forcing_ratio(OmegaConf.create(settings)) == ratio


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_teacher_forcing_rejects_invalid_ratio(ratio):
    settings = {"free_running_only": False, "teacher_forcing_ratio": ratio}
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        _resolve_teacher_forcing_ratio(OmegaConf.create(settings))


def test_teacher_forcing_rejects_conflicting_free_running_mode():
    settings = {"free_running_only": True, "teacher_forcing_ratio": 1.0}
    with pytest.raises(ValueError, match="free_running_only=false"):
        _resolve_teacher_forcing_ratio(OmegaConf.create(settings))


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_encode_once_teacher_forcing_advances_at_four_frame_boundaries(ratio):
    class IdentityEncoder:
        def encode_batch(self, batch):
            return EncodedBatch(
                encoded_inputs=batch.input_fields,
                encoded_output_fields=batch.output_fields,
                global_cond=None,
                encoded_info={},
            )

    class IncrementProcessor(Processor):
        def map(self, x, global_cond=None):  # noqa: ARG002
            increments = rearrange(torch.arange(1.0, 5.0), "t -> 1 t 1 1 1")
            return x[:, -1:] + increments

        def loss(self, batch):  # noqa: ARG002
            return torch.zeros(())

    batch = Batch(
        input_fields=torch.ones(1, 1, 1, 1, 1),
        output_fields=rearrange(torch.arange(10.0, 90.0, 10.0), "t -> 1 t 1 1 1"),
        constant_scalars=None,
        constant_fields=None,
    )
    model = SimpleNamespace(
        encoder_decoder=SimpleNamespace(
            encoder=IdentityEncoder(), decoder=SimpleNamespace(decode=lambda x: x)
        ),
        processor=IncrementProcessor(),
        denormalize_tensor=lambda x: x,
    )
    predict = _build_encode_once_rollout_predict(
        model,
        rollout_stride=4,
        max_rollout_steps=2,
        free_running_only=False,
        n_members=None,
        device="cpu",
        teacher_forcing_ratio=ratio,
    )
    predictions, truth = predict(batch)
    assert predictions.flatten().tolist()[:4] == [2.0, 3.0, 4.0, 5.0]
    expected_second = [41.0, 42.0, 43.0, 44.0] if ratio else [6.0, 7.0, 8.0, 9.0]
    assert predictions.flatten().tolist()[4:] == expected_second
    assert truth is not None
    assert torch.equal(truth, batch.output_fields)


def test_rollout_rendering_forwards_teacher_forcing(tmp_path, monkeypatch):
    captured: dict[str, Any] = {}

    class DummyModel:
        def rollout(self, *_args, **kwargs):
            captured.update(kwargs)
            prediction = torch.zeros(1, 2, 2, 2, 1)
            return prediction, torch.ones_like(prediction)

    monkeypatch.setattr(
        "autocast.scripts.eval.encoder_processor_decoder.plot_spatiotemporal_video",
        lambda **_kwargs: None,
    )
    _render_rollouts(
        model=cast(Any, DummyModel()),
        dataloader=[object()],
        batch_indices=[0],
        video_dir=tmp_path,
        sample_index=0,
        fmt="mp4",
        fps=5,
        stride=1,
        max_rollout_steps=2,
        free_running_only=False,
        teacher_forcing_ratio=1.0,
    )
    assert captured["teacher_forcing_ratio"] == 1.0
    assert captured["free_running_only"] is False
