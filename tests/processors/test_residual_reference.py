import pytest
import torch
from einops import repeat
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from autocast.processors.base import Processor
from autocast.processors.residual_reference import (
    LastFrameReference,
    ProcessorReference,
)
from autocast.types import EncodedBatch, Tensor


class _RecordingProcessor(Processor[EncodedBatch]):
    def __init__(self, n_steps_output: int = 3) -> None:
        super().__init__()
        self.n_steps_output = n_steps_output
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.received_global_cond: Tensor | None = None

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        self.received_global_cond = global_cond
        return repeat(
            self.scale * x[:, -1:],
            "b 1 ... c -> b t ... c",
            t=self.n_steps_output,
        )

    def loss(self, batch: EncodedBatch) -> Tensor:
        raise NotImplementedError


def test_last_frame_reference_repeats_selected_channels():
    x = torch.arange(2 * 3 * 4 * 5 * 2).reshape(2, 3, 4, 5, 2)
    reference = LastFrameReference(n_steps_output=4, channel_indices=(1,))

    actual = reference(x)

    expected = repeat(x[:, -1:, ..., 1:2], "b 1 ... c -> b t ... c", t=4)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_steps_output": 0}, "must be positive"),
        ({"n_steps_output": 1, "channel_indices": ()}, "cannot be empty"),
    ],
)
def test_last_frame_reference_validates_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LastFrameReference(**kwargs)


def test_frozen_processor_reference_passes_conditioning_without_gradients():
    processor = _RecordingProcessor()
    reference = ProcessorReference(processor)
    reference.train()
    x = torch.randn(2, 1, 4, 4, 1, requires_grad=True)
    global_cond = torch.randn(2, 2)

    output = reference(x, global_cond)

    assert processor.received_global_cond is global_cond
    assert not processor.training
    assert not processor.scale.requires_grad
    assert not output.requires_grad


def test_trainable_processor_reference_preserves_gradients():
    processor = _RecordingProcessor()
    reference = ProcessorReference(processor, frozen=False)
    x = torch.randn(2, 1, 4, 4, 1, requires_grad=True)

    reference(x).sum().backward()

    assert processor.scale.grad is not None
    assert x.grad is not None


def test_pretrained_reference_loads_resolved_config_and_processor_weights(tmp_path):
    processor = _RecordingProcessor()
    with torch.no_grad():
        processor.scale.fill_(7.0)
    config_path = tmp_path / "resolved_config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {
                    "processor": {
                        "_target_": f"{__name__}._RecordingProcessor",
                        "n_steps_output": 3,
                    }
                }
            }
        ),
        config_path,
    )
    checkpoint_path = tmp_path / "processor.ckpt"
    torch.save(
        {
            "state_dict": {
                f"processor.{key}": value
                for key, value in processor.state_dict().items()
            }
        },
        checkpoint_path,
    )

    loaded = instantiate(OmegaConf.load(config_path).model.processor)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    loaded.load_state_dict(
        {
            key.removeprefix("processor."): value
            for key, value in state.items()
            if key.startswith("processor.")
        },
        strict=True,
    )
    reference = ProcessorReference(loaded)
    torch.testing.assert_close(
        reference(torch.ones(1, 1, 2, 2, 1)), torch.full((1, 3, 2, 2, 1), 7.0)
    )
    assert not loaded.scale.requires_grad
