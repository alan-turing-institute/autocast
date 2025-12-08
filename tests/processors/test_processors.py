import torch
from torch import nn

from auto_cast.processors.base import Processor
from auto_cast.types import EncodedBatch, Tensor


def _toy_encoded_batch(
    batch_size: int = 2,
    t_in: int = 2,
    t_out: int | None = None,
    w: int = 4,
    h: int = 4,
    c: int = 1,
) -> EncodedBatch:
    t_out = t_out or t_in
    encoded_inputs = torch.randn(batch_size, t_in, w, h, c)
    encoded_outputs = torch.randn(batch_size, t_out, w, h, c)
    return EncodedBatch(
        encoded_inputs=encoded_inputs,
        encoded_output_fields=encoded_outputs,
        encoded_info={},
    )


class _IdentityProcessor(Processor):
    def __init__(self, *, stride: int, max_rollout_steps: int) -> None:
        super().__init__(
            stride=stride,
            max_rollout_steps=max_rollout_steps,
            teacher_forcing_ratio=0.0,
            loss_func=nn.MSELoss(),
        )

    def map(self, x: Tensor) -> Tensor:
        return x


def test_processor_rollout_handles_encoded_batches():
    batch_size = 10
    n_steps_input = 2
    n_steps_output = n_steps_input
    stride = 2
    max_rollout_steps = 4
    trajectory_length = 100
    processor = _IdentityProcessor(stride=stride, max_rollout_steps=max_rollout_steps)
    encoded_batch = _toy_encoded_batch(
        batch_size=batch_size,
        t_in=n_steps_input,
        t_out=trajectory_length - n_steps_input,
    )

    preds, gts = processor.rollout(encoded_batch, return_windows=True)

    assert preds.shape == (10, max_rollout_steps, n_steps_output, 4, 4, 1)
    assert gts is not None
    assert gts.shape == preds.shape

    preds, gts = processor.rollout(encoded_batch, return_windows=False)

    assert preds.shape == (10, max_rollout_steps * n_steps_output, 4, 4, 1)
    assert gts is not None
    assert gts.shape == preds.shape


def test_processor_rollout_handles_short_trajectory():
    """Test rollout when trajectory is shorter than stride * max_rollout_steps.

    In free-running mode the model continues predicting using its own outputs even after
    ground truth data runs out.
    """
    batch_size = 10
    n_steps_input = 2
    n_steps_output = n_steps_input
    stride = 2
    max_rollout_steps = 10

    # Short trajectory: only 6 time steps available for output
    trajectory_length = n_steps_input + 6
    processor = _IdentityProcessor(stride=stride, max_rollout_steps=max_rollout_steps)
    encoded_batch = _toy_encoded_batch(
        batch_size=batch_size,
        t_in=n_steps_input,
        t_out=trajectory_length - n_steps_input,
    )

    preds, gts = processor.rollout(encoded_batch, return_windows=True)

    # In free-running mode, predictions continue for all max_rollout_steps
    assert preds.shape == (batch_size, max_rollout_steps, n_steps_output, 4, 4, 1)

    # Ground truth only available for 3 windows:
    # Window 1: timesteps 0-1 (from initial output_fields[0:2])
    # Window 2: timesteps 2-3 (after advancing by stride=2)
    # Window 3: timesteps 4-5 (after advancing by stride=2 again)
    # After that, output_fields is exhausted
    expected_gt_windows = 3
    assert gts is not None
    assert gts.shape == (batch_size, expected_gt_windows, n_steps_output, 4, 4, 1)

    preds, gts = processor.rollout(encoded_batch, return_windows=False)

    # Predictions for all rollout windows concatenated
    assert preds.shape == (batch_size, max_rollout_steps * n_steps_output, 4, 4, 1)
    # Ground truth only for windows where data was available
    assert gts is not None
    assert gts.shape == (batch_size, expected_gt_windows * n_steps_output, 4, 4, 1)
