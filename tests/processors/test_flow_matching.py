import itertools

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from auto_cast.nn.unet import TemporalUNetBackbone
from auto_cast.processors.flow_matching import FlowMatchingProcessor
from auto_cast.types import EncodedBatch


def _single_item_collate(items):
    return items[0]


class _FlowMatchingEncodedDataset(Dataset):
    """Minimal dataset that generates flow-matching-friendly `EncodedBatch` samples."""

    def __init__(
        self,
        *,
        n_steps_input: int,
        n_steps_output: int,
        n_channels_in: int,
        n_channels_out: int,
        spatial_size: int = 8,
    ) -> None:
        super().__init__()
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        return 2

    def __getitem__(self, _: int) -> EncodedBatch:
        encoded_inputs = torch.randn(
            1,
            self.n_steps_input,
            self.spatial_size,
            self.spatial_size,
            self.n_channels_in,
        )
        encoded_outputs = torch.randn(
            1,
            self.n_steps_output,
            self.spatial_size,
            self.spatial_size,
            self.n_channels_out,
        )
        return EncodedBatch(
            encoded_inputs=encoded_inputs,
            encoded_output_fields=encoded_outputs,
            encoded_info={},
        )


def _build_encoded_loader(
    *,
    n_steps_input: int,
    n_steps_output: int,
    n_channels_in: int,
    n_channels_out: int,
) -> DataLoader:
    dataset = _FlowMatchingEncodedDataset(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=_single_item_collate,
        num_workers=0,
    )


params = list(
    itertools.product(
        [1, 4],  # n_steps_output
        [1, 4],  # n_steps_input
        [1, 2],  # n_channels_in
        [1, 4],  # n_channels_out
    )
)


@pytest.mark.parametrize(
    ("n_steps_output", "n_steps_input", "n_channels_in", "n_channels_out"),
    params,
)
def test_flow_matching_processor(
    n_steps_output: int, n_steps_input: int, n_channels_in: int, n_channels_out: int
):
    encoded_loader = _build_encoded_loader(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    encoded_batch = next(iter(encoded_loader))

    model = FlowMatchingProcessor(
        flow_matching_model=TemporalUNetBackbone(
            in_channels=n_steps_output * n_channels_out,
            out_channels=n_steps_output * n_channels_out,
            cond_channels=n_steps_input * n_channels_in,
            mod_features=256,
            hid_channels=(32, 64, 128),
            hid_blocks=(2, 2, 2),
            spatial=2,
            periodic=False,
        ),
        n_steps_output=n_steps_output,
        n_channels_out=n_channels_out,
        flow_ode_steps=1,
    )

    output = model.map(encoded_batch.encoded_inputs)
    assert output.shape == encoded_batch.encoded_output_fields.shape

    train_loss = model.training_step(encoded_batch, 0)
    assert train_loss.shape == ()
    train_loss.backward()

    L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        enable_model_summary=False,
        accelerator="cpu",
    ).fit(
        model,
        train_dataloaders=encoded_loader,
        val_dataloaders=encoded_loader,
    )
