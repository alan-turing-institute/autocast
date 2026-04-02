"""Tests for the cache_latents script."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autocast.data.encoded_dataset import CachedLatentDataset
from autocast.encoders.base import EncoderWithCond
from autocast.scripts.cache_latents import _encode_and_save_split
from autocast.types.batch import Batch
from autocast.types.types import TensorBNC


class _MockEncoder(EncoderWithCond):
    """Minimal encoder for testing that maps input channels to latent channels."""

    latent_channels = 8
    channel_axis = -1
    encoder_model = nn.Identity()

    def __init__(self, in_channels: int = 2, latent_channels: int = 8):
        super().__init__()
        self._latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.proj = nn.Linear(in_channels, latent_channels)

    def encode(self, batch: Batch) -> TensorBNC:
        return self.proj(batch.input_fields)


def _make_full_traj_dataloader(
    batch_size=2, n_trajectories=3, t_in=1, t_out=9, w=8, h=8, c=2
):
    """Create a dataloader simulating full_trajectory_mode output.

    Each Batch item has input_fields=(B, t_in, W, H, C) and
    output_fields=(B, t_out, W, H, C), representing full trajectories
    split into 1 input step and T-1 output steps.
    """

    class _Dataset(Dataset):
        def __init__(self):
            self.size = n_trajectories

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return Batch(
                input_fields=torch.randn(t_in, w, h, c),
                output_fields=torch.randn(t_out, w, h, c),
                constant_scalars=None,
                constant_fields=None,
            )

    def _collate(items):
        return Batch(
            input_fields=torch.stack([x.input_fields for x in items]),
            output_fields=torch.stack([x.output_fields for x in items]),
            constant_scalars=None,
            constant_fields=None,
        )

    return DataLoader(
        _Dataset(), batch_size=batch_size, collate_fn=_collate, num_workers=0
    )


def _make_cond_dataloader(cond_dim=3, t_in=1, t_out=9):
    """Create a dataloader with constant_scalars for conditioning tests."""

    class _Dataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return Batch(
                input_fields=torch.randn(t_in, 8, 8, 2),
                output_fields=torch.randn(t_out, 8, 8, 2),
                constant_scalars=torch.randn(cond_dim),
                constant_fields=None,
            )

    def _collate(items):
        return Batch(
            input_fields=torch.stack([x.input_fields for x in items]),
            output_fields=torch.stack([x.output_fields for x in items]),
            constant_scalars=torch.stack([x.constant_scalars for x in items]),
            constant_fields=None,
        )

    return DataLoader(_Dataset(), batch_size=2, collate_fn=_collate, num_workers=0)


# --- Tests ---


def test_encode_and_save_creates_traj_files(tmp_path):
    encoder = _MockEncoder(in_channels=2, latent_channels=8)
    encoder.eval()
    dataloader = _make_full_traj_dataloader(
        batch_size=2, n_trajectories=3, t_in=1, t_out=9
    )

    split_dir = tmp_path / "train"
    metadata = _encode_and_save_split(
        encoder, dataloader, split_dir, torch.device("cpu")
    )

    assert split_dir.exists()
    traj_files = sorted(split_dir.glob("traj_*.pt"))
    assert len(traj_files) == 3
    assert metadata["num_trajectories"] == 3


def test_saved_trajectories_have_full_timesteps(tmp_path):
    t_in, t_out = 1, 9
    latent_channels = 4
    encoder = _MockEncoder(in_channels=2, latent_channels=latent_channels)
    encoder.eval()
    dataloader = _make_full_traj_dataloader(
        batch_size=1, n_trajectories=1, t_in=t_in, t_out=t_out
    )

    split_dir = tmp_path / "test"
    _encode_and_save_split(encoder, dataloader, split_dir, torch.device("cpu"))

    traj = torch.load(split_dir / "traj_000000.pt", weights_only=False)
    # Full trajectory: t_in + t_out timesteps
    assert traj["encoded_fields"].shape[0] == t_in + t_out
    assert traj["encoded_fields"].shape[-1] == latent_channels


def test_round_trip_with_cached_dataset(tmp_path):
    """Encode full trajectories, then load with CachedLatentDataset and window."""
    encoder = _MockEncoder(in_channels=2, latent_channels=4)
    encoder.eval()
    dataloader = _make_full_traj_dataloader(
        batch_size=2, n_trajectories=2, t_in=1, t_out=9, c=2
    )

    split_dir = tmp_path / "train"
    _encode_and_save_split(encoder, dataloader, split_dir, torch.device("cpu"))

    # Load with different windowing
    dataset = CachedLatentDataset(
        cache_dir=split_dir, n_steps_input=2, n_steps_output=3, stride=2
    )
    # T=10, window=5, stride=2: (10-5)//2 + 1 = 3 windows per traj, 2 trajs
    assert len(dataset) == 6

    sample = dataset[0]
    assert sample.encoded_inputs.shape[0] == 2  # n_steps_input
    assert sample.encoded_output_fields.shape[0] == 3  # n_steps_output
    assert sample.encoded_inputs.shape[-1] == 4  # latent channels


def test_global_cond_saved_and_loaded(tmp_path):
    """Verify that global_cond is saved and loaded correctly."""
    cond_dim = 3
    encoder = _MockEncoder(in_channels=2, latent_channels=4)
    encoder.eval()

    dataloader = _make_cond_dataloader(cond_dim=cond_dim)

    split_dir = tmp_path / "train"
    _encode_and_save_split(encoder, dataloader, split_dir, torch.device("cpu"))

    dataset = CachedLatentDataset(
        cache_dir=split_dir, n_steps_input=2, n_steps_output=2
    )
    sample = dataset[0]
    assert sample.global_cond is not None
    assert sample.global_cond.shape == (cond_dim,)
