"""Cache latent representations from a trained autoencoder.

Given a trained autoencoder checkpoint and a datamodule configuration, this
script encodes all splits (train, valid, test) and saves full encoded
trajectories as .pt files that can be loaded by ``CachedLatentDataset``
for fast processor-only training with runtime-configurable windowing.
"""

import json
import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from autocast.scripts.data import batch_to_device
from autocast.scripts.execution import resolve_hydra_work_dir
from autocast.scripts.setup import setup_autoencoder_components, setup_datamodule
from autocast.scripts.utils import get_default_config_path
from autocast.types.batch import Batch

log = logging.getLogger(__name__)


def _resolve_device(device: str) -> torch.device:
    """Resolve the device string to a torch.device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _encode_and_save_split(
    encoder,
    dataloader,
    split_dir: Path,
    device: torch.device,
) -> dict:
    """Encode all batches from a dataloader and save full trajectories to disk.

    The dataloader should be configured with ``full_trajectory_mode=True`` so
    that each batch item contains a complete trajectory.  Each trajectory is
    encoded and saved as a single ``.pt`` file containing a dict with keys
    ``encoded_fields`` (the full encoded trajectory) and optionally
    ``global_cond``.

    Returns
    -------
    dict
        Metadata about the encoded split (num_trajectories, shapes).
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    total_trajectories = 0
    sample_shape = None

    for batch_idx, batch in enumerate(dataloader):
        if not isinstance(batch, Batch):
            log.warning(
                "Skipping batch %s: expected Batch, got %s", batch_idx, type(batch)
            )
            continue

        batch = batch_to_device(batch, device)  # noqa: PLW2901

        with torch.no_grad():
            encoded_batch = encoder.encode_batch(batch)

        # The encoded fields from input + output form the full trajectory
        # Concatenate along time dimension (dim=1) to reconstruct full trajectory
        full_encoded = torch.cat(
            [encoded_batch.encoded_inputs, encoded_batch.encoded_output_fields],
            dim=1,
        )

        batch_size = full_encoded.shape[0]

        # Save each trajectory as a separate file
        for i in range(batch_size):
            traj_data = {
                "encoded_fields": full_encoded[i].cpu(),
            }
            if encoded_batch.global_cond is not None:
                traj_data["global_cond"] = encoded_batch.global_cond[i].cpu()

            traj_idx = total_trajectories + i
            save_path = split_dir / f"traj_{traj_idx:06d}.pt"
            torch.save(traj_data, save_path)

        total_trajectories += batch_size
        if sample_shape is None:
            sample_shape = {
                "encoded_fields": list(full_encoded[0].shape),
            }
            if encoded_batch.global_cond is not None:
                sample_shape["global_cond"] = list(encoded_batch.global_cond[0].shape)

        if (batch_idx + 1) % 10 == 0:
            log.info(
                "  Encoded %s batches (%s trajectories)",
                batch_idx + 1,
                total_trajectories,
            )

    log.info("  Split complete: %s trajectories", total_trajectories)

    return {
        "num_trajectories": total_trajectories,
        "sample_shape": sample_shape,
    }


def cache_latents(
    cfg: DictConfig,
    output_dir: Path,
    *,
    device: str = "auto",
) -> Path:
    """Encode all data splits and cache full latent trajectories to disk.

    The datamodule is configured with ``full_trajectory_mode=True`` so that
    full trajectories are encoded rather than windowed sub-sequences.  This
    allows runtime-configurable windowing (``n_steps_input``,
    ``n_steps_output``, ``stride``) when loading with ``CachedLatentDataset``.

    Parameters
    ----------
    cfg
        Hydra configuration (must include ``datamodule`` and ``model.encoder``
        sections, plus ``autoencoder_checkpoint``).
    output_dir
        Directory to write cached latent files into.
    device
        Device to run encoding on (``"auto"``, ``"cpu"``, ``"cuda"``, etc.).

    Returns
    -------
    Path
        The output directory containing cached latents.
    """
    resolved_device = _resolve_device(device)
    log.info("Using device: %s", resolved_device)

    # Force full_trajectory_mode so we cache complete trajectories
    with OmegaConf.read_write(cfg):  # type: ignore[attr-defined]
        if "datamodule" in cfg:
            OmegaConf.update(cfg, "datamodule.full_trajectory_mode", True, merge=True)
            # Ensure n_steps_input is set to 1 for full_trajectory_mode
            # (the dataset will set n_steps_output = T - n_steps_input)
            OmegaConf.update(cfg, "datamodule.n_steps_input", 1, merge=True)
            # Disable shuffling for deterministic caching
            OmegaConf.update(cfg, "datamodule.shuffle_train", False, merge=True)

    # Setup datamodule
    datamodule, cfg, stats = setup_datamodule(cfg)

    # Build encoder from config + checkpoint
    encoder, _decoder = setup_autoencoder_components(cfg, stats)
    encoder = encoder.to(resolved_device)
    encoder.eval()

    log.info("Encoder: %s", type(encoder).__name__)
    log.info("Latent channels: %s", getattr(encoder, "latent_channels", "unknown"))

    # Create output directory
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "encoder_class": type(encoder).__name__,
        "latent_channels": getattr(encoder, "latent_channels", None),
        "splits": {},
    }

    # Encode train split
    datamodule.setup(stage="fit")

    log.info("Encoding train split...")
    train_loader = datamodule.train_dataloader()
    metadata["splits"]["train"] = _encode_and_save_split(
        encoder, train_loader, output_dir / "train", resolved_device
    )

    log.info("Encoding valid split...")
    val_loader = datamodule.val_dataloader()
    metadata["splits"]["valid"] = _encode_and_save_split(
        encoder, val_loader, output_dir / "valid", resolved_device
    )

    # Encode test split
    datamodule.setup(stage="test")
    log.info("Encoding test split...")
    test_loader = datamodule.test_dataloader()
    metadata["splits"]["test"] = _encode_and_save_split(
        encoder, test_loader, output_dir / "test", resolved_device
    )

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved to %s", metadata_path)

    log.info("Caching complete. Output directory: %s", output_dir)
    return output_dir


def _apply_umask(cfg: DictConfig) -> None:
    umask_value = cfg.get("umask")
    if umask_value is not None:
        os.umask(int(str(umask_value), 8))
        log.info("Applied process umask %s", umask_value)


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:
    """CLI entrypoint for caching latent representations."""
    logging.basicConfig(level=logging.INFO)
    _apply_umask(cfg)

    work_dir = resolve_hydra_work_dir(None)

    # Use output_dir from config if provided, otherwise default to work_dir/cached
    cache_cfg = cfg.get("cache_latents", {})
    output_dir_str = cache_cfg.get("output_dir")
    if output_dir_str is not None:
        output_dir = Path(output_dir_str).expanduser().resolve()
    else:
        output_dir = work_dir / "cached"

    device = cache_cfg.get("device", "auto")

    cache_latents(cfg, output_dir, device=device)


if __name__ == "__main__":
    main()
