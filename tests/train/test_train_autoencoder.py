import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autocast.train.autoencoder import (
    _configure_trainer_root,
    _resolve_work_dir,
    compose_training_config,
    train_autoencoder,
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    workdir_path = tmp_path / "workdir"
    workdir_path.mkdir()
    return workdir_path


@pytest.fixture
def autoencoder_cli_args(REPO_ROOT: Path, workdir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config_dir=REPO_ROOT / "configs",
        config_name="autoencoder",
        overrides=[],
        work_dir=workdir,
    )


def test_train_autoencoder(
    autoencoder_cli_args: argparse.Namespace,
) -> None:
    # Config prep for training
    cfg = compose_training_config(autoencoder_cli_args)
    resolved_work_dir = _resolve_work_dir(autoencoder_cli_args, cfg)
    _configure_trainer_root(cfg, resolved_work_dir)

    # Mock wandb logger and related components
    mock_wandb_logger = MagicMock()
    mock_watch_cfg = MagicMock()

    # change epocchs to one
    cfg.trainer.max_epochs = 1

    with (
        patch(
            "autocast.train.autoencoder.create_wandb_logger",
            return_value=(mock_wandb_logger, mock_watch_cfg),
        ) as mock_create_logger,
        patch("autocast.train.autoencoder.maybe_watch_model"),
    ):
        checkpoint_path = train_autoencoder(cfg, work_dir=resolved_work_dir)

        # Verify wandb logger was created with correct parameters
        mock_create_logger.assert_called_once()
        assert mock_create_logger.call_args.args[0] == cfg.get("logging")

        # Verify checkpoint path is returned and exists
        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()

        # Verify that working dir has a checkpoints folder, checkpoint file
        # and a yaml file
        checkpoints_dir = resolved_work_dir / "checkpoints"
        assert checkpoints_dir.exists()

        checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
        assert len(checkpoint_files) > 0, "No checkpoint files found"

        yaml_files = list(resolved_work_dir.glob("*.yaml"))
        assert len(yaml_files) > 0, "No yaml config files found"

        # Assert reconstructions folder exists
        reconstructions_dir = resolved_work_dir / "reconstructions"
        assert reconstructions_dir.exists()

        # Check there are 4 reconstruction images saved
        reconstruction_images = list(reconstructions_dir.glob("*.png"))
        assert len(reconstruction_images) == 4, (
            f"Expected 4 reconstruction images, found {len(reconstruction_images)}"
        )
