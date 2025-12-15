import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autocast.train.configuration import compose_training_config
from autocast.train.encoder_processor_decoder import main


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    workdir_path = tmp_path / "workdir"
    workdir_path.mkdir()
    return workdir_path


@pytest.fixture
def epd_cli_args(REPO_ROOT: Path, workdir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config_dir=REPO_ROOT / "configs",
        config_name="encoder_processor_decoder",
        overrides=[],
        work_dir=workdir,
        autoencoder_checkpoint=None,
        freeze_autoencoder=None,
        n_steps_input=None,
        n_steps_output=None,
        stride=None,
        output_checkpoint=None,
        skip_test=True,
    )


def test_train_encoder_processor_decoder(
    epd_cli_args: argparse.Namespace,
) -> None:
    # Config prep for training
    cfg = compose_training_config(epd_cli_args)

    # Mock wandb logger and related components
    mock_wandb_logger = MagicMock()
    mock_watch_cfg = MagicMock()

    # change epochs to one
    cfg.trainer.max_epochs = 1

    with (
        patch(
            "autocast.train.encoder_processor_decoder.create_wandb_logger",
            return_value=(mock_wandb_logger, mock_watch_cfg),
        ) as mock_create_logger,
        patch("autocast.train.encoder_processor_decoder.maybe_watch_model"),
        patch(
            "autocast.train.encoder_processor_decoder.parse_args",
            return_value=epd_cli_args,
        ),
    ):
        main()

        # Verify wandb logger was created with correct parameters
        mock_create_logger.assert_called_once()
        assert mock_create_logger.call_args.args[0] == cfg.get("logging")

        # Verify checkpoint file was created
        checkpoint_files = list(epd_cli_args.work_dir.glob("**/*.ckpt"))
        assert len(checkpoint_files) > 0, "No checkpoint files found"

        # Check that the default checkpoint exists
        default_checkpoint = epd_cli_args.work_dir / "encoder_processor_decoder.ckpt"
        assert default_checkpoint.exists(), (
            f"Expected checkpoint at {default_checkpoint}"
        )
        assert default_checkpoint.suffix == ".ckpt"
