"""Train an encoder-processor-decoder with optional pretrained autoencoder."""

import logging
import os
from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig

from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.training import run_training
from autocast.scripts.utils import get_default_config_path

log = logging.getLogger(__name__)


def _apply_umask(cfg: DictConfig) -> None:
    umask_value = cfg.get("umask")
    if umask_value is not None:
        os.umask(int(str(umask_value), 8))
        log.info("Applied process umask %s", umask_value)


def run_epd_training(
    cfg: DictConfig,
    *,
    work_dir: Path | None = None,
    job_type: str = "train-encoder-processor-decoder",
) -> DictConfig:
    """Run EPD training from a composed config and return resolved config."""
    _apply_umask(cfg)
    work_dir = work_dir or Path.cwd()

    datamodule, cfg, stats = setup_datamodule(cfg)
    L.seed_everything(cfg.get("seed", 42), workers=True)
    model = setup_epd_model(cfg, stats, datamodule=datamodule)

    output_cfg = cfg.get("output", {})
    skip_test = output_cfg.get("skip_test", False)
    output_checkpoint = output_cfg.get("checkpoint_path")

    run_training(
        cfg,
        model,
        datamodule,
        work_dir,
        skip_test=skip_test,
        output_checkpoint_path=output_checkpoint,
        job_type=job_type,
    )

    return cfg


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:
    """CLI entrypoint for training the encoder-processor-decoder."""
    logging.basicConfig(level=logging.INFO)
    run_epd_training(cfg, work_dir=Path.cwd())


if __name__ == "__main__":
    main()
