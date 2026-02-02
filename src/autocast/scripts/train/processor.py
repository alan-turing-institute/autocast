"""Train a ProcessorModel directly on encoded datasets."""

import logging
from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig

from autocast.scripts.setup import setup_datamodule, setup_processor_model
from autocast.scripts.training import run_training

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../../../configs", config_name="processor"
)
def main(cfg: DictConfig) -> None:
    """CLI entrypoint for training the processor."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Work directory is managed by Hydra
    work_dir = Path.cwd()

    # Setup datamodule and resolve config
    datamodule, cfg, stats = setup_datamodule(cfg)

    # Seed
    L.seed_everything(cfg.get("seed", 42), workers=True)

    # Setup Model
    model = setup_processor_model(cfg, stats)

    # Get output config
    output_cfg = cfg.get("output", {})
    skip_test = output_cfg.get("skip_test", False)
    output_checkpoint = output_cfg.get("checkpoint_path")

    # Run Training
    run_training(
        cfg,
        model,
        datamodule,
        work_dir,
        skip_test=skip_test,
        output_checkpoint_path=output_checkpoint,
        job_type="train-processor",
    )


if __name__ == "__main__":
    main()
