"""Train a ProcessorModel directly on encoded datasets."""

import logging

import lightning as L

from autocast.scripts.cli import parse_common_args
from autocast.scripts.config import load_config
from autocast.scripts.setup import (
    run_training,
    setup_encoded_datamodule,
    setup_processor_model,
)

log = logging.getLogger(__name__)


def main():
    """CLI entrypoint for training the processor."""
    # Parse args
    args = parse_common_args(
        description=(
            "Train a ProcessorModel directly on encoded datasets with "
            "Hydra-configured processor components."
        ),
        default_config_name="processor",
    )
    logging.basicConfig(level=logging.INFO)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Compose config
    cfg = load_config(args)

    # Setup datamodule and resolve config
    datamodule, cfg, stats = setup_encoded_datamodule(cfg)

    # Seed
    L.seed_everything(cfg.get("seed", 42), workers=True)

    # Setup Model
    model = setup_processor_model(cfg, stats)

    # Run Training
    run_training(
        cfg,
        model,
        datamodule,
        work_dir,
        skip_test=args.skip_test,
        output_checkpoint_path=args.output_checkpoint,
        job_type="train-processor",
    )


if __name__ == "__main__":
    main()
