"""Train an encoder-processor-decoder with optional autoencoder warm-start."""

from __future__ import annotations

import logging

import lightning as L

from autocast.scripts.train.configuration import (
    load_config,
    parse_common_args,
)
from autocast.scripts.train.setup import (
    run_training,
    setup_datamodule,
    setup_epd_model,
)

log = logging.getLogger(__name__)


def main():
    """CLI entrypoint for training the processor."""
    # Parse args
    args = parse_common_args(
        description=(
            "Train an encoder-processor-decoder model with Hydra-configured "
            "encoder, decoder, and processor components."
        ),
        default_config_name="encoder_processor_decoder",
    )
    logging.basicConfig(level=logging.INFO)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Compose config
    cfg = load_config(args)

    # Setup datamodule and resolve config
    datamodule, cfg, stats = setup_datamodule(cfg)

    # Seed
    L.seed_everything(cfg.seed, workers=True)

    # Setup Model (includes AE loading, processor creation, ensemble logic)
    model = setup_epd_model(cfg, stats)

    # Run Training
    run_training(
        None,
        cfg,
        model,
        datamodule,
        work_dir,
        skip_test=args.skip_test,
        output_checkpoint_path=args.output_checkpoint,
        job_type="train-encoder-processor-decoder",
    )


if __name__ == "__main__":
    main()
