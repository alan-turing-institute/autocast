"""Train an encoder-processor-decoder with optional pretrained autoencoder."""

import logging

import lightning as L

from autocast.scripts.cli import parse_common_args
from autocast.scripts.config import load_config
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.training import run_training

log = logging.getLogger(__name__)


def main():
    """CLI entrypoint for training the processor."""
    args = parse_common_args(
        description=(
            "Train an encoder-processor-decoder model with Hydra-configured "
            "encoder, decoder, and processor components."
        ),
        config_name="encoder_processor_decoder",
    )

    # Setup logging and working directory
    logging.basicConfig(level=logging.INFO)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Compose config
    cfg = load_config(args)

    # Setup datamodule and resolve config
    datamodule, cfg, stats = setup_datamodule(cfg)

    # Seed
    L.seed_everything(cfg.get("seed", 42), workers=True)

    # Setup Model (includes AE loading, processor creation, ensemble logic)
    model = setup_epd_model(cfg, stats)

    # Run Training
    run_training(
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
