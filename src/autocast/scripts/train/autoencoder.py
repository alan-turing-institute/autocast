"""Train autoencoder defined by Hydra config."""

import logging

from autocast.scripts.cli import parse_common_args
from autocast.scripts.config import load_config
from autocast.scripts.training import train_autoencoder

log = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint for autoencoder training."""
    args = parse_common_args(
        description=("Train autoencoder defined by Hydra config under configs/."),
        config_name="autoencoder",
    )
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    train_autoencoder(cfg, work_dir)


if __name__ == "__main__":
    main()
