"""Train autoencoder defined by Hydra config."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from autocast.scripts.training import train_autoencoder

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../../../configs", config_name="autoencoder"
)
def main(cfg: DictConfig) -> None:
    """CLI entrypoint for autoencoder training."""
    logging.basicConfig(level=logging.INFO)

    # Work directory is managed by Hydra
    work_dir = Path.cwd()

    train_autoencoder(cfg, work_dir)


if __name__ == "__main__":
    main()
