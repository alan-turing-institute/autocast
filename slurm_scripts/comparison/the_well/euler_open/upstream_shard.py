"""Invoke LoLA's original cache_latent without its dawgz scheduler."""

import runpy
import sys
from pathlib import Path

from omegaconf import OmegaConf

if __name__ == "__main__":
    checkout, config, dataset, split, file = sys.argv[1:]
    sys.path.insert(0, checkout)
    script = Path(checkout) / "experiments" / "cache_latents.py"
    runpy.run_path(str(script))["cache_latent"](
        OmegaConf.load(config), dataset, split, file
    )
