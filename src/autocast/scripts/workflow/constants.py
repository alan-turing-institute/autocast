"""Shared constants for the workflow CLI."""

from __future__ import annotations

TRAIN_MODULES: dict[str, str] = {
    "ae": "autocast.scripts.train.autoencoder",
    "epd": "autocast.scripts.train.encoder_processor_decoder",
    "processor": "autocast.scripts.train.processor",
}

EVAL_MODULE = "autocast.scripts.eval.encoder_processor_decoder"
BENCHMARK_MODULE = "autocast.scripts.benchmark.encoder_processor_decoder"
TRAIN_EVAL_MODULE = "autocast.scripts.train_eval.encoder_processor_decoder"
CACHE_LATENTS_MODULE = "autocast.scripts.cache_latents"

NAMING_DEFAULT_KEYS: set[str] = {
    "processor@model.processor",
    "input_noise_injector@model.input_noise_injector",
}

DATASET_NAME_TOKENS: dict[str, str] = {
    "advection_diffusion_multichannel_64_64": "adm64",
    "advection_diffusion_multichannel": "adm32",
    "advection_diffusion": "ad64",
    "reaction_diffusion": "rd64",
    "gray_scott": "gs64",
    "lattice_boltzmann_128_32": "lb128x32",
    "conditioned_navier_stokes": "cns64",
    "gpe_low_complexity": "gpelc64",
    "gpe_high_complexity": "gpehc64",
    "shallow_water2d": "sw2d64",
    "shallow_water2d_4": "sw2d464",
}
