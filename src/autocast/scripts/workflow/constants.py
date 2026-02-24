"""Shared constants for the workflow CLI."""

from __future__ import annotations

TRAIN_MODULES: dict[str, str] = {
    "ae": "autocast.scripts.train.autoencoder",
    "epd": "autocast.scripts.train.encoder_processor_decoder",
    "processor": "autocast.scripts.train.processor",
}

EVAL_MODULE = "autocast.scripts.eval.encoder_processor_decoder"
TRAIN_EVAL_MODULE = "autocast.scripts.train_eval.encoder_processor_decoder"

NAMING_DEFAULT_KEYS: set[str] = {
    "processor@model.processor",
    "input_noise_injector@model.input_noise_injector",
}

DATASET_NAME_TOKENS: dict[str, str] = {
    "advection_diffusion_multichannel_64_64": "adm64",
    "advection_diffusion_multichannel": "adm32",
    "advection_diffusion_singlechannel": "ad32",
    "reaction_diffusion": "rd32",
}
