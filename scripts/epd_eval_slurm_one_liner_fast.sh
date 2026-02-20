#!/bin/bash
# Mapping guide:
# - Keys from src/autocast/configs/hydra/launcher/slurm.yaml map to CLI as
#   hydra.launcher.<key>=<value>
#   e.g. timeout_min -> hydra.launcher.timeout_min=30
# - Experiment preset lives in src/autocast/configs/experiment/epd_flow_matching_64_fast.yaml
#   and is applied with experiment=epd_flow_matching_64_fast.
# - In `autocast train-eval`, positional overrides apply to TRAIN.
# - Eval-specific settings go under `--eval-overrides`.
# - For different SLURM times per step, set:
#   TRAIN: hydra.launcher.timeout_min=<minutes>
#   EVAL:  --eval-overrides hydra.launcher.timeout_min=<minutes>

uv run autocast train-eval --mode slurm --detach \
    --dataset advection_diffusion_multichannel_64_64 \
    experiment=epd_flow_matching_64_fast \
    hydra.launcher.timeout_min=30 \
    autoencoder_checkpoint=/projects/u5gf/ai4physics/outputs/2026-02-06/advection_diffusion_multichannel_64_64_no_norm/autoencoder.ckpt \
    --eval-overrides \
    hydra.launcher.timeout_min=10 \
    +model.n_members=10

