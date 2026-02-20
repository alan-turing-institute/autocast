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

uv run autocast train-eval --mode slurm \
    --dataset advection_diffusion_multichannel_64_64 \
    experiment=epd_flow_matching_64_fast \
    hydra.launcher.timeout_min=30 \
    autoencoder_checkpoint=/projects/u5gf/ai4physics/outputs/autoencoders/adm_64_1000.ckpt \
    --eval-overrides \
    eval.n_members=10 \
    datamodule.batch_size=8 \
    eval.max_test_batches=1 \
    eval.max_rollout_batches=1 \
    eval.batch_indices=[0]


