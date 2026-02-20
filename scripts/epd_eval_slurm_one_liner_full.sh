#!/bin/bash

uv run autocast train-eval --mode slurm \
    --dataset advection_diffusion_multichannel_64_64 \
    local_experiment=epd_crps_vit_latent_768_ps4_noise_32 \
    hydra.launcher.timeout_min=1440 \
    autoencoder_checkpoint=/projects/u5gf/ai4physics/outputs/autoencoders/adm_64_1000.ckpt \
    --eval-overrides \
    eval.n_members=10 \
    datamodule.batch_size=16 \
    eval.batch_indices=[0,1,2,3,4,5,6,7]
