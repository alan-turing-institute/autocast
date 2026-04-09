# End-to-End Pipeline Tutorial

This tutorial demonstrates the full AutoCast pipeline: training an autoencoder, caching latents, training a processor, and running evaluation.

## Overview

The full AutoCast pipeline has the following stages:

1. **Train autoencoder** — Learn to compress and reconstruct spatiotemporal fields
2. **Cache latents** — Pre-compute latent representations for faster processor training
3. **Train processor** — Learn to evolve latent states forward in time
4. **Evaluate** — Assess forecasting quality on held-out data

## Step 1: Train the autoencoder

```bash
uv run autocast ae \
    datamodule=reaction_diffusion \
    encoder=dc \
    decoder=dc \
    trainer.max_epochs=20 \
    --run-group e2e-tutorial
```

## Step 2: Cache latents (optional)

Caching latents speeds up processor training by pre-computing encoder outputs:

```bash
uv run cache_latents \
    --checkpoint outputs/e2e-tutorial/00/encoder_decoder.ckpt
```

## Step 3: Train the processor

Train the full encoder-processor-decoder pipeline:

```bash
uv run autocast epd \
    datamodule=reaction_diffusion \
    processor=unet \
    trainer.max_epochs=50 \
    --run-group e2e-tutorial
```

## Step 4: Evaluate

```bash
uv run autocast eval --workdir outputs/e2e-tutorial/01
```

## All-in-one: train-eval

For convenience, use `train-eval` to run training and evaluation in a single command:

```bash
uv run autocast train-eval \
    datamodule=reaction_diffusion \
    --run-group e2e-tutorial
```

## Available processors

- **UNet** — Classic UNet architecture for latent evolution (`processor=unet`)
- **Swin ViT** — Swin Vision Transformer (`processor=swin_vit`)
- **ViT** — Axial Vision Transformer (`processor=vit`)
- **Flow Matching** — ODE-based flow matching processor (`processor=flow_matching`)
- **Diffusion** — Diffusion-based processor (`processor=diffusion`)

## SLURM support

For HPC environments, use `--mode slurm` to submit jobs:

```bash
uv run autocast train-eval \
    --mode slurm \
    datamodule=reaction_diffusion \
    --run-group e2e-tutorial \
    trainer.devices=4 \
    trainer.strategy=ddp \
    hydra.launcher.gpus_per_node=4
```

## Experiment tracking

Enable Weights & Biases logging:

```bash
uv run autocast epd \
    datamodule=reaction_diffusion \
    logging.wandb.enabled=true \
    logging.wandb.project=autocast-experiments \
    logging.wandb.name=e2e-baseline
```
