# Quickstart

This tutorial covers the basics of using AutoCast to train and evaluate a spatiotemporal forecasting model.

## Overview

AutoCast uses a modular encoder-processor-decoder architecture:

1. **Encoder** — compresses high-dimensional spatiotemporal fields into a compact latent representation
2. **Processor** — evolves the latent state forward in time
3. **Decoder** — reconstructs the full spatiotemporal field from the latent state

## Training an autoencoder

The simplest way to get started is to train an autoencoder (encoder-decoder) on a dataset:

```bash
uv run autocast ae \
    datamodule=reaction_diffusion \
    trainer.max_epochs=5
```

## Training an encoder-processor-decoder

Once you have a trained autoencoder, you can train the full pipeline:

```bash
uv run autocast epd \
    datamodule=reaction_diffusion \
    trainer.max_epochs=10
```

## Training and evaluating in one step

Use the `train-eval` command to train and evaluate in a single run:

```bash
uv run autocast train-eval \
    datamodule=reaction_diffusion \
    --run-group quickstart
```

## Evaluating a trained model

To evaluate a previously trained model:

```bash
uv run autocast eval --workdir outputs/quickstart/00
```

Evaluation writes a CSV of aggregate metrics to `eval.csv_path` and, when `eval.batch_indices` is provided,
stores rollout animations for the specified test batches.

## Configuration

AutoCast uses [Hydra](https://hydra.cc/) for configuration management. You can override any configuration parameter from the command line:

```bash
uv run autocast ae \
    datamodule=reaction_diffusion \
    encoder=dc \
    decoder=dc \
    trainer.max_epochs=20
```

For more details on configuration, see the [Scripts and Configs documentation](https://github.com/alan-turing-institute/autocast/blob/main/docs/SCRIPTS_AND_CONFIGS.md).
