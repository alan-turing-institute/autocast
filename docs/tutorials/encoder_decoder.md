# Encoder-Decoder Tutorial

This tutorial demonstrates how to use AutoCast's encoder-decoder architecture programmatically.

## Overview

The encoder-decoder (autoencoder) compresses spatiotemporal fields into a compact latent representation and reconstructs them. This is typically the first step in the AutoCast pipeline before training a processor.

## Using the Python API

```python
import lightning as L
from autocast.data import SpatioTemporalDataModule
from autocast.models import EncoderDecoder

# Set up data
datamodule = SpatioTemporalDataModule(
    dataset_name="reaction_diffusion",
    batch_size=8,
)

# Create model
model = EncoderDecoder()

# Train
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule=datamodule)
```

## Using the CLI

```bash
uv run autocast ae \
    datamodule=reaction_diffusion \
    encoder=dc \
    decoder=dc \
    trainer.max_epochs=10 \
    --run-group ae-tutorial
```

## Available encoders

- **DC (Deep Compressed)** — Progressively downsamples the input (`encoder=dc`)
- **UNet** — UNet-based encoder (`encoder=unet`)
- **Identity** — No-op encoder, passes input through unchanged (`encoder=identity`)

## Available decoders

- **DC (Deep Compressed)** — Inverse of the DC encoder (`decoder=dc`)
- **Identity** — No-op decoder (`decoder=identity`)
- **Channels Last** — Channels-last decoder (`decoder=channels_last`)

## Resuming training

Resume from a checkpoint:

```bash
uv run autocast ae \
    --workdir outputs/ae-tutorial/00 \
    --resume-from outputs/ae-tutorial/00/encoder_decoder.ckpt
```

## Next steps

Once your autoencoder is trained, proceed to train a processor that evolves the latent state forward in time. See the [End-to-End Tutorial](end_to_end.md).
