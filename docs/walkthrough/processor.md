# Training a processor

In [the previous page](./autoencoder.md), we showed how to train an autoencoder model and how to use it to encode (cache) data into a latent space.
We'll now show how to train a processor model, i.e., one which actually does the spatiotemporal forecasting task.
We first start by training a processor in the latent space, and then show how to train a processor in the original ambient space.

You should have the following directory structure with several subfolders (other irrelevant files are hidden for brevity):

```
parent_folder
├── ad_data
├── ae_output
│   └── cached_latents
├── autocast
└── autosim
```

Processor models can be trained with the `uv run autocast processor` command.
Much like before, we're going to have to specify some overrides.

First of all, we want to point the processor to the cached latents we generated in the previous step.
AutoCast already provides [a helpful `cached_latents` datamodule configuration file](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/datamodule/cached_latents.yaml), which we can go ahead and directly use: we additionally need to specify the path to the actual files.

Like before, we'll restrict training to 10 epochs to make it quick.

```
 uv run autocast processor \
      --workdir ../proc_output \
      datamodule=cached_latents \
      ++datamodule.data_path=/path/to/parent_folder/ae_output/cached_latents \
      ++trainer.max_epochs=10
```

:::{note}
Notice that here `datamodule` is not prefixed with `++`.
The reason for this is because it's a [default-list override in Hydra](https://hydra.cc/docs/advanced/defaults_list/): `cached_latents` is a YAML file itself whose entire contents need to be loaded.
In contrast, `datamodule.data_path` is a single value that is being overridden.
If we were to write `++datamodule=cached_latents`, that would set the value of `datamodule` to the plain string `"cached_latents"`, not the corresponding configuration!
:::

The default processor model is a flow matching processor with a U-Net backbone.
(To see this, you can trace the configuration files: the default configuration for `uv run autocast processor` is [`src/autocast/configs/processor.yaml`](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/processor.yaml), which in turn loads [`model: processor`](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/model/processor.yaml), which in turn loads [`processor: flow_matching`](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/processor/flow_matching.yaml).
The `processor` directory contains a number of different model architectures which can be swapped out.

Once training is complete, you should see a new folder, `proc_output`:

```
parent_folder
├── ad_data
├── ae_output
│   └── cached_latents
├── autocast
├── autosim
└── proc_output
    ├── lightning_logs
    │   └── version_0
    │       ├── checkpoints
    │       │   ├── best-val-0008-2.5373.ckpt
    │       │   ├── last.ckpt
    │       │   ├── snapshot-0p06-0000-00000003.ckpt
    │       │   └── (more snapshots...)
    │       └── metrics.csv
    ├── processor.ckpt
    ├── processor.log
    ├── resolved_config.yaml
    └── validation_metrics
        └── validation_metrics.png
```

The structure of this directory is very similar to the autoencoder output: there are trained model checkpoints, validation metrics, and configuration files which help with reproducibility.

On a successful run, `processor.ckpt` always contains the last checkpoint.
We can check this again by running Python from the AutoCast directory:

```python
>>> import torch

>>> ckpt = torch.load('../proc_output/processor.ckpt')

>>> ckpt["epoch"]
10
```

If you instead want to load the best checkpoint according to validation metrics, you can do so by loading the `best-val-*.ckpt` file in the `checkpoints` subfolder.

## Processor coordinate systems

A processor operates entirely in the coordinate system of the tensors it
receives. For a cached-latent workflow these are latent coordinates; with
identity encoder and decoder modules they are ambient coordinates. A processor
does not implicitly decode its output or reverse dataset normalization.

Residual processors likewise form their target and reference in processor
coordinates. Residual standardization is an additional affine transform inside
that coordinate system and is distinct from dataset normalization. A reference
provided by another processor must therefore accept the same inputs and return
the same ambient or latent coordinates, output horizon, and channels. An
external ambient model must be encoded before it can provide a reference to a
latent processor.

## Residual flow matching

The API has one coordinate contract: inputs, targets, references and reconstructed
outputs use the processor's coordinates. Let `E` and `D` denote the frozen encoder
and decoder, and `x` and `y` the input and target after any dataset normalization.

| Workflow | Residual training target before standardization | Forecast after decoding, before dataset denormalization |
| --- | --- | --- |
| Ambient, identity codec | `y - reference(x)` | `reference(x) + generated_residual` |
| Latent, frozen codec | `E(y) - reference(E(x))` | `D(reference(E(x)) + generated_residual)` |

Cached-latent training uses the second row with `E(x)` and `E(y)` already stored;
the processor returns complete latent states and decoding happens separately.
Caches, references and evaluation must use the same encoder checkpoint and data
normalization. Dataset denormalization happens after full-state reconstruction.
For online latent training, configure the matching encoder and decoder and load
their pretrained weights with `autoencoder_checkpoint=/path/to/autoencoder.ckpt`;
keep `model.train_in_latent_space=true` to use the processor's objective.

For nonlinear codecs, `E(y) - E(x)` is generally different from `E(y - x)`;
likewise, `D(reference + residual)` is not `D(reference) + D(residual)`.
Latent persistence is a baseline in representation space. With a lossy codec,
even a zero latent residual returns `D(E(x_last))`, which need not exactly match
the last physical frame. This PR does not supply an adapter for running an
external ambient reference model inside a latent processor: such an adapter must
explicitly handle its input coordinates, normalization and output encoding.

For example, train against a persistence reference in ambient coordinates:

```bash
uv run autocast epd \
    --workdir ../residual_output \
    ++datamodule.data_path=/path/to/parent_folder/ad_data \
    encoder@model.encoder=identity \
    decoder@model.decoder=identity \
    processor@model.processor=residual_flow_matching_vit \
    model.train_in_latent_space=true \
    model.loss_func=null \
    model.residual_prediction=false
```

Here `train_in_latent_space=true` selects the processor's flow matching loss
after the frozen encoder; the identity encoder keeps these coordinates ambient.
Set `model.loss_func=null` to retain the processor's flow matching objective when
using an ensemble model too. Setting `train_in_latent_space=false` selects a loss
on decoded forecasts instead; it does not switch the residual's coordinates.
Automatic residual-statistics fitting for raw batches requires the frozen codec
and `train_in_latent_space=true`. The frozen residual-flow codec stays in
evaluation mode during training, so dropout and running statistics cannot change
the coordinate system after fitting.

For cached latents:

```bash
uv run autocast processor \
    --workdir ../residual_latent_output \
    datamodule=cached_latents \
    ++datamodule.data_path=/path/to/parent_folder/ae_output/cached_latents \
    processor@model.processor=residual_flow_matching_vit \
    model.loss_func=null
```

The reference then repeats the latest latent frame, and reconstructed outputs
remain latent until passed through the matching decoder.

The training target is `(target - reference - mean) / scale`. Both `map` and
`map_from_source` return `reference + mean + scale * generated_residual`.
Keep `model.residual_prediction=false`: that separate option reconstructs
increments after decoding and would add a second reference.
If an exact ambient additive reference is needed with a non-identity decoder,
the separate decoded-state residual interface uses an ordinary processor,
`model.residual_prediction=true`, `model.train_in_latent_space=false` and a
decoded-state loss. It is a different training objective from residual flow
matching in this processor.

The supplied configuration starts with IID Gaussian noise. To prescribe a
correlated source, add `flow_source@model.processor.source=temporal_ou` or
`flow_source@model.processor.source=separable_gaussian`. These covariance
parameters are configured explicitly; they are not fitted from data.
`ProcessorReference` can instead wrap a pretrained processor, frozen by default;
its output must match the residual processor's coordinates, horizon and channels.
Use a deterministic reference. `frozen=true` disables gradients and dropout but
does not make a generative processor's stochastic `map` deterministic. A fresh,
unconditioned reference draw would change the baseline between residual training
and reconstruction. For the persistence reference, input and output channels must
align; use `reference.channel_indices` to select the forecast channels when
conditioning includes extra channels.

Structured-source length scales and boundary settings apply to the processor
grid, which may be a latent grid. They do not enforce physical boundary conditions
or conservation on the learned vector field, decoded forecasts or rollouts.
For `map_from_source`, supply initial noise in the standardized residual flow's
coordinates, with the output shape, dtype and device; supply complete conditioning
states to `x`. The method returns reconstructed states, not standardized residuals.

With an unfitted standardizer, the training entry point inserts
`ResidualStatisticsCallback` before EMA initialization. It fits mean and
population scale using only the training loader, with a fixed reference and
frozen encoder, and preserves RNG state. Input noise must be disabled during
this fit. Mean, scale and the fitted flag are checkpoint buffers; loading or
resuming a checkpoint restores them and skips the fitting pass. For a custom
Lightning loop, install this callback before EMA yourself. Set
`model.processor.standardizer=null` to omit residual standardization, or provide
both `mean` and `scale` to use fixed statistics.

## Processors in ambient space

The processor models in AutoCast are designed to operate in the latent space, i.e., with encoded data.
If you want to run a processor in the original ambient space, you can do so by training a full `encoder_processor_decoder` stack and setting both `encoder` and `decoder` to no-ops.

Here is the full command (explained later):

```
uv run autocast epd \
    --workdir ../epd_output \
    ++datamodule.data_path=/path/to/parent_folder/ad_data \
    encoder@model.encoder=identity \
    decoder@model.decoder=identity \
    processor@model.processor=flow_matching \
    ++trainer.max_epochs=10
```

- The main command for training a full stack is `uv run autocast epd`.
- Much like for the autoencoder, the default `datamodule` for `epd` is `autosim`.
  We just need to specify `++datamodule.data_path` to point to the original data.
- We override both the encoder and decoder to be `identity`, i.e., no-ops.
- For this demonstration we set `processor` to be the same U-Net flow matching model used above.
  The default processor for `epd` is in fact `flow_matching_vit`, which uses a ViT backbone instead and is much larger (around 50M parameters compared to `flow_matching`'s 3.5M).
