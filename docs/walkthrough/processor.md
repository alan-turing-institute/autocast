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


## Residual flow matching

A residual processor uses one coordinate system for inputs, targets, references
and reconstructed outputs: ambient with an identity codec, latent with a frozen
codec or cached latents. Let `x` and `y` be dataset-normalized states, `E`/`D` the
encoder/decoder, and `b` a deterministic reference in processor coordinates.

| Workflow | Residual target `r` | Forecast before dataset denormalization |
| --- | --- | --- |
| Ambient, identity codec | `y - b(x)` | `b(x) + generated_residual` |
| Latent, frozen codec | `E(y) - b(E(x))` | `D(b(E(x)) + generated_residual)` |

`processor.residual_normalizer` optionally applies `(r - mean) / scale` before
flow matching and reverses it before adding the reference. Its `normalize()` and
`denormalize()` methods only change residual coordinates; dataset normalization
uses the existing dataset/model `norm`, and physical units are restored after
full-state reconstruction and decoding. `map` and `map_from_source` return full
processor states, so cached-latent outputs still require the matching decoder.

For nonlinear codecs, `E(y) - E(x)` differs from `E(y - x)`, and decoding a sum
is not the sum of separately decoded terms. Latent persistence with a lossy codec
returns `D(E(x_last))` at zero residual, which need not match the physical last
frame. Caches, references and evaluation must use the same encoder checkpoint
and dataset normalization. Online latent training loads the matching pretrained
codec with `autoencoder_checkpoint=/path/to/autoencoder.ckpt`; the frozen residual
codec stays in evaluation mode throughout training.

To use residual flow matching in the ambient command above, replace/add:

```bash
processor@model.processor=residual_flow_matching_vit \
model.train_in_latent_space=true \
model.loss_func=null \
model.residual_prediction=false
```

`train_in_latent_space=true` selects the processor's flow matching objective;
with an identity encoder those coordinates remain ambient. `false` selects a
loss on decoded forecasts and does not change residual coordinates. Keep
`model.loss_func=null` for ensemble flow matching too. The separate decoded-state
increment interface uses an ordinary processor with `model.residual_prediction=true`,
`model.train_in_latent_space=false` and a decoded-state loss. Combining that option
with residual flow matching would add a second reference and is rejected.

For cached latents:

```bash
uv run autocast processor \
    --workdir ../residual_latent_output \
    datamodule=cached_latents \
    ++datamodule.data_path=/path/to/parent_folder/ae_output/cached_latents \
    processor@model.processor=residual_flow_matching_vit \
    model.loss_func=null
```

The default reference repeats the latest frame in processor coordinates. Input
and forecast channels must align; use `reference.channel_indices` when inputs
include additional conditioning channels. `ProcessorReference` is a Python API
for wrapping an **already loaded, deterministic** processor with matching input
coordinates, output horizon and channels. It does not infer nested `auto` values
or load pretrained weights. For a matching resolved config and processor-model
checkpoint:

```python
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from autocast.processors.residual_reference import ProcessorReference

config = OmegaConf.load("/path/to/run/resolved_config.yaml")
processor = instantiate(config.model.processor)
state = torch.load("/path/to/run/processor.ckpt", map_location="cpu", weights_only=False)["state_dict"]
processor.load_state_dict(
    {key.removeprefix("processor."): value for key, value in state.items()
     if key.startswith("processor.")},
    strict=True,
)
reference = ProcessorReference(processor)
```

Freezing disables gradients and dropout; it does not make a generative `map`
deterministic. An external ambient reference for a latent processor needs an
explicit adapter handling input normalization and output encoding.

With an unfitted residual normalizer, the training entry point inserts
`ResidualStatisticsCallback` before EMA. It fits training-only population moments
while preserving RNG state. Raw batches require a frozen encoder and
`train_in_latent_space=true`; encoder/reference evaluation uses Trainer precision,
and moments use FP32 reductions with CPU FP64 accumulation. Automatic fitting
does not support configured input noise: remove the injector or supply fixed
statistics. Custom Lightning loops must install this callback before EMA.

Distributed fitting partitions a finite map-style training DataLoader across
ranks without padding and combines count/mean/M2. It requires standard full-dataset
samplers, implicit batching via `batch_size`, and `drop_last=False`; iterable
datasets, subset/replacement samplers, sampler subclasses or an explicit
`batch_sampler` need fixed statistics instead. Represent a training
subset as a dataset rather than a sampler to use automatic distributed fitting.
Sampler data-source lengths must match the loader's dataset length, and both the
loader and any distributed sampler must use `drop_last=False`.
Fitting covers the full training split, independently of `limit_train_batches`.
Mean, scale and the fitted flag are checkpoint buffers, so loading/resuming skips
refitting. Omit the transform with `model.processor.residual_normalizer=null`,
or provide both fixed statistics using Hydra's append syntax:

```bash
+model.processor.residual_normalizer.mean=0 \
+model.processor.residual_normalizer.scale=1
```

Existing saved configs using `ResidualStandardizer`/`standardizer` and checkpoint
keys under `standardizer.*` remain loadable, including raw and EMA resume state.
New configs and checkpoints use `ResidualNormalizer`/`residual_normalizer`.
Providing both names is an error.

Sources default to IID Gaussian. Configure prescribed correlations with
`flow_source@model.processor.source=temporal_ou` or `separable_gaussian`; covariance
parameters are not fitted. These sources currently accept 2-D spatial grids.
Boundary settings apply to the processor grid, which may be latent, and do not
enforce physical boundaries or conservation on forecasts. For `map_from_source`,
provide full conditioning states as `x` and initial noise with the forecast shape,
dtype and device in normalized residual coordinates; the result is reconstructed
processor states.
