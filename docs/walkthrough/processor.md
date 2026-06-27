# Training a processor

In [the previous page](./autoencoder.md), we showed how to train an autoencoder model and how to use it to encode (cache) data into a latent space.
We'll now show how to train a processor model, i.e., one which actually does the spatiotemporal forecasting task.

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
