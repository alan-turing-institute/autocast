# Training an autoencoder

We'll begin by training an autoencoder on the AD dataset we just simulated.

`autocast` provides a high-level tool for training autoencoders: `uv run autocast ae <options>`.
The default options are stored in YAML configuration files, which can be found in the `src/autocast/configs` directory.
For example, the default configuration for autoencoders is [here](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/autoencoder.yaml).

:::{note}
Both `autocast` and `autosim` use Hydra, a library which allows you to compose YAML configurations to form full specifications for experiments.
You can read more about Hydra in [its docs](https://hydra.cc/).
:::

## Specifying the dataset

If you look at the default configuration linked above, it points to a `datamodule` called `autosim`, which is itself a configuration file [here](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/datamodule/autosim.yaml).

The datamodule is a specification of where the dataset is, how to load it, and other details such as normalisation.
In this case, the `autosim` datamodule is a blank slate which lets us pass the path to any `autosim`-generated dataset.
It has a `data_path` field which we can set to the path of the dataset we generated in the previous section.

We'll specify this by providing the `datamodule.data_path` field as a command-line override.

## Autoencoder hyperparameters

The autoencoder configuration also points to a `model: autoencoder` ([here](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/model/autoencoder.yaml)), which in turn points to an `encoder: dc_deep_256_v2` ([here](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/encoder/dc_deep_256_v2.yaml)).
These configurations specify the architecture of the encoder and decoder networks, as well as the loss function (mean-squared error loss).

## Training

Additionally, to make this process quick, we'll train only for 10 epochs.
We'll also provide the `workdir` option to specify where the output of the training should be saved.

```bash
uv run autocast ae \
    --workdir ../ae_output \
    ++datamodule.data_path=/path/to/parent_folder/ad_data \
    ++trainer.max_epochs=10
```

Notice that there are two different kinds of options here: the `--workdir` option is a standard command-line option that is directly passed to and used by `autocast`, whereas the `++` options are Hydra configuration overrides.

:::{warning}
`datamodule.data_path` has to be provided as an absolute path, not a relative path.
:::

This should finish within a very reasonable time even on a modest laptop.

During the training, information about the model as well as progress logs will be printed.
For instance, we can see the summary of the model architecture and the number of parameters:

```
  | Name         | Type             | Params | Mode
----------------------------------------------------------
0 | encoder      | DCEncoder        | 3.5 M  | train
1 | decoder      | DCDecoder        | 3.5 M  | train
2 | loss_func    | AELoss           | 0      | train
3 | val_metrics  | MetricCollection | 0      | train
4 | test_metrics | MetricCollection | 0      | train
----------------------------------------------------------
```

Once the training is complete, we can find the trained model in the `ae_output` folder:

```
parent_folder
├── ad_data
├── ae_output
│   ├── autoencoder.ckpt
│   ├── autoencoder.log
│   ├── lightning_logs
│   │   └── version_0
│   │       ├── checkpoints
│   │       │   ├── best-val-0009-0.0008.ckpt
│   │       │   ├── last.ckpt
│   │       │   └── ...  (periodic snapshot checkpoints)
│   │       └── metrics.csv
│   ├── reconstructions
│   │   ├── batch_00.png
│   │   ├── batch_01.png
│   │   └── batch_02.png
│   ├── resolved_autoencoder_config.yaml
│   └── validation_metrics
│       └── validation_metrics.png
├── autocast
└── autosim
```

The most important file here is `autoencoder.ckpt`, which contains the trained model weights; although a variety of other checkpoints are also saved which may be useful for debugging or analysis.
The `reconstructions` folder contains some example reconstructions of the input data (i.e. a roundtrip through the encoder), and the `validation_metrics` folder contains plots of different kinds of validation loss metrics over the course of training.

## Generating latent representations

Once the autoencoder has been trained, we can use it to generate latent representations of the data.
This is done with the `uv run autocast cache-latents` command.

It's by far easiest to set the `--workdir` flag to be the autoencoder's output directory: that way, `autocast` will automatically pick up the configuration file `resolved_autoencoder_config.yaml` and use that when generating the latents.

```
uv run autocast cache-latents \
    --workdir ../ae_output \
    --output-dir ../ae_output/cached_latents \
    ++datamodule.data_path=/path/to/parent_folder/ad_data \
    ++autoencoder_checkpoint=/path/to/parent_folder/ae_output/autoencoder.ckpt
```

This will generate a new folder `ae_output/cached_latents`:

```
parent_folder
├── ad_data
├── ae_output
│   ├── autoencoder.ckpt
│   ├── autoencoder.log
│   ├── cache_latents.log
│   ├── cached_latents
│   │   ├── autoencoder_config.yaml
│   │   ├── metadata.json
│   │   ├── test
│   │   │   ├── traj_000000.pt
│   │   │   ├── traj_000001.pt
│   │   │   └── traj_000002.pt
│   │   ├── train
│   │   │   ├── traj_000000.pt
│   │   │   ├── traj_000001.pt
│   │   │   ├── traj_000002.pt
│   │   │   ├── traj_000003.pt
│   │   │   ├── traj_000004.pt
│   │   │   ├── traj_000005.pt
│   │   │   ├── traj_000006.pt
│   │   │   ├── traj_000007.pt
│   │   │   ├── traj_000008.pt
│   │   │   └── traj_000009.pt
│   │   └── valid
│   │       ├── traj_000000.pt
│   │       ├── traj_000001.pt
│   │       └── traj_000002.pt
│   └── ...
├── autocast
└── autosim
```

We can check the sizes of the latent representations by loading one of the `.pt` files in the `cached_latents` folder:

```python
>>> import torch

>>> torch.load("../ae_output/cached_latents/train/traj_000000.pt")["encoded_fields"].shape
torch.Size([11, 4, 4, 2])
```

Because each trajectory has been split into its own `.pt` file, we only have four dimensions remaining:

- 11 time steps
- 4x4 spatial grid
- 2 latent channels

The time steps are the same as before, but the spatial dimensions have been reduced from 16×16 to 4×4, and instead of 1 input channel we now have 2 latent channels: this is a result of the autoencoder architecture.
The encoder has three depth levels (configured via `hid_channels`) with a stride of 2 between each pair, so there are two downsampling steps giving a total spatial reduction of 2^2 = 4 in each dimension.
The number of latent channels is controlled by the `out_channels` setting in the encoder config.
