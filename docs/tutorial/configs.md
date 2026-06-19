# Extending configurations

Hydra allows you to add or override any configuration value from the command line.
See [the Hydra documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) for more details.
As an example, to _override_ the number of training epochs for the `ae` command, you can run:

```bash
uv run autocast ae trainer.max_epochs=5
```

Note that this only works if the `trainer.max_epochs` key is defined in the default configuration for that command.
If the key is not defined, you have to prefix it with `+` to tell Hydra to add it:

```bash
uv run autocast ae +trainer.max_epochs=5
```

If you want to specify the option regardless of whether it is defined in the default config or not, you can use `++`:

```bash
uv run autocast ae ++trainer.max_epochs=5
```

In general, usage of `autocast` usually involves some invocation of `uv run autocast <command> <config-overrides>`.

On this page we'll describe some of the options you will almost certainly need to override when using `autocast`.

## Data modules

By default, these commands all point to their own _data modules_, which specify the dataset and how it gets loaded.
The data module configurations are stored in `src/autocast/configs/datamodule/`.

`autocast` can read datasets stored in two different formats:

| Format                                                                       | Appropriate setting for `datamodule` | Override...                                                    |
| ------                                                                       | ------------------------------------ | -----------                                                    |
| `.pt` files from [autosim](https://github.com/alan-turing-institute/autosim) | `datamodule=advection_diffusion`     | `datamodule.data_path`                                         |
| HDF5 files from [The Well](https://polymathic-ai.org/the_well/)              | `datamodule=the_well`                | `datamodule.well_base_path` and `datamodule.well_dataset_name` |

For example, let's say that you have used `autosim` to generate a dataset of advection-diffusion simulations.
We'll keep the size of the spatial grid (`simulator.n`) and the number of trajectories (`dataset.n_...`) small for this example.
We'll also manually specify the output directory for the generated dataset, so that we can point `autocast` to it later (otherwise `autosim` will automatically generate a directory for you by default):

```bash
# See the autosim repository for more information on this.

uv run autosim simulator=advection_diffusion \
    simulator.n=16 dataset.n_train=10 dataset.n_valid=2 dataset.n_test=2 \
    dataset.output_dir=/path/to/dataset
```

You can then train an autoencoder on that dataset (with all other settings inherited from that default) with:

```bash
uv run autocast ae \
    datamodule=advection_diffusion \
    datamodule.data_path=/path/to/dataset \
    +trainer.max_epochs=5
```

## Output paths

To specify the output directory for an experiment, you can either use the `--workdir` flag:

```bash
uv run autocast ae \
    --workdir /path/to/output/directory \
    datamodule=advection_diffusion \
    datamodule.data_path=/path/to/dataset \
    +trainer.max_epochs=5
```

or alternatively, specify `--run-group` and `--run-id`

```bash
uv run autocast ae \
    --run-group MYGROUP \
    --run-id MYID \
    datamodule=advection_diffusion \
    datamodule.data_path=/path/to/dataset \
    +trainer.max_epochs=5
```

and `autocast` will automatically use `outputs/MYGROUP/MYID` as the output directory.
If logging to Weights and Biases is enabled, the run ID will also be used for the default W&B run name.

`--run-group` defaults to the current date, and `--run-id` defaults to a legacy-style run id (a concatenation of dataset/model/hash/uuid).

## Using multiple GPUs / nodes

When running on SLURM with `--mode slurm` (see [the previous page](./commands.md)), you can enable multi-GPU and multi-node SLURM runs through distributed presets (found in `src/autocast/configs/distributed/`):

- To use 4 GPUs on a single node with DDP, add `++distributed=ddp_4gpu_slurm`
- To use 8 GPUs across 2 nodes with DDP, add `++distributed=ddp_4gpu_2node_slurm`
- To use 12 GPUs across 3 nodes with DDP, add `++distributed=ddp_4gpu_2node_slurm ++trainer.num_nodes=3 ++eval.num_nodes=3 ++hydra.launcher.nodes=3`

The preset configurations set both Lightning `trainer.devices`/`trainer.num_nodes` and the matching Slurm `hydra.launcher.nodes`/`gpus_per_node`/`tasks_per_node` values.
For more fine-grained control, you can also explicitly override these configuration values:

```bash
uv run autocast epd --mode slurm \
	datamodule=reaction_diffusion \
	trainer.devices=4 trainer.num_nodes=2 trainer.strategy=ddp \
	hydra.launcher.nodes=2 hydra.launcher.gpus_per_node=4 \
	hydra.launcher.tasks_per_node=4
```

## Resuming from a checkpoint

The following extra CLI options can be passed to the `ae`, `epd`, and `train-eval` subcommands (or added to configuration files):

- Resume from a saved checkpoint.
  The default is to perform a full-state resume, which restores model + optimizer/scheduler/trainer loop state.

  ```bash
  --resume-from path/to/encoder_processor_decoder.ckpt
  ```

- To additionally reset the timer budget:

  ```bash
  --resume-from path/to/encoder_processor_decoder.ckpt ++trainer.max_time="00:04:00:00" ++train_eval.reset_resume_time_budget=true
  ```

- To restore only the model weights and generate a fresh optimizer/trainer state:
 
  ```bash
  --resume-from path/to/encoder_processor_decoder.ckpt ++trainer.max_time="00:04:00:00" ++train_eval.resume_weights_only=true
  ```

  In conjunction with `trainer.max_time`, this allows you to continue training with a fresh timer budget.
  Note that if `resume_weights_only=true` is set without a checkpoint, AutoCast raises an error.


## Weights & Biases logging

AutoCast optionally integrates with [Weights & Biases](https://wandb.ai/); this is driven by the Hydra config in [`src/autocast/configs/logging/wandb.yaml`](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/logging/wandb.yaml).

Logging to W&B can be enabled (or disabled) with `++logging.wandb.enabled=true` (or `false` respectively).

The W&B project name can be set with `++logging.wandb.project=MYPROJECT`.

By default the `--run-id` is used as the W&B run name.
You can override this with `++logging.wandb.name=MYNAME`.

If you are resuming from a previous run and want to continue logging to the same W&B run, you need to set:
- `++logging.wandb.id=EXISTING_RUN_ID` (the run ID of the existing W&B run)
- `++logging.wandb.resume=allow` (or `must`)

Without this combination, W&B will create a new run even if the `logging.wandb.name` matches an existing run name.

## Making your own configurations

Finally, if you find yourself making the same overrides repeatedly, it is probably worth it to make a new YAML configuration that specifies these overrides.
If they are generally useful for the package, these can stored in `src/autocast/configs/experiments/<myexpt>.yaml` and specified on the command-line with `+experiment=<myexpt>`.

Some of the configurations we used for our experiments are stored in the `local_hydra/local_experiment` folder.
These are not in the main `src/autocast/configs` folder because they are not meant to be distributed as part of the package.
These configuration files can, however, still be used by setting `local_experiment=<name>` on the command line.

(This works because `autocast` makes sure to add `local_hydra` to Hydra's search path, allowing you to load configurations from there even though they are outside the package.)
