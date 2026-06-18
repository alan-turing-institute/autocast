# AutoCast <img src="https://raw.githubusercontent.com/alan-turing-institute/autocast/refs/heads/main/AC.png" align="right" height="138" />
[![All Contributors](https://img.shields.io/badge/all_contributors-10-orange.svg?style=flat-square)](#contributors-)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://alan-turing-institute.github.io/autocast/)

## Installation

### Prerequisites

- [uv](https://github.com/astral-sh/uv): running scripts; managing virtual environments
- [ffmpeg](https://ffmpeg.org/): optional video generation during evaluation

### Usage

If you'd just like to use the code in autocast:

```
# Clone the repo
git clone https://github.com/alan-turing-institute/autocast.git
cd autocast

# Install dependencies
uv sync
```

This will allow you to run `uv run autocast` from within the repository.

### Development

If you want to contribute to the autocast codebase, the following will get you set up:

```bash
# Clone the repo
git clone https://github.com/alan-turing-institute/autocast.git
cd autocast

# Install development dependencies
uv sync --extra dev

# Set up pre-commit checks, so that any pushed commits pass CI
uv run pre-commit install
```

## Introduction

`autocast` is primarily meant to be used as a CLI tool.

The `autocast` CLI is built on top of [Hydra](https://hydra.cc/).
This means that configurations are specified in YAML files and can be composed together to quickly switch between different datasets and model architectures.

### Base configurations and subcommands

The 'base' configurations for datasets and model architectures are stored in `src/autocast/configs`.
In particular, `autocast` comes with some subcommands for training standard model stacks:

| Command                     | Description                                | Default config                                                                                               |
| --------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `uv run autocast ae`        | Train an autoencoder                       | [`src/autocast/configs/autoencoder.yaml`](src/autocast/configs/autoencoder.yaml)
| `uv run autocast cache-latents` | Cache latents from an encoder | [`src/autocast/configs/cache_latents.yaml`](src/autocast/configs/cache_latents.yaml)                                 |
| `uv run autocast processor` | Train a processor (frozen encoder/decoder) | [`src/autocast/configs/processor.yaml`](src/autocast/configs/processor.yaml)                                 |
| `uv run autocast epd`       | Train an encoder-processor-decoder         | [`src/autocast/configs/encoder_processor_decoder.yaml`](src/autocast/configs/encoder_processor_decoder.yaml) |
| `uv run autocast eval`       | Evaluate a trained model         | [`src/autocast/configs/eval/encoder_processor_decoder.yaml`](src/autocast/configs/eval/encoder_processor_decoder.yaml) |

Notice that each of these YAML files in turn refer to a number of _other_ YAML files.
For example, `src/autocast/configs/autoencoder.yaml` specifies (amongst other things)

```yaml
defaults:
  - model: autoencoder
  - logging: wandb
```

which in turn point to `src/autocast/configs/model/autoencoder.yaml` and `src/autocast/configs/logging/wandb.yaml` respectively.
In this way, configurations can be built up from smaller pieces in a modular way.

### Adding and overriding configurations

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

### Data modules

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

### Output paths

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

### Making your own configurations

If you find yourself making the same overrides repeatedly, it is probably worth it to make a new YAML configuration that specifies these overrides.
If they are generally useful for the package, these can stored in `src/autocast/configs/experiments/<myexpt>.yaml` and specified on the command-line with `+experiment=<myexpt>`.

Some of the configurations we used for our experiments are stored in the `local_hydra/local_experiment` folder.
These are not in the main `src/autocast/configs` folder because they are not meant to be distributed as part of the package.
These configuration files can, however, still be used by setting `local_experiment=<name>` on the command line.

(This works because `autocast` makes sure to add `local_hydra` to Hydra's search path, allowing you to load configurations from there even though they are outside the package.)

## Running experiments on SLURM

`autocast` supports running experiments on SLURM clusters by adding the `--mode slurm` flag.
This automatically generates a submission Bash script and submits it to the cluster, so you don't have to worry about writing your own submission scripts.

For example, to run the same autoencoder training as above, but on SLURM, you can run:

```bash
uv run autocast ae --mode slurm \
    datamodule=advection_diffusion \
    datamodule.data_path=/path/to/dataset \
    +trainer.max_epochs=5
```

## Evaluating models

To run a series of preset evaluation tests on a saved model checkpoint, including single-step predictions and autoregressive rollout, you can use the `eval` subcommand and set `--workdir` to the run folder containing the configuration and model checkpoint to evaluate.

```bash
uv run autocast eval \
    --workdir /path/to/outputs
```

Some useful Hydra options for further controlling the evaluation are:

- `autoencoder_checkpoint`: the path to the autoencoder checkpoint to use for evaluation (if applicable).
  This is used if you trained a standalone processor (i.e., `uv run autocast processor`) in latent space.
  If you trained a full encoder-processor-decoder stack (i.e., `uv run autocast epd`), the autoencoder is already part of the model checkpoint, so does not need to be supplied separately.
- `eval.metrics`: a list of metrics to compute during evaluation.
- `eval.n_members`: the number of members to use for ensemble evaluation. Increasing this allows you to get a smoother estimate of the model's uncertainty.


## Other useful configuration flags

### Performing a dry run

Add `--dry-run` to print the commands that will be executed without actually running them.

### Using multiple GPUs / nodes

Multi-GPU and multi-node SLURM runs are supported through distributed presets (found in `src/autocast/configs/distributed/`):

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

### Resuming from a checkpoint

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


### Weights & Biases logging

AutoCast optionally integrates with [Weights & Biases](https://wandb.ai/); this is driven by the Hydra config in [`src/autocast/configs/logging/wandb.yaml`](src/autocast/configs/logging/wandb.yaml).

Logging to W&B can be enabled (or disabled) with `++logging.wandb.enabled=true` (or `false` respectively).

The W&B project name can be set with `++logging.wandb.project=MYPROJECT`.

By default the `--run-id` is used as the W&B run name.
You can override this with `++logging.wandb.name=MYNAME`.

If you are resuming from a previous run and want to continue logging to the same W&B run, you need to set:
- `++logging.wandb.id=EXISTING_RUN_ID` (the run ID of the existing W&B run)
- `++logging.wandb.resume=allow` (or `must`)

Without this combination, W&B will create a new run even if the `logging.wandb.name` matches an existing run name.

## Direct usage of lower-level Hydra scripts

The `autocast` CLI is a convenient wrapper around the lower-level Hydra scripts in `src/autocast/scripts/`.
Here are some example invocations:

#### Train autoencoder script
```bash
uv run train_autoencoder \
	hydra.run.dir=outputs/rd/00 \
	datamodule.data_path=$AUTOCAST_DATASETS/reaction_diffusion \
	datamodule.use_simulator=false \
	optimizer.learning_rate=0.00005 \
	trainer.max_epochs=10 \
	logging.wandb.enabled=true
```

#### Train processor script
```bash
uv run train_encoder_processor_decoder \
	hydra.run.dir=outputs/rd/00 \
	datamodule.data_path=$AUTOCAST_DATASETS/reaction_diffusion \
	datamodule.use_simulator=false \
	optimizer.learning_rate=0.0001 \
	trainer.max_epochs=10 \
	logging.wandb.enabled=true \
	'autoencoder_checkpoint=outputs/rd/00/autoencoder.ckpt'
```

#### Evaluation script
```bash
uv run evaluate_encoder_processor_decoder \
	hydra.run.dir=outputs/rd/00/eval \
	eval.checkpoint=outputs/rd/00/encoder_processor_decoder.ckpt \
	eval.batch_indices=[0,1,2,3] \
	eval.video_dir=outputs/rd/00/eval/videos \
	datamodule.data_path=$AUTOCAST_DATASETS/reaction_diffusion \
	datamodule.use_simulator=false
```

## Ethical guidance

See [ETHICAL_GUIDANCE.md](ETHICAL_GUIDANCE.md) for guidance on the intended scope and use of AutoCast.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://www.jasonmcewen.org"><img src="https://avatars.githubusercontent.com/u/3181701?v=4?s=100" width="100px;" alt="Jason McEwen "/><br /><sub><b>Jason McEwen </b></sub></a><br /><a href="#ideas-jasonmcewen" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-jasonmcewen" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/radka-j"><img src="https://avatars.githubusercontent.com/u/29207091?v=4?s=100" width="100px;" alt="Radka Jersakova"/><br /><sub><b>Radka Jersakova</b></sub></a><br /><a href="#ideas-radka-j" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-radka-j" title="Project Management">📆</a> <a href="https://github.com/alan-turing-institute/autocast/commits?author=radka-j" title="Code">💻</a> <a href="https://github.com/alan-turing-institute/autocast/pulls?q=is%3Apr+reviewed-by%3Aradka-j" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://paolo-conti.com/"><img src="https://avatars.githubusercontent.com/u/51111500?v=4?s=100" width="100px;" alt="Paolo Conti"/><br /><sub><b>Paolo Conti</b></sub></a><br /><a href="#ideas-ContiPaolo" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/autocast/commits?author=ContiPaolo" title="Code">💻</a> <a href="https://github.com/alan-turing-institute/autocast/pulls?q=is%3Apr+reviewed-by%3AContiPaolo" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/marjanfamili"><img src="https://avatars.githubusercontent.com/u/44607686?v=4?s=100" width="100px;" alt="Marjan Famili"/><br /><sub><b>Marjan Famili</b></sub></a><br /><a href="#ideas-marjanfamili" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/autocast/commits?author=marjanfamili" title="Code">💻</a> <a href="https://github.com/alan-turing-institute/autocast/pulls?q=is%3Apr+reviewed-by%3Amarjanfamili" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://cisprague.github.io/"><img src="https://avatars.githubusercontent.com/u/17131395?v=4?s=100" width="100px;" alt="Christopher Iliffe Sprague"/><br /><sub><b>Christopher Iliffe Sprague</b></sub></a><br /><a href="#ideas-cisprague" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/autocast/commits?author=cisprague" title="Code">💻</a> <a href="https://github.com/alan-turing-institute/autocast/pulls?q=is%3Apr+reviewed-by%3Acisprague" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/EdwinB12"><img src="https://avatars.githubusercontent.com/u/64434531?v=4?s=100" width="100px;" alt="Edwin "/><br /><sub><b>Edwin </b></sub></a><br /><a href="#ideas-EdwinB12" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/autocast/commits?author=EdwinB12" title="Code">💻</a> <a href="https://github.com/alan-turing-institute/autocast/pulls?q=is%3Apr+reviewed-by%3AEdwinB12" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sgreenbury"><img src="https://avatars.githubusercontent.com/u/50113363?v=4?s=100" width="100px;" alt="Sam Greenbury"/><br /><sub><b>Sam Greenbury</b></sub></a><br /><a href="#ideas-sgreenbury" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-sgreenbury" title="Project Management">📆</a> <a href="https://github.com/alan-turing-institute/autocast/commits?author=sgreenbury" title="Code">💻</a> <a href="https://github.com/alan-turing-institute/autocast/pulls?q=is%3Apr+reviewed-by%3Asgreenbury" title="Reviewed Pull Requests">👀</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/qiencai"><img src="https://avatars.githubusercontent.com/u/185296522?v=4?s=100" width="100px;" alt="QC"/><br /><sub><b>QC</b></sub></a><br /><a href="#ideas-qiencai" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/autocast/commits?author=qiencai" title="Code">💻</a> <a href="https://github.com/alan-turing-institute/autocast/issues?q=author%3Aqiencai" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/penelopeysm"><img src="https://avatars.githubusercontent.com/u/122629585?v=4?s=100" width="100px;" alt="Penelope Yong"/><br /><sub><b>Penelope Yong</b></sub></a><br /><a href="#ideas-penelopeysm" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/autocast/commits?author=penelopeysm" title="Code">💻</a> <a href="https://github.com/alan-turing-institute/autocast/issues?q=author%3Apenelopeysm" title="Bug reports">🐛</a> <a href="https://github.com/alan-turing-institute/autocast/pulls?q=is%3Apr+reviewed-by%3Apenelopeysm" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/farhanferoz"><img src="https://avatars.githubusercontent.com/u/34422685?v=4?s=100" width="100px;" alt="farhanferoz"/><br /><sub><b>farhanferoz</b></sub></a><br /><a href="#ideas-farhanferoz" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/autocast/commits?author=farhanferoz" title="Code">💻</a> <a href="https://github.com/alan-turing-institute/autocast/issues?q=author%3Afarhanferoz" title="Bug reports">🐛</a> <a href="https://github.com/alan-turing-institute/autocast/pulls?q=is%3Apr+reviewed-by%3Afarhanferoz" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/acocac"><img src="https://avatars.githubusercontent.com/u/13321552?v=4?s=100" width="100px;" alt="Alejandro ©"/><br /><sub><b>Alejandro ©</b></sub></a><br /><a href="https://github.com/alan-turing-institute/autocast/issues?q=author%3Aacocac" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
