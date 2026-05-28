# AutoCast <img src="https://raw.githubusercontent.com/alan-turing-institute/autocast/refs/heads/main/AC.png" align="right" height="138" />
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-10-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## Installation

### Prerequisites

- [uv](https://github.com/astral-sh/uv): running scripts; managing virtual environments
- [ffmpeg](https://ffmpeg.org/): optional video generation during evaluation

### Usage

If you'd just like to use the code in autocast:

```
# Clone the repo
git clone git@github.com:alan-turing-institute/autocast.git
cd autocast

# Install dependencies
uv sync
```

This will allow you to run `uv run autocast` from within the repository.

### Development

If you want to contribute to the autocast codebase, the following will get you set up:

```bash
# Clone the repo
git clone git@github.com:alan-turing-institute/autocast.git
cd autocast

# Install development dependencies
uv sync --extra dev

# Set up pre-commit checks, so that any pushed commits pass CI
uv run pre-commit install
```

## Introduction

`autocast` is primarily meant to be used as a command-line tool.

The `autocast` CLI is built on top of [Hydra](https://hydra.cc/).
This means that configurations are specified in YAML files and can be composed together to quickly switch between different datasets and model architectures.

### Base configurations and subcommands

The 'base' configurations for datasets and model architectures are stored in `src/autocast/configs`.
In particular, `autocast` comes with some subcommands for training standard model stacks:

| Command                     | Description                                | Default config                                                                                               |
| --------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `uv run autocast ae`        | Train an autoencoder                       | [`src/autocast/configs/autoencoder.yaml`](src/autocast/configs/autoencoder.yaml)
| `uv run autocast processor` | Train a processor (frozen encoder/decoder) | [`src/autocast/configs/processor.yaml`](src/autocast/configs/processor.yaml)                                 |
| `uv run autocast epd`       | Train an encoder-processor-decoder         | [`src/autocast/configs/encoder_processor_decoder.yaml`](src/autocast/configs/encoder_processor_decoder.yaml) |

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
| HDF5 files from [The Well](https://polymathic-ai.org/the_well/)              | `datamodule=the_well`                | `datamodule.well_base_path` and `datamodule.well_dataset_name` |
| `.pt` files from [autosim](https://github.com/alan-turing-institute/autosim) | `datamodule=advection_diffusion`     | `datamodule.data_path`                                         |

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

To specify the output directory for an experiment, you can use the `--workdir` flag:

```bash
uv run autocast ae \
    --workdir /path/to/output/directory \
    datamodule=advection_diffusion \
    datamodule.data_path=/path/to/dataset \
    +trainer.max_epochs=5
```

Or alternatively, specify `--run-group=RUN_GROUP` and `--run-id=RUN_ID` to have `autocast` automatically generate the output directory as `outputs/RUN_GROUP/RUN_ID`.
If logging to Weights and Biases is enabled, the run ID will also be used for the default W&B run name.

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

---------------------------------

## Example pipeline

This assumes you have the `reaction_diffusion` dataset stored at the path specified by
the `AUTOCAST_DATASETS` environment variable.

### Autocast API

#### Core commands and workflow options
```bash
uv run autocast ae \
	datamodule=reaction_diffusion \
	--run-group rd
```

Unified workflow CLI supports both local and SLURM launch modes:

```bash
# Local (default)
uv run autocast epd \
	datamodule=reaction_diffusion \
	--run-group my_label \
	trainer.max_epochs=5

# SLURM submit-and-exit via sbatch
uv run autocast epd \
	--mode slurm \
	datamodule=reaction_diffusion \
	--run-group my_label \
	trainer.max_epochs=5
```

When `--mode slurm`, `autocast` writes an sbatch script, submits it, and exits
immediately. Outputs are written under `outputs/<run_group>/<run_id>`.

Resume training from a checkpoint:
```bash
uv run autocast epd \
	datamodule=reaction_diffusion \
	--workdir outputs/rd/00 \
	--resume-from outputs/rd/00/encoder_processor_decoder.ckpt
```

Resume modes (applies to `ae`, `epd`, and `train-eval`):
- Full-state resume (default): restores model + optimizer/scheduler/trainer loop state.
- Weights-only resume: restores model weights only (fresh optimizer/trainer state), useful for "train for another N hours" workflows with `trainer.max_time`.
- Full-state + reset timer budget: keeps optimizer/scheduler state and clears the restored elapsed-time offset for Lightning's `max_time` timer.

```bash
# Weights-only resume (example for train-eval)
uv run autocast train-eval \
	--workdir outputs/rd/00 \
	--resume-from outputs/rd/00/encoder_processor_decoder.ckpt \
	trainer.max_time="00:04:00:00" \
	+train_eval.resume_weights_only=true
```

If `resume_weights_only=true` is set without a checkpoint, AutoCast raises an error.

If you want to keep optimizer state and still run for a fresh `max_time` window, use:

```bash
uv run autocast train-eval \
	--workdir outputs/rd/00 \
	--resume-from outputs/rd/00/encoder_processor_decoder.ckpt \
	trainer.max_time="00:04:00:00" \
	+train_eval.reset_resume_time_budget=true
```

Train + evaluate in one command:
```bash
uv run autocast train-eval \
	datamodule=reaction_diffusion \
	--run-group rd
```

For `train-eval`, evaluation starts only after training has completed successfully
(including in `--mode slurm`).

To run  `eval` on a previously trained model, set `--workdir` to the run folder containing the configuration and checkpoint to evaluate:
```bash
uv run autocast eval --workdir outputs/rd/00
```

#### Configuration and overrides
Keep private experiment presets in `local_hydra/local_experiment/` and select
them with `local_experiment=<name>`. YAML files in that folder are ignored by
git by default.

To load configs from a separate directory (including packaged installs), set:

```bash
export AUTOCAST_CONFIG_PATH=/absolute/path/to/configs
```

Override mapping quick reference:
- `configs/hydra/launcher/slurm.yaml` key `X` maps to CLI `hydra.launcher.X=...`
- Use `hydra/launcher=slurm_baskerville` for Baskerville module/setup defaults
from `local_hydra/hydra/launcher/slurm_baskerville.yaml`.
- In `autocast train-eval`, positional overrides are train-only.
- Eval-only overrides go in `--eval-overrides ...`.
- `--eval-overrides` is a separator: place train overrides before it and eval
overrides after it.

Permissions quick reference:
- Lower-level Hydra training/evaluation scripts use config key `umask` (default `0002` in `encoder_processor_decoder`).

Use `--dry-run` to print resolved commands/scripts without executing.

Launch many prewritten runs from a manifest file:
```bash
bash scripts/launch_from_manifest.sh run_manifests/example_runs.txt
```

Date handling is automatic: if `--run-group` is omitted, current date is used.
Run naming is also automatic: if `--run-id` is omitted, `autocast` generates
a legacy-style run id (dataset/model/hash/uuid based) and uses it for both
the run folder and default `logging.wandb.name`.
Pass `--run-group` only to override the top-level folder label.
Backward-compatible aliases remain available: `--run-label` and `--run-name`.

W&B naming behavior:
- `--run-group` only changes the parent output folder (`outputs/<run_group>/...`).
- `--run-id` sets the run folder name and, by default, `logging.wandb.name`.
- Set `logging.wandb.name=...` via Hydra overrides to explicitly name the W&B run.

Workdir/chdir behavior:
- The workflow wrapper always forwards `--workdir` to Hydra as `hydra.run.dir=<workdir>`.
- Training configs set `hydra.job.chdir=true`, so script execution runs inside that run directory.
- Script internals resolve workdir via Hydra runtime output dir (`resolve_hydra_work_dir(...)`), avoiding cwd/path mismatches.

Multi-GPU and multi-node SLURM runs are supported through distributed presets
under `src/autocast/configs/distributed/`:
```bash
uv run autocast epd --mode slurm \
	datamodule=reaction_diffusion \
	+distributed=ddp_4gpu_slurm
```

For a 2-node, 4-GPU-per-node DDP run:
```bash
uv run autocast epd --mode slurm \
	datamodule=reaction_diffusion \
	+distributed=ddp_4gpu_2node_slurm
```

The preset sets both Lightning `trainer.devices`/`trainer.num_nodes` and the
matching Slurm `hydra.launcher.nodes`/`gpus_per_node`/`tasks_per_node` values.
Equivalent explicit overrides also work, e.g.:
```bash
uv run autocast epd --mode slurm \
	datamodule=reaction_diffusion \
	trainer.devices=4 trainer.num_nodes=2 trainer.strategy=ddp \
	hydra.launcher.nodes=2 hydra.launcher.gpus_per_node=4 \
	hydra.launcher.tasks_per_node=4
```

### Experiment Tracking with Weights & Biases

AutoCast optionally integrates with [Weights & Biases](https://wandb.ai/) that is
 driven by the Hydra config under `src/autocast/configs/logging/wandb.yaml`.

Enable logging by passing Hydra config overrides as positional arguments:

```bash
uv run autocast epd \
	logging.wandb.enabled=true \
	logging.wandb.project=autocast-experiments \
	logging.wandb.name=processor-baseline
```

To continue an existing W&B run on resume, set both:
- `logging.wandb.id=<existing_run_id>`
- `logging.wandb.resume=allow` (or `must`)

Without `id`+`resume`, W&B creates a new run even if `logging.wandb.name` matches.
Also note:
- Full-state resume continues Lightning global step/epoch progression.
- Weights-only resume starts trainer loop counters from zero (new optimizer/trainer state).

All example notebooks contain a dedicated cell that instantiates a `wandb_logger` via `autocast.logging.create_wandb_logger`. Toggle the `enabled` flag in that cell to control tracking when experimenting interactively.

When `enabled` remains `false` (the default), the logger is skipped entirely, so the stack can be used without a W&B account.

## Direct usage of lower-level Hydra scripts

The `autocast` CLI is a convenient wrapper around the lower-level Hydra scripts in `src/autocast/scripts/`. You can run those directly if you prefer, for example:

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

## Running on HPC 

The `autocast` CLI directly supports SLURM submission via `--mode slurm`.
This section is a quick reference for common HPC usage.

For single-job SLURM usage (`autocast epd --mode slurm` or
`autocast train-eval --mode slurm`), see the examples above in
`Example pipeline`.

### Multiple Jobs

Use Hydra multi-run directly for sweeps, e.g.
`uv run autocast epd --mode slurm datamodule=reaction_diffusion trainer.max_epochs=5,10`.

Or launch prewritten jobs from a manifest:
`bash scripts/launch_from_manifest.sh run_manifests/example_runs.txt`.

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
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
