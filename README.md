# AutoCast
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-7-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## Installation

### Prereqiuisites

- [uv](https://github.com/astral-sh/uv): running scripts; managing virtual environments
- [ffmpeg](https://ffmpeg.org/): optional video generation during evaluation

### Development
For development, install with [`uv`](https://github.com/astral-sh/uv):
```bash
uv sync --extra dev
```

If contributing to the codebase, you can run 
```bash 
 pre-commit install 
 ```
This will setup the pre-commit checks so any pushed commits will pass the CI. 

For detailed documentation on the available scripts and configuration system, see [docs/SCRIPTS_AND_CONFIGS.md](docs/SCRIPTS_AND_CONFIGS.md).

## Quickstart

The `autocast` CLI is built on top of [Hydra](https://hydra.cc/), meaning you can pass configuration overrides directly to the commands.

Train an encoder-decoder stack:

```bash
uv run autocast ae trainer.max_epochs=5
```

Train and evaluate an encoder-processor-decoder stack:

```bash
uv run autocast train-eval datamodule=reaction_diffusion
```

Evaluation writes a CSV of aggregate metrics to `eval.csv_path` and, when `eval.batch_indices` is provided,
stores rollout animations for the specified test batches.

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

Multi-GPU is supported by passing trainer/Hydra overrides, e.g.:
```bash
uv run autocast epd --mode slurm \
	datamodule=reaction_diffusion \
	trainer.devices=4 trainer.strategy=ddp hydra.launcher.gpus_per_node=4
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
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!