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

Train an encoder-decoder stack and evaluate the resulting checkpoint:

```bash
# Train
uv run train_autoencoder
```

Train an encoder-processor-decoder stack and evaluate the resulting checkpoint:

```bash
# Train
uv run train_encoder_processor_decoder \
	hydra.run.dir=outputs/encoder_processor_decoder_run
	
# Evaluate
uv run evaluate_encoder_processor_decoder \
	hydra.run.dir=outputs/processor_eval \
	eval.checkpoint=outputs/encoder_processor_decoder_run/encoder_processor_decoder.ckpt \
	eval.batch_indices=[0,1] \
	eval.video_dir=outputs/encoder_processor_decoder_run/videos
```

Evaluation writes a CSV of aggregate metrics to `eval.csv_path` (defaults to
`<work-dir>/evaluation_metrics.csv`) and, when `eval.batch_indices` is provided,
stores rollout animations for the specified test batches.

## Example pipeline

This assumes you have the `reaction_diffusion` dataset stored at the path specified by
the `AUTOCAST_DATASETS` environment variable.

### Train autoencoder
```bash
uv run train_autoencoder \
	hydra.run.dir=outputs/rd/00 \
	datamodule.data_path=$AUTOCAST_DATASETS/reaction_diffusion \
	datamodule.use_simulator=false \
	optimizer.learning_rate=0.00005 \
	trainer.max_epochs=10 \
	logging.wandb.enabled=true
```

Or alternatively with the unified workflow CLI:
```bash
uv run autocast ae \
	--dataset reaction_diffusion \
	--run-label rd
```

Unified workflow CLI supports both local and SLURM launch modes:

```bash
# Local (default)
uv run autocast epd \
	--dataset reaction_diffusion \
	--run-label my_label \
	trainer.max_epochs=5

# SLURM submit-and-exit via sbatch
uv run autocast epd \
	--mode slurm \
	--dataset reaction_diffusion \
	--run-label my_label \
	trainer.max_epochs=5
```

When `--mode slurm`, `autocast` writes an sbatch script, submits it, and exits
immediately. Outputs are written under `outputs/<run_label>/<run_id>`.

Resume training from a checkpoint:
```bash
uv run autocast epd \
	--dataset reaction_diffusion \
	--workdir outputs/rd/00 \
	--resume-from outputs/rd/00/encoder_processor_decoder.ckpt
```

Train + evaluate in one command:
```bash
uv run autocast train-eval \
	--dataset reaction_diffusion \
	--run-label rd
```

For `train-eval`, evaluation starts only after training has completed successfully
(including in `--mode slurm`).

Execution modes for `train-eval`:
- one SLURM job runs train then eval.

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

Permissions quick reference:
- Training/eval scripts use config key `umask` (default `0002` in `encoder_processor_decoder`).

Use `--dry-run` to print resolved commands/scripts without executing.

Equivalent CLI commands for removed `slurm_scripts/*.sh` examples:
```bash
bash scripts/cli_equivalents.sh
```

Launch many prewritten runs from a manifest file:
```bash
bash scripts/launch_from_manifest.sh run_manifests/example_runs.txt
```

Date handling is automatic: if `--date` is omitted, current date is used.
Run naming is also automatic: if `--run-name` is omitted, `autocast` generates
a legacy-style run id (dataset/model/hash/uuid based) and uses it for both
the run folder and default `logging.wandb.name`.
Pass `--run-label` (or legacy alias `--date`) only to override the top-level folder label.

Multi-GPU is supported by passing trainer/Hydra overrides, e.g.:
```bash
uv run autocast epd --mode slurm --dataset reaction_diffusion \
	trainer.devices=4 trainer.strategy=ddp hydra.launcher.gpus_per_node=4
```
```

### Train processor

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

Or alternatively with the unified workflow CLI:
```bash
uv run autocast epd \
	--dataset reaction_diffusion \
	--run-label rd
```

### Evaluation
```bash
uv run evaluate_encoder_processor_decoder \
	hydra.run.dir=outputs/rd/00/eval \
	eval.checkpoint=outputs/rd/00/encoder_processor_decoder.ckpt \
	eval.batch_indices=[0,1,2,3] \
	eval.video_dir=outputs/rd/00/eval/videos \
	datamodule.data_path=$AUTOCAST_DATASETS/reaction_diffusion \
	datamodule.use_simulator=false
```

Or alternatively with the unified workflow CLI:
```bash
uv run autocast eval \
	--dataset reaction_diffusion \
	--workdir outputs/rd/00
```

## Experiment Tracking with Weights & Biases

AutoCast now ships with an optional [Weights & Biases](https://wandb.ai/) integration that is
fully driven by the Hydra config under `src/autocast/configs/logging/wandb.yaml`.

- Enable logging for CLI workflows by passing Hydra config overrides as positional arguments:

	```bash
	uv run train_encoder_processor_decoder \
		logging.wandb.enabled=true \
		logging.wandb.project=autocast-experiments \
		logging.wandb.name=processor-baseline
	```

- The autoencoder/processor training CLIs pass the configured `WandbLogger` directly into Lightning so that metrics, checkpoints, and artifacts are synchronized automatically.
- The evaluation CLI reports aggregate test metrics to the same run when logging is enabled, making it easy to compare training and evaluation outputs in one dashboard.
- All notebooks contain a dedicated cell that instantiates a `wandb_logger` via `autocast.logging.create_wandb_logger`. Toggle the `enabled` flag in that cell to control tracking when experimenting interactively.

When `enabled` remains `false` (the default), the logger is skipped entirely, so the stack can
be used without a W&B account.

## Running on HPC 

### Legacy SLURM Scripts

Legacy scripts under `slurm_scripts/` have been retired to reduce duplication and
maintenance overhead. Use the unified `autocast` CLI workflows instead:

```bash
# train->eval in one SLURM job
uv run autocast train-eval --mode slurm --dataset reaction_diffusion

# run many prewritten jobs from a manifest
bash scripts/launch_from_manifest.sh run_manifests/example_runs.txt
```

### Single Job 

To run a single job from this repository, use `autocast` directly, for example:

`uv run autocast epd --mode slurm --dataset reaction_diffusion trainer.max_epochs=10`

This will train and evaluate the model using the settings in the corresponding config (found in the configs folder). Outputs from both train and eval will be written out to an outputs folder with the following naming convention: 

`outputs/{job_name}/{$date +%Y%m%d_%H%M%S}`. 

### Multiple Jobs

Use Hydra multi-run directly for sweeps (or the manifest launcher), e.g.
`uv run autocast epd --mode slurm --dataset reaction_diffusion trainer.max_epochs=5,10`.

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