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

Or alternatively with the included bash script:
```bash
./scripts/ae.sh rd 00 reaction_diffusion
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

Or alternatively with the included bash script:
```bash
./scripts/epd.sh rd 00 reaction_diffusion
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

Or alternatively with the included bash script:
```bash
./scripts/eval.sh rd 00 reaction_diffusion
```

## Experiment Tracking with Weights & Biases

AutoCast now ships with an optional [Weights & Biases](https://wandb.ai/) integration that is
fully driven by the Hydra config under `configs/logging/wandb.yaml`.

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

In the [slurm_templates](/slurm_templates/) folders, template slurm scripts can be found for the following use cases: 

- train_and_eval_autoencoder.sh : Training and evaluation of the autoencoder 
- train_and_eval_encoder-processor-decoder.sh : Training and evaluation of the encoder-processor-decoder approach
- encoder-processor-decoder-parameter_sweep : Same as above but runs a parameter sweep 

We advise you copy these scripts into a folder called `slurm_scripts` (which is in the gitignore) and edit as you see fit. 

### Single Job 

To run, simply navigate to the top level of this repository, and run: 

`sbatch scripts/train_and_eval_encoder-processor-decoder.sh` or 
`sbatch scripts/train_and_eval_autoencoder.sh` depending on which model you would like to run.

This will train and evaluate the model using the settings in the corresponding config (found in the configs folder). Outputs from both train and eval will be written out to an outputs folder with the following naming convention: 

`outputs/{job_name}/{$date +%Y%m%d_%H%M%S}`. 

### Multiple Jobs

`scripts/encoder-processor-decoder-parameter_sweep.sh` is an example parameter sweep. 

It uses slurm arrays and hydra override functionality to sweep through combinations of parameters. The resulting output structure looks like this: 

- outputs	
	- {job_name}
		- job-{job_id} # Unique for each sweep run 
			- parameter_lookup.csv # csv file mapping task id to parameter values. 
			- task-0 # 0 is the slurm array task id. It is unique for each set of parameters
			- task-1
			- etc. 

A checklist of things to change in the example script:

- `--array=0-8` : This is the number of parallel jobs to run. This should be equal to the number of parameter combinations you want to run. 
- `JOB_NAME="encoder_processor_decoder_sweep"` : Name of the Job. This is the top level directory. 
- The whole Define Parameter Grid section. 
- The columns to be writte to the parameter csv file 
- The flags in the python script to overright the hydra config. e.g. `trainer.max_epochs=${MAX_EPOCH}`

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