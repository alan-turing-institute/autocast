# AutoCast

## Installation

For development, install with [`uv`](https://github.com/astral-sh/uv):
```bash
uv sync --extra dev
```

## Quickstart

Train an encoder-decoder stack and evaluate the resulting checkpoint:

```bash
# Train
uv run python -m autocast.train.autoencoder --config-path=configs/
```

Train an encoder-processor-decoder stack and evaluate the resulting checkpoint:

```bash
# Train
uv run train_encoder_processor_decoder \
    --config-path=configs/ \
	--work-dir=outputs/encoder_processor_decoder_run
	
# Evaluate
uv run evaluate_encoder_processor_decoder \
	--config-path=configs/ \
	--work-dir=outputs/processor_eval \
	--checkpoint=outputs/encoder_processor_decoder_run/encoder_processor_decoder.ckpt \
	--batch-index=0 --batch-index=3 \
	--video-dir=outputs/encoder_processor_decoder_run/videos
```

Evaluation writes a CSV of aggregate metrics to `--csv-path` (defaults to
`<work-dir>/evaluation_metrics.csv`) and, when `--batch-index` is provided,
stores rollout animations for the specified test batches.

## Experiment Tracking with Weights & Biases

AutoCast now ships with an optional [Weights & Biases](https://wandb.ai/) integration that is
fully driven by the Hydra config under `configs/logging/wandb.yaml`.

- Enable logging for CLI workflows by passing Hydra config overrides as positional arguments:

	```bash
	uv run train_encoder_processor_decoder \
		--config-path=configs \
		logging.wandb.enabled=true \
		logging.wandb.project=autocast-experiments \
		logging.wandb.name=processor-baseline
	```

- The autoencoder/processor training CLIs pass the configured `WandbLogger` directly into Lightning so that metrics, checkpoints, and artifacts are synchronized automatically.
- The evaluation CLI reports aggregate test metrics to the same run when logging is enabled, making it easy to compare training and evaluation outputs in one dashboard.
- All notebooks contain a dedicated cell that instantiates a `wandb_logger` via `autocast.logging.create_wandb_logger`. Toggle the `enabled` flag in that cell to control tracking when experimenting interactively.

When `enabled` remains `false` (the default), the logger is skipped entirely, so the stack can
be used without a W&B account.