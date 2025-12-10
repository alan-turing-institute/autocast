# auto-cast

## Installation

For development, install with [`uv`](https://github.com/astral-sh/uv):
```bash
uv sync --extra dev
```

## Quickstart

Train an encoder-processor-decoder stack and evaluate the resulting checkpoint:

```bash
# Train
uv run train_processor --config-path=configs/ --work-dir=outputs/processor_run

# Evaluate
uv run evaluate_processor \
	--config-path=configs/ \
	--work-dir=outputs/processor_eval \
	--checkpoint=outputs/processor_run/encoder_processor_decoder.ckpt \
	--batch-index=0 --batch-index=3 \
	--video-dir=outputs/processor_eval/videos
```

Evaluation writes a CSV of aggregate metrics to `--csv-path` (defaults to
`<work-dir>/evaluation_metrics.csv`) and, when `--batch-index` is provided,
stores rollout animations for the specified test batches.