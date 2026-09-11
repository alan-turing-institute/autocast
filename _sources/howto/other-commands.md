# Direct usage of lower-level Hydra scripts

The AutoCast CLI is a convenient wrapper around the lower-level Hydra scripts in `src/autocast/scripts/`.
Advanced users may want to invoke these scripts directly for more control.
Here are some example invocations:

## Train autoencoder script

```bash
uv run train_autoencoder \
	hydra.run.dir=outputs/rd/00 \
	datamodule.data_path=$AUTOCAST_DATASETS/reaction_diffusion \
	datamodule.use_simulator=false \
	optimizer.learning_rate=0.00005 \
	trainer.max_epochs=10 \
	logging.wandb.enabled=true
```

## Train processor script

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

## Evaluation script

```bash
uv run evaluate_encoder_processor_decoder \
	hydra.run.dir=outputs/rd/00/eval \
	eval.checkpoint=outputs/rd/00/encoder_processor_decoder.ckpt \
	eval.batch_indices=[0,1,2,3] \
	eval.video_dir=outputs/rd/00/eval/videos \
	datamodule.data_path=$AUTOCAST_DATASETS/reaction_diffusion \
	datamodule.use_simulator=false
```

