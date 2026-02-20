#!/bin/bash
uv run autocast train-eval --mode slurm \
	--dataset advection_diffusion_multichannel_64_64 \
	datamodule.use_normalization=false \
	logging.wandb.enabled=false \
	processor@model.processor=flow_matching_vit \
	datamodule.batch_size=128 \
	optimizer.learning_rate=0.0002 \
	encoder@model.encoder=dc_deep_256 \
	decoder@model.decoder=dc_deep_256 \
	model.train_in_latent_space=true \
	model.processor.backbone.hid_channels=512 \
	trainer.max_epochs=200 \
	autoencoder_checkpoint=/projects/u5gf/ai4physics/outputs/2026-02-06/advection_diffusion_multichannel_64_64_no_norm/autoencoder.ckpt \
	--eval-overrides \
	+model.n_members=10 \
	eval.batch_indices=[0,1,2,3]
