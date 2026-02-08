#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00        
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --job-name epd_and_eval
#SBATCH --output=logs/epd_and_eval%j.out
#SBATCH --error=logs/epd_and_evla%j.err

set -e
uv sync --extra dev

export AUTOCAST_DATASETS="$PWD/datasets"
DATAPATH="advection_diffusion_multichannel"
WORKING_DIR="$PWD/outputs/2026-02-06/${DATAPATH}_no_norm/"
USE_NORMALIZATION="false"

exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"


# Flow matching
MODEL_PARAMS=(
    "optimizer.learning_rate=0.0002"
    "encoder@model.encoder=dc_deep_256"
    "decoder@model.decoder=dc_deep_256"
)


# CRPS loss with ViT_large
# MODEL_PARAMS=(
#      "optimizer.learning_rate=0.0002"
#      "encoder@model.encoder=permute_concat"
#      "decoder@model.decoder=channels_last"
#      "processor@model.processor=vit_large"
#      "model.processor.n_noise_channels=1024"
#      "model.processor.spatial_resolution=[32,32]"
#      "model.processor.patch_size=1"
#      "+model.n_members=10"
#      "model.loss_func._target_=autocast.losses.ensemble.CRPSLoss"
#      "+model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS"
# )

CKPT="${WORKDIR}/autoencoder.ckpt"
if [ -f "${CKPT}" ]; then
    MODEL_PARAMS+=( "+autoencoder_checkpoint=${CKPT}" )
fi

srun uv run train_encoder_processor_decoder \
    hydra.run.dir=${WORKING_DIR} \
	datamodule="${DATAPATH}" \
	datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
	datamodule.use_normalization="${USE_NORMALIZATION}" \
	trainer.max_epochs=100 \
	logging.wandb.enabled=true \
	 "${MODEL_PARAMS[@]}"
	
	
	
CKPT_PATH="${WORKING_DIR}/encoder_processor_decoder.ckpt"
EVAL_DIR="${WORKING_DIR}/eval"

uv run evaluate_encoder_processor_decoder \
        hydra.run.dir="${EVAL_DIR}" \
        eval=encoder_processor_decoder \
        datamodule="${DATAPATH}" \
        datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
        eval.checkpoint=${CKPT_PATH} \
        eval.batch_indices=[0,1,2,3] \
        eval.video_dir="${EVAL_DIR}/videos" \
	"${MODEL_PARAMS[@]}"
