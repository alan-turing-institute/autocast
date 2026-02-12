#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00        
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --job-name epd_and_eval
#SBATCH --output=logs/epd_and_eval_%j.out
#SBATCH --error=logs/epd_and_eval_%j.err

set -e

# Enable write permissions for group
umask 0002

# Might be used within python scripts
export AUTOCAST_DATASETS="$PWD/datasets"

# Set configuration parameters
DATAPATH="advection_diffusion_multichannel_64_64" # Options: "advection_diffusion_multichannel_64_64", "advection_diffusion_multichannel"
USE_NORMALIZATION="false" # Options: "true" or "false"
MODEL="diffusion_vit" # Options (any compatible config in configs/processors/), currently: "flow_matching_vit", "diffusion_vit"
EPOCHS=720
BACKBONE="vit_256" # options: "vit_512", "vit_256"
EVAL_BATCH_SIZE=16
LEARNING_RATE=0.001
EVAL_ONLY="false"
WORKING_DIR=""

if [ ${DATAPATH} == "advection_diffusion_multichannel_64_64" ]; then
    AE_CHECKPOINT="/home/u5gf/ltcx7228.u5gf/autocast/outputs/autoencoders/adm_64_1000.ckpt"
elif [ ${DATAPATH} == "advection_diffusion_multichannel" ]; then
    AE_CHECKPOINT="/projects/u5gf/ai4physics/outputs/2026-02-06/advection_diffusion_multichannel_no_norm/autoencoder.ckpt"
fi

# One model params block for now since shared config pattern
MODEL_PARAMS=(
    "processor@model.processor=${MODEL}"
    "backbone@model.processor.backbone=${BACKBONE}"
    "datamodule.batch_size=128"
    "optimizer.learning_rate=${LEARNING_RATE}"
    "encoder@model.encoder=dc_deep_256_v2"
    "decoder@model.decoder=dc_deep_256_v2"
    "model.train_in_latent_space=true"
    "trainer.max_epochs=${EPOCHS}"
)

# Derive code and unique run identifiers
GIT_HASH=$(git rev-parse --short=7 HEAD | tr -d '\n')
UUID=$(uuidgen | tr -d '\n' | tail -c 7)

# Load dataset aliases
source "$PWD/slurm_templates/isambard/dataset_aliases.sh"

# Run name and working directory
RUN_NAME="diff_${DATA_SHORT}_${MODEL}_${BACKBONE}_${GIT_HASH}_${UUID}"

if [ ${EVAL_ONLY} = "false" ]; then
	WORKING_DIR="$PWD/outputs/$(date +%F)/${RUN_NAME}/"
else
	if [ "${WORKING_DIR}" = "" ]; then
		echo "Error: WORKING_DIR must be set when EVAL_ONLY is true."
		exit 1
	fi
fi

# Check if there's a pretrained autoencoder checkpoint in the working directory
if [ ${EVAL_ONLY} = "false" ]; then
	mkdir -p "${WORKING_DIR}"
	cd "${WORKING_DIR}"
	ln -s "${AE_CHECKPOINT}" autoencoder.ckpt
	cd -
fi
CKPT="${WORKING_DIR}/autoencoder.ckpt"
if [ -f "${CKPT}" ]; then
    MODEL_PARAMS+=( "+autoencoder_checkpoint=${CKPT}" )
fi

# Redirect output and error logs to the working directory
exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"


# Training
if [ ${EVAL_ONLY} = "false" ]; then
	srun uv run train_encoder_processor_decoder \
		hydra.run.dir=${WORKING_DIR} \
		datamodule="${DATAPATH}" \
		datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
		datamodule.use_normalization="${USE_NORMALIZATION}" \
		logging.wandb.enabled=true \
		logging.wandb.name="${RUN_NAME}" \
		"${MODEL_PARAMS[@]}"
fi
	

# Eval
CKPT_PATH="${WORKING_DIR}/encoder_processor_decoder.ckpt"
EVAL_DIR="${WORKING_DIR}/eval"

srun uv run evaluate_encoder_processor_decoder \
    hydra.run.dir="${EVAL_DIR}" \
    eval=encoder_processor_decoder \
    datamodule="${DATAPATH}" \
    datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
    +model.n_members=10 \
    eval.checkpoint=${CKPT_PATH} \
    eval.batch_indices=[0,1,2,3,4,5,6,7] \
    eval.video_dir="${EVAL_DIR}/videos" \
    "${MODEL_PARAMS[@]}" \
    datamodule.batch_size=${EVAL_BATCH_SIZE}
