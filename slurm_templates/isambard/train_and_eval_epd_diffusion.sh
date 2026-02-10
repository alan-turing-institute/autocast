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

# Might be used within python scripts
export AUTOCAST_DATASETS="$PWD/datasets"

# Set configuration parameters
DATAPATH="advection_diffusion_multichannel_64_64" # Options: "advection_diffusion_multichannel_64_64", "advection_diffusion_multichannel"
USE_NORMALIZATION="false" # Options: "true" or "false"
MODEL="flow_matching_vit" # Options (any compatible config in configs/processors/), currently: "flow_matching_vit", "diffusion_vit"

if [ ${DATAPATH} == "advection_diffusion_multichannel_64_64" ]; then
    AE_CHECKPOINT="/projects/u5gf/ai4physics/outputs/2026-02-06/advection_diffusion_multichannel_64_64_no_norm/autoencoder.ckpt"
elif [ ${DATAPATH} == "advection_diffusion_multichannel" ]; then
    AE_CHECKPOINT="/projects/u5gf/ai4physics/outputs/2026-02-06/advection_diffusion_multichannel_no_norm/autoencoder.ckpt"
fi


# # Hidden dimension parameters
# if [ ${MODEL} == "vit_large" ]; then
#     HIDDEN_PARAMS="model.processor.hidden_dim=${HIDDEN_DIM}"
# else
#     HIDDEN_PARAMS="model.processor.hidden_channels=${HIDDEN_DIM}"
# fi

# One model params block for now since shared config pattern
MODEL_PARAMS=(
    "processor@model.processor=${MODEL}"
    "datamodule.batch_size=128"
    "optimizer.learning_rate=0.0002"
    "encoder@model.encoder=dc_deep_256"
    "decoder@model.decoder=dc_deep_256"
    "model.train_in_latent_space=true"
)

# Derive code and unique run identifiers
GIT_HASH=$(git rev-parse --short=7 HEAD | tr -d '\n')
UUID=$(uuidgen | tr -d '\n' | tail -c 7)

# Run name and working directory
RUN_NAME="diff_${DATAPATH}_${MODEL}_${MODEL_NOISE}_${HIDDEN_DIM}_${GIT_HASH}_${UUID}"
WORKING_DIR="$PWD/outputs/$(date +%F)/${RUN_NAME}/"

# Check if there's a pretrained autoencoder checkpoint in the working directory
cd "${WORKING_DIR}"
ln -s "${AE_CHECKPOINT}" autoencoder.ckpt
cd -
CKPT="${WORKING_DIR}/autoencoder.ckpt"
if [ -f "${CKPT}" ]; then
    MODEL_PARAMS+=( "+autoencoder_checkpoint=${CKPT}" )
fi

# Make directories and redirect output and error logs to the working directory
mkdir -p $WORKING_DIR
exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"


# Training
srun uv run train_encoder_processor_decoder \
    hydra.run.dir=${WORKING_DIR} \
	datamodule="${DATAPATH}" \
	datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
	datamodule.use_normalization="${USE_NORMALIZATION}" \
	logging.wandb.enabled=true \
	logging.wandb.name="${RUN_NAME}" \
	 "${MODEL_PARAMS[@]}"
	

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
    eval.batch_indices=[0,1,2,3] \
    eval.video_dir="${EVAL_DIR}/videos" \
    "${MODEL_PARAMS[@]}"










# --------------------
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00        
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --job-name epd_and_eval
#SBATCH --output=logs/epd_and_eval%j.out
#SBATCH --error=logs/epd_and_eval%j.err

set -e

export AUTOCAST_DATASETS="$PWD/datasets"

# Set configuration parameters
DATAPATH="advection_diffusion_multichannel_64_64"
USE_NORMALIZATION="false"
# MODEL="fno" # ViT
MODEL="vit_large" # FNO
HIDDEN_DIM=256
MODEL_NOISE="concat" # Options: "cln", "concat"

# Derive remaining parameters based on the dataset and model choices
if [ ${DATAPATH} == "advection_diffusion_multichannel_64_64" ]; then
    NOISE_CHANNELS=4096
else
    NOISE_CHANNELS=1024
fi
GIT_HASH=$(git rev-parse --short=7 HEAD | tr -d '\n')
UUID=$(uuidgen | tr -d '\n' | tail -c 7)
RUN_NAME="${DATAPATH}_${MODEL}_${MODEL_NOISE}_${HIDDEN_DIM}_crps_no_norm_${GIT_HASH}_${UUID}"
WORKING_DIR="$PWD/outputs/$(date +%F)/${RUN_NAME}/"

mkdir -p $WORKING_DIR
exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"


# Flow matching
# MODEL_PARAMS=(
#     "optimizer.learning_rate=0.0002"
#     "encoder@model.encoder=dc_deep_256"
#     "decoder@model.decoder=dc_deep_256"
# )

# Hidden dimension parameters
if [ ${MODEL} == "vit_large" ]; then
    HIDDEN_PARAMS="model.processor.hidden_dim=${HIDDEN_DIM}"
else
    HIDDEN_PARAMS="model.processor.hidden_channels=${HIDDEN_DIM}"
fi

# Input noise injection
if [ ${MODEL_NOISE} == "cln" ]; then
    MODEL_NOISE_PARAMS="model.processor.n_noise_channels=${NOISE_CHANNELS}"
else
    MODEL_NOISE_PARAMS="input_noise_injector@model.input_noise_injector=concat"
fi

# Spatial resolution parameters
if [ ${DATAPATH} == "advection_diffusion_multichannel_64_64" ]; then
    SPATIAL_RESOLUTION_PARAMS="model.processor.spatial_resolution=[64,64]"
else
    SPATIAL_RESOLUTION_PARAMS="model.processor.spatial_resolution=[32,32]"
fi

if [ ${MODEL} == "vit_large" ]; then
    MODEL_SPECIFIC_PARAMS=(
        "processor@model.processor=${MODEL}"
        "${MODEL_NOISE_PARAMS}"
        "${SPATIAL_RESOLUTION_PARAMS}"
        "${HIDDEN_PARAMS}"
        "model.processor.patch_size=null"
		"datamodule.batch_size=64"
    )
elif [ ${MODEL} == "fno" ]; then
    MODEL_SPECIFIC_PARAMS=(
        "processor@model.processor=${MODEL}"
        "${HIDDEN_PARAMS}"
        "${MODEL_NOISE_PARAMS}"
		"datamodule.batch_size=16"
    )
fi

# Combine all model parameters
MODEL_PARAMS=(
     "optimizer.learning_rate=0.0002"
     "encoder@model.encoder=permute_concat"
     "model.encoder.with_constants=true"
     "decoder@model.decoder=channels_last"
)
MODEL_PARAMS+=("${MODEL_SPECIFIC_PARAMS[@]}")
MODEL_PARAMS+=(
	 "model.train_in_latent_space=false"
     "+model.n_members=10"
     "model.loss_func._target_=autocast.losses.ensemble.CRPSLoss"
     "+model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS"
)

CKPT="${WORKING_DIR}/autoencoder.ckpt"
if [ -f "${CKPT}" ]; then
        MODEL_PARAMS+=( "+autoencoder_checkpoint=${CKPT}" )
fi

# Training (many hrs)
srun uv run train_encoder_processor_decoder \
    hydra.run.dir=${WORKING_DIR} \
	datamodule="${DATAPATH}" \
	datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
	datamodule.use_normalization="${USE_NORMALIZATION}" \
	logging.wandb.enabled=true \
	logging.wandb.name="${RUN_NAME}" \
	 "${MODEL_PARAMS[@]}"
	
	
	
# Eval (5-10 mins)
CKPT_PATH="${WORKING_DIR}/encoder_processor_decoder.ckpt"
EVAL_DIR="${WORKING_DIR}/eval"

srun uv run evaluate_encoder_processor_decoder \
    hydra.run.dir="${EVAL_DIR}" \
    eval=encoder_processor_decoder \
    datamodule="${DATAPATH}" \
    datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
    eval.checkpoint=${CKPT_PATH} \
    eval.batch_indices=[0,1,2,3] \
    eval.video_dir="${EVAL_DIR}/videos" \
    "${MODEL_PARAMS[@]}"
