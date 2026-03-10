#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 1:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --job-name eval_masked_epd

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/.conda/envs/autocast/bin/python -m autocast.scripts.eval.encoder_processor_decoder \
  --config-dir outputs/seaice/epd_flow_pixels_in2_out1_masked__selectedyears/2026-02-13_15-12-29 \
  --config-name resolved_config \
  hydra.run.dir=outputs/seaice/epd_flow_pixels_in2_out1_masked__selectedyears/2026-02-13_15-12-29/eval_run \
  datamodule.data_path=/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_selectedyears \
  eval.checkpoint=/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast/outputs/seaice/epd_flow_pixels_in2_out1_masked__selectedyears/2026-02-13_15-12-29/encoder_processor_decoder.ckpt \
  eval.free_running_only=true \
  eval.batch_indices=[0] \
  eval.video_dir=outputs/seaice/epd_flow_pixels_in2_out1_masked__selectedyears/2026-02-13_15-12-29/eval_videos \
  eval.video_format=mp4 \
  eval.fps=5 \
  eval.device=cuda \
  +model.processor.mask_path=/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf/land_mask.pt
