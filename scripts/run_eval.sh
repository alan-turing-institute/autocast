#!/bin/bash

set -euo pipefail

export AUTOCAST_DATASETS="${AUTOCAST_DATASETS:-$PWD/datasets}"

OUTPUTS=(
  "outputs/2026-02-12/crps_adm64_fno_concat_256_416c86d_1d4d1e6/"
  "outputs/2026-02-17/crps_adm64_vit_large_cln_128_7c9f3ba_74275e8/"
  "outputs/2026-02-17/crps_adm64_vit_large_cln_128_7c9f3ba_dc7c6cc/"
  "outputs/2026-02-18/crps_adm64_vit_latent_cln_64_d5ebd6f_7fc8bda/"
  "outputs/2026-02-12/diff_adm64_diffusion_vit_512_a5ad325_db3f170/"
  "outputs/2026-02-12/diff_adm64_flow_matching_vit_512_a5ad325_4e14e6f/"
  "outputs/2026-02-19/crps_adm64_fno_concat_128_0025526_2b520bf/"
  "outputs/2026-02-19/crps_adm64_vit_latent_cln_768_5100bb5_0d57d75/"
  "outputs/2026-02-20/crps_adm64_vit_latent_132450d_5c44ec9/"
)

EVAL_BATCH_SIZE=4
for output in "${OUTPUTS[@]}"; do
  echo
  echo "uv run autocast eval --mode slurm --workdir ${output} eval.batch_indices=[0,1,2,3,4,5,6,7] datamodule.batch_size=${EVAL_BATCH_SIZE} hydra.launcher.timeout_min=180"
  uv run autocast eval --mode slurm --workdir "${output}" eval.batch_indices=[0,1,2,3,4,5,6,7] datamodule.batch_size=${EVAL_BATCH_SIZE} hydra.launcher.timeout_min=180
done
