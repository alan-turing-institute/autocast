# Conditioning: global_cond (AdaLN) vs permute_concat

Swap the CRPS ambient conditioning path from `permute_concat` (spatial
channel concatenation) to `identity` encoder + `include_global_cond:
true` (AdaLN modulation on the backbone). Makes conditioning flow match
FM ambient, isolating the encoder effect.

**Status:** CNS data point exists for CRPS-ViT —
`outputs/2026-04-18/crps_cns64_vit_azula_large_0f89f06_cf53b48`. No new
CRPS-ViT training needed for this pass; U-Net equivalent is pending.

## Baselines

- CRPS-ViT with identity+global_cond:
  `local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large_identity_global_cond.yaml`.
- CRPS-ViT with permute_concat (main baseline):
  `.../crps_vit_azula_large.yaml`.

## Outstanding

- U-Net analogue: need `crps_unet_large_identity_global_cond.yaml`
  mirroring the ViT ablation. U-Net backbone `include_global_cond` path
  to be verified against
  `src/autocast/processors/` U-Net module.
- Eval for the existing CNS ViT ablation run is covered by
  `slurm_scripts/comparison/eval/submit_eval_crps_ambient.sh` (included
  in its RUN_DIRS).
