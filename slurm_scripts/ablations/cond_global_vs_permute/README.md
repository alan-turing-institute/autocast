# Conditioning: global_cond (AdaLN) vs permute_concat

Swap the CRPS ambient conditioning path from `permute_concat` (spatial
channel concatenation) to `identity` encoder + `include_global_cond:
true` (AdaLN modulation on the backbone). Makes conditioning flow match
FM ambient, isolating the encoder effect.

**Status:** CNS config exists and is included in the planned CNS batch. An
older data point exists at
`outputs/2026-04-18/crps_cns64_vit_azula_large_0f89f06_cf53b48`, but the
current comparison basis uses 2026-04-24 CRPS runs, so rerun this ablation if
date-aligned comparison is required.

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
- Eval for a rerun should live under this ablation until it is promoted into
  the main comparison eval set.
