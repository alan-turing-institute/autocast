# Architecture comparison: U-Net, FNO, ViT

Compare U-Net and FNO backbones against the ViT (Azula) baseline on the
CRPS ambient path.

**Status:** stub — no scripts yet.

## Baseline

`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`
(ViT-Azula, ~81M params).

## Knob

Swap `model.processor` backbone while trying to match parameter count
(~80M) and per-epoch budget. Candidate configs to crib from:

- `local_hydra/local_experiment/epd_crps_unet_azula.yaml` — U-Net +
  CRPS.
- `local_hydra/local_experiment/epd_crps_fno.yaml` — FNO + CRPS.

Each will need per-CNS `local_experiment/ablations/arch/<arch>.yaml`
that matches the ambient baseline's encoder/decoder/loss so only the
backbone varies.

## Datasets

CNS only for now. Table says 2 datasets × 2 non-ViT archs = 4 runs
(CNS gives 2: U-Net and FNO).

## Outstanding decisions

- How to match parameter count across architectures — the comparison
  table for the main study (see `slurm_scripts/comparison/README.md`)
  locked ~80M for ViT variants; we need equivalent targets for U-Net
  and FNO.
- Whether FNO needs a different patch-size / token structure.
