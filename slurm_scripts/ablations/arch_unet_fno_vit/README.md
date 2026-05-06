# Architecture comparison: U-Net, FNO, ViT

Compare U-Net and FNO backbones against the ViT (Azula) baseline on the
CRPS ambient path.

**Status:** U-Net CNS config added; FNO remains unscheduled.

## Baseline

`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`
(ViT-Azula, ~81M params).

## Knob

Swap `model.processor` backbone while trying to match parameter count
(~80M) and per-epoch budget. Candidate configs to crib from:

- `local_hydra/local_experiment/epd_crps_unet_azula.yaml` — U-Net +
  CRPS.
- `local_hydra/local_experiment/epd_crps_fno.yaml` — FNO + CRPS.

The planned U-Net run uses
`local_hydra/local_experiment/ablations/arch_unet_fno_vit/conditioned_navier_stokes/crps_unet_azula_80m.yaml`.
It matches the ambient baseline's encoder/decoder/loss and uses an Azula U-Net
channel ladder `[47, 94, 188, 376]`, measured at ~80.9M processor params for
CNS ambient shapes.

FNO still needs a matching per-CNS config before scheduling.

## Datasets

CNS only for now. Current planned coverage is U-Net only; FNO is held back
until the parameter-matching decision is settled.

## Outstanding decisions

- How to match FNO parameter count — the U-Net target is now fixed at ~80.9M
  to match the 80M ViT variants.
- Whether FNO needs a different patch-size / token structure.
