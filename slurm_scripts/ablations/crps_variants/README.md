# CRPS loss variants

Compare `AlphaFairCRPS` (baseline) vs `FairCRPS` vs `CRPS`.

**Status:** FairCRPS and plain CRPS configs exist for CNS, GS, GPE, and AD.
AlphaFairCRPS is the 2026-04-24 CRPS baseline.

## Baseline

`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`
(uses `AlphaFairCRPSLoss`).

## Knob

Swap `model.loss_func._target_` and the matching `train_metrics.crps`
target:

| variant | loss_func | metric |
|---|---|---|
| AlphaFairCRPS (baseline) | `autocast.losses.ensemble.AlphaFairCRPSLoss` | `autocast.metrics.ensemble.AlphaFairCRPS` |
| FairCRPS | `autocast.losses.ensemble.FairCRPSLoss` | `autocast.metrics.ensemble.FairCRPS` |
| CRPS | `autocast.losses.ensemble.CRPSLoss` | `autocast.metrics.ensemble.CRPS` |

Class paths were verified against `src/autocast/losses/ensemble.py` and
`src/autocast/metrics/ensemble.py`.

## Datasets

Planned batch 01 adds the two non-baseline CNS loss variants:

- `local_hydra/local_experiment/ablations/crps_variants/conditioned_navier_stokes/crps_vit_fair.yaml`
- `local_hydra/local_experiment/ablations/crps_variants/conditioned_navier_stokes/crps_vit_plain.yaml`

Planned batch 04 extends the FairCRPS m=8 ViT run to the other raw-field
EPD datasets:

- `local_hydra/local_experiment/ablations/crps_variants/gray_scott/crps_vit_fair.yaml`
- `local_hydra/local_experiment/ablations/crps_variants/gpe_laser_wake_only/crps_vit_fair.yaml`
- `local_hydra/local_experiment/ablations/crps_variants/advection_diffusion/crps_vit_fair.yaml`

Planned batch 05 mirrors planned batch 04 with plain CRPS loss:

- `local_hydra/local_experiment/ablations/crps_variants/gray_scott/crps_vit_plain.yaml`
- `local_hydra/local_experiment/ablations/crps_variants/gpe_laser_wake_only/crps_vit_plain.yaml`
- `local_hydra/local_experiment/ablations/crps_variants/advection_diffusion/crps_vit_plain.yaml`

## Implementation sketch

The CNS cross-cutting submitter is
`slurm_scripts/ablations/submit_planned_01_{timing,large}.sh`; it keeps the
loss-variant runs alongside the other planned CNS ablations. The cross-dataset
FairCRPS submitter is
`slurm_scripts/ablations/submit_planned_04_{timing,large}.sh`, with eval in
`slurm_scripts/ablations/submit_eval_planned_04.sh`. The cross-dataset plain
CRPS submitter is `slurm_scripts/ablations/submit_planned_05_{timing,large}.sh`,
with eval staging in `slurm_scripts/ablations/submit_eval_planned_05.sh`.
