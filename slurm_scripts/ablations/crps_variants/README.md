# CRPS loss variants

Compare `AlphaFairCRPS` (baseline) vs `FairCRPS` vs `CRPS`.

**Status:** FairCRPS and plain CRPS CNS configs added; AlphaFairCRPS is the
2026-04-24 CRPS baseline.

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

CNS only for now. The planned batch adds the two non-baseline loss variants:

- `local_hydra/local_experiment/ablations/crps_variants/conditioned_navier_stokes/crps_vit_fair.yaml`
- `local_hydra/local_experiment/ablations/crps_variants/conditioned_navier_stokes/crps_vit_plain.yaml`

## Implementation sketch

The cross-cutting submitter is
`slurm_scripts/ablations/submit_planned_cns_{timing,large}.sh`; it keeps the
loss-variant runs alongside the other planned CNS ablations.
