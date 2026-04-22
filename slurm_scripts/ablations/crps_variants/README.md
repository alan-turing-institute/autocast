# CRPS loss variants

Compare `AlphaFairCRPS` (baseline) vs `FairCRPS` vs `CRPS`.

**Status:** stub — no scripts yet.

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

Exact class paths to be verified against
`src/autocast/losses/ensemble.py` and `metrics/ensemble.py` before
scripting.

## Datasets

CNS only for now. Table spec'd 2 datasets × 3 losses = 6 runs — CNS
gives us 3 runs for this pass.

## Implementation sketch

Single-file sweep via CLI overrides in `submit_crps_variants_*.sh` with
a `LOSSES` array of `(name, loss_target, metric_target)` triples.
