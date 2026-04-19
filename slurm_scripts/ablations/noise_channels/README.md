# Noise channels ablation

Sweep `model.processor.n_noise_channels` (the AdaLN-modulation noise
dimensionality) for the CRPS ambient baseline.

**Status:** stub — no scripts yet.

## Baseline

`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`
(baseline is `n_noise_channels=1024`).

## Knob

- `model.processor.n_noise_channels` — e.g. `{256, 1024, 4096}` or a
  finer grid. Values TBD.

## Datasets

CNS only for now.

## Outstanding decisions

- Exact sweep values. Existing config
  `epd_crps_vit_azula_n_noise_1024.yaml` and
  `epd_crps_vit_azula_noise4096_mod1024.yaml` hint at historical values.
- Whether to include the Concat-noise variant (see table notes
  "Concat vs with...") — currently red/skipped in the provisional table,
  so leave out.
