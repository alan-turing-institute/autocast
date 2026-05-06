# Noise channels ablation

Sweep `model.processor.n_noise_channels` (the AdaLN-modulation noise
dimensionality) for the CRPS ambient baseline.

**Status:** CNS `n_noise_channels=256` config added; timing is part of the
planned CNS batch.

## Baseline

`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`
(baseline is `n_noise_channels=1024`).

## Knob

- `model.processor.n_noise_channels=256`, compared against the 1024-channel
  2026-04-24 CRPS baseline. To keep the processor near 80M params, this run
  holds depth/heads fixed and uses `hidden_dim=704` as the single balancing
  knob (~79.9M params for CNS ambient shapes).

Config:
`local_hydra/local_experiment/ablations/noise_channels/conditioned_navier_stokes/crps_vit_noise256.yaml`.

## Datasets

CNS only for now.

## Outstanding decisions

- Whether to add a wider value such as 4096 after the 256-vs-1024 contrast.
- Whether to include the Concat-noise variant (see table notes
  "Concat vs with...") — currently red/skipped in the provisional table,
  so leave out.
