# FM vs diffusion

Compare flow matching (baseline) against DDPM/EDM-style diffusion on the
same ambient ViT backbone.

**Status:** stub — no scripts yet.

## Baseline

`local_hydra/local_experiment/epd/conditioned_navier_stokes/fm_vit_large.yaml`
(flow matching, 50 ODE steps, identity encoder).

## Knob

Swap `model.processor` from `flow_matching_vit` to the diffusion
equivalent. Existing configs to crib from:

- `local_hydra/local_experiment/epd_diffusion_dm_256_dc_large.yaml`
  (DDPM-style with DC large AE — note: ambient baseline in the ablation
  table uses identity encoder, not DC AE, so we need a matching
  `epd_diffusion_dm_256_identity.yaml`-style config.)
- `local_hydra/local_experiment/epd_diffusion_fm_256_identity.yaml` —
  FM ambient equivalent.

## Datasets

CNS only for now. Table says 2 datasets × 1 comparison = 2 runs (CNS
gives 1).

## Outstanding decisions

- DDPM vs EDM vs something else. Table just says "diffusion" — need to
  pick a specific processor.
- Sampler / step count for the diffusion side (FM uses 50 ODE steps;
  diffusion would typically use more).
- Whether the comparison should hold n_function_evaluations fixed at
  inference for fairness.
