# FM vs diffusion

Compare flow matching (baseline) against DDPM/EDM-style diffusion on the
same ambient ViT backbone.

**Status:** CNS diffusion config added; timing is part of the planned CNS batch.

## Baseline

`local_hydra/local_experiment/epd/conditioned_navier_stokes/fm_vit_large.yaml`
(flow matching, 50 ODE steps, identity encoder).

## Knob

Swap `model.processor` from `flow_matching_vit` to `diffusion_vit` while
keeping the FM ambient backbone and conditioning path:

- identity encoder/decoder
- ViT backbone `hid_channels=704`, `hid_blocks=12`, `attention_heads=8`,
  `patch_size=4`
- `datamodule.batch_size=256`
- AdamW-half LR `1e-4`
- 50 Euler sampler steps, matching FM's `flow_ode_steps=50` as the closest
  equal-NFE comparison

Config:
`local_hydra/local_experiment/ablations/fm_vs_diffusion/conditioned_navier_stokes/diffusion_vit_large.yaml`.

## Datasets

CNS only for now. Table says 2 datasets × 1 comparison = 2 runs (CNS
gives 1).

## Outstanding decisions

- Whether a second diffusion eval should also report a higher-step sampler for
  quality, separate from the equal-50-step fairness run.
