# Model size ablation

Run a smaller ViT (and possibly a larger one) against the ~80M baseline.

**Status:** stub — no scripts yet.

## Baseline

`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`
— hidden_dim=568, n_layers=12, num_heads=8.

## Knob

Scale `hidden_dim` and optionally `n_layers` to hit a target param count
(e.g. ~20M small, ~80M baseline, ~200M large). DiT-canonical proportions
suggest keeping head_dim≈64, so adjust `hidden_dim` in steps of 64×heads.

## Datasets

CNS only for now. Table says 2 datasets × 1 smaller variant = 2 runs
(CNS gives 1).

## Outstanding decisions

- Exact param-count targets (small / large).
- Whether to also try `vit_azula_4gpu_632` or `..._768` (configs exist
  but were used for the 4-dataset comparison earlier at different
  sizes).
