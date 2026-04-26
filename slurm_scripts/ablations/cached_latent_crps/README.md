# Cached-latent CRPS

CRPS loss trained in cached-latent space (processor-only training on
pre-encoded latents, decoded only at eval time).

**Status:** CNS data point exists as the current latent CRPS basis —
`outputs/2026-04-20/crps_cns64_vit_azula_large_09490da_8b7573d`.
No new training script needed for this pass; comparison eval is handled by
`slurm_scripts/comparison/eval/submit_eval_crps_latent.sh` via the default
`auto -> encode_once` path.

## Baseline

`local_hydra/local_experiment/processor/conditioned_navier_stokes/crps_vit_azula_large.yaml`.

## Next steps

- When the second dataset is added, extend the `RUN_DIRS` list and `AE_CKPT`
  map in `submit_eval_crps_latent.sh` and submit a matching training run via
  `slurm_scripts/comparison/cached_latents/submit_crps_latent_large.sh`.
- If we ever need a latent-only diagnostic again, use the eval CLI directly
  with explicit overrides rather than keeping a dedicated comparison submitter.
