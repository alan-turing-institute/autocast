# Cached-latent CRPS

CRPS loss trained in cached-latent space (processor-only training on
pre-encoded latents, decoded only at eval time).

**Status:** CNS data point exists —
`outputs/2026-04-19/crps_cns64_vit_azula_large_58712c4_71ba7be`.
No new training script needed for this pass; eval is handled by
`slurm_scripts/comparison/eval/submit_eval_crps_latent.sh`.

## Baseline

`local_hydra/local_experiment/processor/conditioned_navier_stokes/crps_vit_azula_large.yaml`.

## Next steps

- When the second dataset is added, extend the `DATASETS` map in
  `submit_eval_crps_latent.sh` and submit a matching training run via
  `slurm_scripts/comparison/cached_latents/submit_crps_latent_large.sh`.
- Decide whether to include `eval.mode=latent` ablation alongside
  `eval.mode=ambient` for this ablation specifically — it answers "how
  much of the latent-CRPS gap is decode/encode drift?".
