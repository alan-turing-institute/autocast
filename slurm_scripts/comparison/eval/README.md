# Eval scripts for the comparison study

This directory is for the canonical comparison-suite evals. When an eval
submitter only targets a study-specific ablation run set, keep it under
`slurm_scripts/ablations/<name>/eval/` until that run set is promoted into the
main comparison.

Four submission scripts cover ambient and cached-latent checkpoints produced
under `outputs/2026-04-18/` and `outputs/2026-04-20/`. Each script iterates
`--dry-run` first, then submits for real.

All comparison eval submitters explicitly pass `eval.n_members=10` for now so
comparison numbers do not silently drift if the global eval default changes.

| script | runs covered | eval.mode | eval.batch_size |
|---|---|---|---|
| `submit_eval_crps_ambient.sh` | `outputs/2026-04-18/crps_*` (4 primary + 2 CNS ablations) | default (auto → ambient) | 8 |
| `submit_eval_fm_ambient.sh` | `outputs/2026-04-18/diff_*` ambient (4 datasets) | default (auto → ambient) | 4 |
| `submit_eval_crps_latent.sh` | `outputs/2026-04-20/crps_*` cached-latent (CNS so far) | default (`auto -> encode_once`) | 8 |
| `submit_eval_fm_latent.sh` | `outputs/2026-04-20/diff_*` cached-latent (4 datasets) | default (`auto -> encode_once`) | 4 |

## Batch-size rationale

Empirically, the knobs are tight because eval rolls out with n_members=10
for 25 steps on 64×64 fields:

- **CRPS** (single forward per step) handles `eval.batch_size=8` fine.
- **FM / diffusion** integrates `flow_ode_steps=50` per rollout step, so
  ambient fits `eval.batch_size=4` — drop to 2 if OOM.
- **Cached-latent via `auto -> encode_once`** encodes once up front,
  decodes per step, and scores against raw ground truth. It is cheaper
  than the ambient ablation while still being faithful for processor-only
  evaluation, so the CRPS variant stays at 8 and the FM variant stays at 4
  for easy comparison with the ambient scripts.

## eval.mode for cached latents

The cached-latent comparison scripts now rely on the default
`eval.mode=auto`, which resolves to `encode_once` for processor-only
cached-latent runs when `autoencoder_checkpoint=<ae.ckpt>` is supplied.
That behavior landed in
[PR #339](https://github.com/alan-turing-institute/autocast/pull/339).
It keeps metrics in raw data space while avoiding the extra decode/encode
drift charged by the explicit ambient ablation. That is now the only
comparison-suite path we keep under `slurm_scripts/comparison/eval/`.

## Submission order

These scripts are all independent — each run eval'd against its own
checkpoint. There are no branch prerequisites for the cached-latent scripts.

Dry-run everything first, review the printed sbatch commands, then re-run
without `RUN_DRY_STATES` edits to submit. Outputs land under each run's
`eval/` subdirectory (`evaluation_metrics.csv`, rollout videos, etc.).
