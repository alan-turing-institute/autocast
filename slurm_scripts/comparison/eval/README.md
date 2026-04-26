# Eval scripts for the comparison study

This directory is for the canonical comparison-suite evals. When an eval
submitter only targets a study-specific ablation run set, keep it under
`slurm_scripts/ablations/<name>/eval/` until that run set is promoted into the
main comparison.

The canonical basis is now:

- CRPS ambient: latest 2026-04-24 EPD runs, with the optional
  best-multi-Winkler-from-0.25 submitter for the current preferred checkpoint
  selection.
- FM/diff: 2026-04-20 cached-latent `diff_*` runs. The default latent eval uses
  `auto -> encode_once`; the ambient eval explicitly supplies the matching AE
  checkpoint and writes to `eval_ambient/`.

Each script iterates `--dry-run` first, then submits for real.

All comparison eval submitters explicitly pass `eval.n_members=10` for now so
comparison numbers do not silently drift if the global eval default changes.

| script | runs covered | eval.mode | eval.batch_size |
|---|---|---|---|
| `submit_eval_crps_ambient.sh` | `outputs/2026-04-24/crps_*` primary final checkpoints | explicit `ambient` | 8 |
| `submit_eval_crps_ambient_best_multiwinkler_from0p25.sh` | same 2026-04-24 CRPS runs, best multi-Winkler after 25% progress | explicit `ambient` | 8 |
| `submit_eval_fm_ambient.sh` | `outputs/2026-04-20/diff_*` cached-latent FM basis, final checkpoints | explicit `ambient` | 4 |
| `submit_eval_crps_latent.sh` | `outputs/2026-04-20/crps_*` cached-latent (CNS so far) | default (`auto -> encode_once`) | 8 |
| `submit_eval_fm_latent.sh` | `outputs/2026-04-20/diff_*` cached-latent (4 datasets) | default (`auto -> encode_once`) | 4 |
| `submit_eval_fm_latent_0p25.sh` | same 2026-04-20 FM cached-latent runs at 25% progress | explicit `auto -> encode_once` | 4 |
| `submit_eval_fm_latent_0p50.sh` | same 2026-04-20 FM cached-latent runs at 50% progress | explicit `auto -> encode_once` | 4 |
| `submit_eval_fm_latent_0p75.sh` | same 2026-04-20 FM cached-latent runs at 75% progress | explicit `auto -> encode_once` | 4 |
| `submit_eval_fm_ambient_0p25.sh` | same 2026-04-20 FM cached-latent runs at 25% progress | explicit `ambient` | 4 |
| `submit_eval_fm_ambient_0p50.sh` | same 2026-04-20 FM cached-latent runs at 50% progress | explicit `ambient` | 4 |
| `submit_eval_fm_ambient_0p75.sh` | same 2026-04-20 FM cached-latent runs at 75% progress | explicit `ambient` | 4 |

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

The CRPS best-checkpoint submitter prefers
`best-multiwinkler-from0p25-*.ckpt` and writes to
`eval_best_multiwinkler_from0p25/`.

The FM progress-checkpoint submitters prefer `snapshot-0p25-*.ckpt`,
`snapshot-0p50-*.ckpt`, or `snapshot-0p75-*.ckpt` when present, and fall
back to the legacy first, second, or third sorted `quarter-*.ckpt` checkpoint
saved by the 2026-04-20 runs. The default cached-latent progress evals write
to `eval_0p25/`, `eval_0p50/`, and `eval_0p75/`. The explicit ambient
variants write to `eval_0p25_ambient/`, `eval_0p50_ambient/`, and
`eval_0p75_ambient/` so they do not overwrite the `auto -> encode_once`
outputs.
