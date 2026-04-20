# Eval scripts for the comparison study

Six submission scripts cover ambient and cached-latent checkpoints produced
under `outputs/2026-04-18/` and `outputs/2026-04-19/`. Each script iterates
`--dry-run` first, then submits for real.

| script | runs covered | eval.mode | eval.batch_size |
|---|---|---|---|
| `submit_eval_crps_ambient.sh` | `outputs/2026-04-18/crps_*` (4 primary + 2 CNS ablations) | default (auto → ambient) | 8 |
| `submit_eval_fm_ambient.sh` | `outputs/2026-04-18/diff_*` ambient (4 datasets) | default (auto → ambient) | 4 |
| `submit_eval_crps_latent.sh` | `outputs/2026-04-19/crps_*` cached-latent (CNS so far) | `ambient` | 8 |
| `submit_eval_fm_latent.sh` | `outputs/2026-04-18/diff_*` cached-latent (4 datasets) | `ambient` | 4 |
| `submit_eval_crps_latent_rollout_latent.sh` | same runs as `submit_eval_crps_latent.sh` | `latent` (writes to `eval_latent/`) | 8 |
| `submit_eval_fm_latent_rollout_latent.sh` | same runs as `submit_eval_fm_latent.sh` | `latent` (writes to `eval_latent/`) | 4 |

## Batch-size rationale

Empirically, the knobs are tight because eval rolls out with n_members=10
for 25 steps on 64×64 fields:

- **CRPS** (single forward per step) handles `eval.batch_size=8` fine.
- **FM / diffusion** integrates `flow_ode_steps=50` per rollout step, so
  ambient fits `eval.batch_size=4` — drop to 2 if OOM.
- **Cached-latent in ambient mode** still encodes/decodes at every step
  but the processor forward is cheaper (64 tokens vs 256 for
  ambient-patch4), so the CRPS variant matches ambient CRPS at 8 and the
  FM variant matches ambient FM at 4. Can try bumping up if there's
  headroom.
- **Cached-latent in latent mode** avoids per-step AE encode/decode and is
  typically cheaper. We keep 8 (CRPS) / 4 (FM) for consistency across
  comparisons; increase only after confirming cluster headroom.

## eval.mode for cached latents

The cached-latent scripts use the `eval.mode` selector that landed via
[PR #327](https://github.com/alan-turing-institute/autocast/pull/327) and is
now available in-tree. `eval.mode=ambient` forces full
`encoder → processor → decoder` rollout, so the decode/encode drift is
included in the metrics — the only fair regime for cross-comparison with
ambient CRPS/FM baselines that roll out in data space natively. Latent-only
rollout (`eval.mode=latent`) is faster and is useful as an additional
diagnostic view when written to a separate subdir (`eval_latent/`).

When `eval.mode=ambient` is set on a cached-latents datamodule, the eval
script auto-substitutes the raw datamodule from
`<cache_dir>/autoencoder_config.yaml`, and the AE weights are supplied via
`autoencoder_checkpoint=<ae.ckpt>` (hard-coded per run in each script).

## Submission order

These scripts are all independent — each run eval'd against its own
checkpoint. There are no branch prerequisites for the cached-latent scripts.

Dry-run everything first, review the printed sbatch commands, then re-run
without `RUN_DRY_STATES` edits to submit. Outputs land under each run's
`eval/` (ambient rollout) or `eval_latent/` (latent rollout) subdirectory
(`evaluation_metrics.csv`, rollout videos, etc.).
