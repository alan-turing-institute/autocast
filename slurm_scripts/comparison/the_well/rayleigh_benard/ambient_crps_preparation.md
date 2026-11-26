# Prepare ambient Rayleigh-Benard experiments

The matched preset uses a width-1024 transformer (226M parameters) on native
512-by-128 fields; `--size lola` selects the width-2048, 859M extension.
Both retain 4096 epochs, 64 updates per epoch and the full cosine schedule.
CRPS uses 32 real windows with eight members per update. Matching latent
update counts does not match real-window exposure or compute.

Training presets under `local_hydra/local_experiment/the_well/rayleigh_benard/`:

| Model | Preset |
| --- | --- |
| Ambient CRPS 226M | [crps_spatiotemporal_ambient_matched.yaml](../../../../local_hydra/local_experiment/the_well/rayleigh_benard/crps_spatiotemporal_ambient_matched.yaml) |
| Ambient CRPS 859M | [crps_spatiotemporal_ambient.yaml](../../../../local_hydra/local_experiment/the_well/rayleigh_benard/crps_spatiotemporal_ambient.yaml) |
| Ambient CRPS gathered graph | [crps_vit_azula_graph_ambient.yaml](../../../../local_hydra/local_experiment/the_well/rayleigh_benard/crps_vit_azula_graph_ambient.yaml) |
| Latent CRPS 223M | [crps_spatiotemporal_latent.yaml](../../../../local_hydra/local_experiment/the_well/rayleigh_benard/crps_spatiotemporal_latent.yaml) |
| Latent FM 223M | [fm_spatiotemporal_latent.yaml](../../../../local_hydra/local_experiment/the_well/rayleigh_benard/fm_spatiotemporal_latent.yaml) |

## Prepare training

From a clean, committed checkout with the existing uv environment:

```bash
uv run --frozen --no-sync python -m slurm_scripts.comparison.the_well.rayleigh_benard.prepare_ambient_crps training \
  --phase train --size matched \
  --datasets-root /path/to/datasets --output /path/to/new_training
```

Use `--size lola` for the larger model, or `--phase smoke` / `--phase timing`
for deployment checks. `--nodes` supports 1, 2, 4 or 8; `--microbatch`
adjusts accumulation while preserving the global batch and update count.
Preparation writes resolved configs, source/config hashes and batch scripts
to a fresh directory. Inspect those artifacts before explicit submission.

Use `--size graph` for the ambient Azula ViT with 4×4 patches, two-hop
gathered attention and horizontal wrapping. It uses one observed frame to
predict the next four jointly and defaults to one real window per GPU per
microbatch. It retains eight draws per window and the same update count.
Training uses BF16 mixed precision and compiles the spatial ViT backbone.
The smoke job allows 30 minutes for compilation and one optimizer update.
The preset disables Azula's relative position bias because it measures
distances in unwrapped coordinates across the periodic seam.

## Continue a run

AutoCast already provides full-state resume through
`resume_from_checkpoint`, `resume_weights_only=false` and
`reset_resume_time_budget=true`. Retain the total epoch/cosine horizons,
W&B ID and checkpoint-selection state across allocations.

The optional `rb_manual.py` controller uses that existing training path,
adding immutable checkpoint checks, per-rank restore verification and
allocation staging. It pins scientific code to `4b2c0499`, retained in this
branch's history; prepared artifacts also record the controller hash.

- `prepare-first` accepts a JSON source record with `checkout`, `head`,
  `git_status`, `config` and `source_config_sha256`.
- `seal` verifies a completed epoch-boundary checkpoint.
- `prepare-resume` prepares a continuation from the sealed checkpoint.
- `prepare-chain` prepares a bounded dependent chain for the larger model.

Use the controller's `--help` for paths and allocation options.
Its CLI preserves the original two-allocation matched campaign and the
four-allocation larger-model chain; additional continuations use the existing
training resume interface.
Preparation does not submit jobs. Running the generated `queue.sh ACCOUNT
PARTITION` submits a chain; later allocations skip training after a verified
262,144-update endpoint.

## Prepare archived evaluation jobs

```bash
uv run --frozen --no-sync python -m slurm_scripts.comparison.the_well.rayleigh_benard.prepare_ambient_crps evaluation \
  --config /path/to/run/resolved_config.yaml \
  --checkpoint /path/to/run/selected.ckpt --expected-step N \
  --output /path/to/new_evaluation
```

This prepares paired-window and start16 rollout jobs with 16 members,
the saved 64-trajectory cohort and 512 windows. Keep each model's
normalization and codec. See [diagnostics](diagnostics.md) for same-pass
energy scores, trajectory uncertainty and inference latency.
