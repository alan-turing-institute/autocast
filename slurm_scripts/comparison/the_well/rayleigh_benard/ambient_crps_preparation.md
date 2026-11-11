# Large masked ambient CRPS preparation

The existing `crps_spatiotemporal_ambient` preset is the large ambient
counterpart to the completed `fm_spatiotemporal_latent` and
`crps_spatiotemporal_latent` experiments. Reuse those latent models and their
frozen C64 autoencoder/cache. This preparation adds no latent training runs.

The ambient model uses normalized 512×128 fields with four channels, identity
encoder/decoder, a width-2048 transformer with 16 blocks/16 heads and
`[1,16,16]` patches. Five frames form 1,280 joint spatiotemporal tokens. The
shared prefix-mask policy and unreduced alpha-fair CRPS loss are unchanged.
`train_in_latent_space: true` delegates loss to the processor; the identity
codec keeps its inputs and targets in ambient space.

## Prepare locally

The preparation module writes resolved configs, provenance and batch scripts.
It never invokes a scheduler, trains a model or loads dataset/checkpoint files.
Use a committed checkout and a fresh absolute output directory. Dataset paths
can refer to their eventual location on the execution machine:

```bash
uv run --frozen --no-sync python -m slurm_scripts.comparison.the_well.rayleigh_benard.prepare_ambient_crps training \
  --phase smoke --datasets-root /path/to/datasets --output /path/to/new_smoke

uv run --frozen --no-sync python -m slurm_scripts.comparison.the_well.rayleigh_benard.prepare_ambient_crps training \
  --phase timing --datasets-root /path/to/datasets --output /path/to/new_timing
```

| Profile | Epochs | Updates/epoch | Allocation | Cosine horizon |
| --- | ---: | ---: | ---: | ---: |
| Smoke | 1 | 1 | 30 min | 4096 epochs |
| Timing | 16 by default | 64 | 4 h | 4096 epochs |
| Train | Explicit measured horizon | 64 | 24 h | Same selected horizon |

All request one node and four GPUs/tasks. The default microbatch is one real
window per GPU with accumulation eight. Together with eight training members,
this preserves 32 real windows and 256 member-window predictions per optimizer
update. `--microbatch` accepts 1, 2, 4 or 8; accumulation adjusts accordingly.
Training and validation batch limits scale with accumulation, retaining 64
optimizer updates per sampling epoch and 512 real validation windows per check.
Smoke uses only one validation batch. Validation runs every eight epochs during
timing/training; timing epochs must be a multiple of eight. No downsampling or
reduced-width model is used by these prepared GPU jobs.

Check actual loader lengths and full-width memory use before timing. Accumulation
preserves exposure counts but changes execution and logging overhead. Time the
microbatch that production will use. Both validation cycles and final checkpoint
writing must complete before treating the default 16-epoch probe as evidence.
Timing records are read with the existing workflow:

```bash
uv run --frozen --no-sync autocast time-epochs \
  --from-checkpoint /path/to/new_timing/run/timing.ckpt -b 24 -m 0.05
```

An alternative total budget can be assessed with `-b`, after deciding the
scientific comparison. Neither 24 nor 48 training hours are inferred from the
old ambient models or reheated continuation results. Once selected, prepare
production with `--phase train --epochs N`; omitting `--epochs` is an error.
There is no production placeholder or automatic job submission.

Production uses an epoch-boundary Timer with 30 minutes reserved in each
24-hour allocation. Verify the measured worst epoch, validation and final save
fit that headroom. A longer logical schedule needs separately reviewed manual
continuations: restore full model/optimizer/scheduler/EMA state from this run's
completed-epoch checkpoint, retain its total epoch/cosine horizons and W&B ID,
and use `reset_resume_time_budget=true` for a fresh allocation Timer. Cumulative
training-time diagnostics remain available separately. Both Slurm and Lightning
automatic requeue are disabled. Initial training scripts reject existing run
directories, including partial attempts without a checkpoint. Use a fresh output
directory for another probe; production continuations require full-state resume.

## Matched evaluation preparation

The fixed-cohort and paired-window tools reproduce the completed comparison.
AB3/16 remains the sampling configuration of the completed latent FM reference.
For a completed ambient model, supply its saved resolved config, selected
checkpoint and recorded optimizer step:

```bash
uv run --frozen --no-sync python -m slurm_scripts.comparison.the_well.rayleigh_benard.prepare_ambient_crps evaluation \
  --config /path/to/run/resolved_config.yaml \
  --checkpoint /path/to/run/selected.ckpt --expected-step N \
  --output /path/to/new_evaluation
```

This writes two one-GPU scripts. Rollout uses the frozen 64-trajectory seed-0
selection, 16 members, conditioning frame 16 and all 183 available forecast
frames (absolute frames 17–199). Paired evaluation uses the same 512 windows:
eight saved contexts per trajectory, one input and four future frames. Both
score physical-field targets with the original raw normalization, raw weights
and sampling seed 42; neither evaluates autoencoded targets. The selected step
is checked at execution. New output paths preserve existing results.

The generic full-test single-step pass is disabled in the rollout job; the
paired job supplies the controlled next-window comparison. Both inherited test
and rollout batch limits are cleared so all 64 trajectories are evaluated.
The raw The Well loader does not expose generic trajectory-statistics metadata,
so these jobs use the established RB cohort/paired tools instead of that option.
Judge spatial spectra and matched videos alongside VRMSE, spread/skill and
coverage, retaining the distinction between mean and median statistics.

Regenerate all prepared artifacts from the final checkout on the execution
machine. Scripts verify the code commit and resolved config checksum, require
a Slurm allocation and reject existing run/evaluation directories. Inspect the
generated scripts before submission.
