# Matched masked ambient CRPS preparation

The `crps_spatiotemporal_ambient_matched` preset matches the transformer trunk
of the completed `fm_spatiotemporal_latent` and `crps_spatiotemporal_latent`
experiments. Reuse those latent models and their frozen C64 autoencoder/cache.
This preparation adds no latent training runs.

The ambient model uses normalized 512×128 fields with four channels, identity
encoder/decoder, a width-1024 transformer with 16 blocks/eight heads and
`[1,16,16]` patches. Five frames form 1,280 joint spatiotemporal tokens. The
shared prefix-mask policy and unreduced alpha-fair CRPS loss are unchanged.
`train_in_latent_space: true` delegates loss to the processor; the identity
codec keeps its inputs and targets in ambient space.

The ambient processor has 225,969,408 parameters versus 222,888,512 for latent
FM and 223,019,328 for latent CRPS. Ambient patch projections account for the
small difference. These counts exclude the frozen latent autoencoder; report
that component separately. Use `--size lola` to prepare the original width-2048,
16-block/16-head ambient preset (859,499,776 parameters) as a separate capacity
extension; it matches LOLA's ambient backbone geometry while retaining our
masked CRPS objective and noise adapter. The default is `--size matched`.

Production retains **4096 epochs × 64 updates = 262,144 optimizer updates**
and the full 4096-epoch cosine horizon across all allocations. This matches
the latent update schedule, not elapsed time or GPU-hours. CRPS uses 32 real
windows × eight members per update; latent FM uses 256 real windows × one
member, so their real-example exposure differs.

On eight nodes, the matched run projects 253–264 node-hours and the LOLA-size
run roughly 601 training-only node-hours, assuming the matched pilot's 4.94×
speedup (full-size multi-node scaling is unmeasured).

## Prepare locally

The preparation module writes resolved configs, provenance and batch scripts.
It never invokes a scheduler, trains a model or loads dataset/checkpoint files.
Use a committed checkout and a fresh absolute output directory. Dataset paths
can refer to their eventual location on the execution machine:

```bash
uv run --frozen --no-sync python -m slurm_scripts.comparison.the_well.rayleigh_benard.prepare_ambient_crps training \
  --phase train --datasets-root /path/to/datasets --output /path/to/new_training

uv run --frozen --no-sync python -m slurm_scripts.comparison.the_well.rayleigh_benard.prepare_ambient_crps training \
  --phase train --size lola --datasets-root /path/to/datasets --output /path/to/new_lola_training
```

Optional deployment checks can be prepared separately:

```bash
uv run --frozen --no-sync python -m slurm_scripts.comparison.the_well.rayleigh_benard.prepare_ambient_crps training \
  --phase smoke --datasets-root /path/to/datasets --output /path/to/new_smoke
uv run --frozen --no-sync python -m slurm_scripts.comparison.the_well.rayleigh_benard.prepare_ambient_crps training \
  --phase timing --datasets-root /path/to/datasets --output /path/to/new_timing
```

| Profile | Epochs | Updates/epoch | Allocation | Cosine horizon |
| --- | ---: | ---: | ---: | ---: |
| Smoke | 1 | 1 | 15 min | 4096 epochs |
| Timing | 16 by default | 64 | 30 min on eight nodes | 4096 epochs |
| Train | 4096 total | 64 | 24 h per allocation | 4096 epochs |

The default is eight nodes, four GPUs/tasks per node, one real window per GPU
and no accumulation: 32 real windows and 256 member-window predictions per
optimizer update. `--nodes` also accepts 1, 2 or 4, with default per-GPU batches
8, 4 or 2 respectively. An explicit smaller `--microbatch` increases accumulation
to preserve the global batch. Larger batches that change exposure are rejected.
Multi-node scripts load the existing Isambard NCCL/OFI module and record transport
logs; verify Libfabric/CXI/GDRDMA in those logs when deploying a new runtime.

Training and validation batch limits scale with accumulation, retaining 64
optimizer updates per epoch and 512 real validation windows per check. Smoke
uses one validation batch. Validation runs every eight epochs during timing and
production; timing epochs must be a multiple of eight. This matches latent CRPS's
validation exposure/cadence; latent FM validates on more real windows. Inputs
retain native resolution and the five-frame masked spatiotemporal window.

Use the existing matching-model memory, throughput and resume measurements
where the runtime and accumulation are unchanged. Further probes can check a
new deployment; they do not select a shorter production schedule. Production
always uses 4096 total epochs, and a conflicting `--epochs` override is rejected.
Timing on fewer than eight nodes retains a four-hour allocation cap.

Production uses an epoch-boundary Timer with 30 minutes reserved in each
24-hour allocation. Check epoch, validation and save times against that headroom.
The complete schedule may need multiple allocations, using manual
continuations: restore full model/optimizer/scheduler/EMA state from this run's
completed-epoch checkpoint, retain its total 4096-epoch/cosine horizons and W&B ID,
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
