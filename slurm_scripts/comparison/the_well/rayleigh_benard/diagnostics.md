# Reproduce the frozen-checkpoint RB comparison

`complete_diagnostics.py` evaluates the models in `diagnostics_models.yaml`:
three completed 226M/223M checkpoints and the exploratory 859M midpoint.
Checkpoint and codec weights remain external; manifests pin their hashes.

This runner preserves the archived comparison's protocol. Shared reference
evaluation ([PR #489](https://github.com/alan-turing-institute/autocast/pull/489))
and an RB reference preset are follow-on work, including re-evaluation of these
checkpoints. Report those as new reference evaluations when sampling,
aggregation or score definitions change.

## Prepare and evaluate

Supply the saved resolved configs and window selections at the recipe's paths
under `--outputs-root`, together with the raw The Well data, fitted normalization
files and model/codec checkpoints referenced by those configs.

From a clean checkout with `uv sync --frozen --extra dev` completed:

```bash
uv run --frozen python -m slurm_scripts.comparison.the_well.rayleigh_benard.complete_diagnostics prepare \
  --outputs-root "$RB_OUTPUTS_ROOT" \
  --output "$RB_DIAGNOSTICS_DIR" --account "$RB_SLURM_ACCOUNT"
```

Preparation writes frozen configs, normalization statistics, cohort/window
selections and pilot/production scripts. It refuses existing destinations
and never submits jobs. `--raw-data-root` and `--autoencoder-root` relocate
data and codec files without changing normalization.

Inspect and explicitly submit a pilot, then its corresponding production job:

```bash
sbatch "$RB_DIAGNOSTICS_DIR/ambient_matched-pilot.sbatch"
# After the pilot succeeds:
sbatch "$RB_DIAGNOSTICS_DIR/ambient_matched-production.sbatch"
```

Repeat for the other models. Production requires a successful pilot with
the identical manifest. Preserve partial outputs after failures and prepare
a fresh destination.

## Protocol

- 64 independent test trajectories and 512 paired four-frame windows.
- 16 members, raw weights and fixed seeds; latent FM retains AB3-16.
- Recursive forecasts start at frame 16 and use 180 frames by default;
  `--horizon 100` supports a shorter common horizon.
- Metrics and coverage use one prediction pass per task. Energy uses a
  spatial vector per lead/channel, rather than a joint temporal norm.
- Means and SEs average dependent windows/channels within each trajectory;
  coverage-MAE SE retains covariance across levels. SEs condition on fixed
  checkpoints.
- Latency measures physical-output prediction, including normalization and
  codec/sampling, with five warmups and 50 synchronized timings.

The 859M checkpoint has 131,072 updates; the others have 262,144. Its
half-training label remains in plots. Updates do not match compute/exposure.

## Collect and plot

```bash
uv run --frozen python -m slurm_scripts.comparison.the_well.rayleigh_benard.collect_diagnostics \
  --host "$RB_SSH_HOST" --remote-root "$RB_REMOTE_DIAGNOSTICS_DIR" \
  --local-root "$RB_LOCAL_DIAGNOSTICS_DIR"
```

The collector waits for completion markers, downloads metrics/configs and
rebuilds comparisons. Alternatively, after all production jobs complete:

```bash
uv run --frozen python -m slurm_scripts.comparison.the_well.rayleigh_benard.plot_diagnostics \
  --manifest "$RB_DIAGNOSTICS_DIR/manifest.json" \
  --output "$RB_DIAGNOSTICS_DIR/comparison"
```

The builder verifies hashes, matched physical identities, checkpoint steps
and complete finite outputs before writing figures, tables and provenance.
Choose a fresh comparison destination.
