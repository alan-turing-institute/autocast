# Seed-43 main-comparison campaign

This is the reusable continuation of the completed CNS seed-43 rerun for
advection--diffusion (AD), the laser-wake GPE dataset, and Gray--Scott (GS).
It repeats the original data-generating procedures with data seed 43, keeps
training seed 42, trains ambient CRPS, encodes the new data with each
published dataset-specific autoencoder, trains latent flow matching, and then
runs the full trajectory-statistics evaluation.

Nothing is submitted by `plan` or `prepare`. Every submission requires both a
single named stage and `--yes-submit`. The pipeline intentionally has manual
gates instead of automatically queuing the whole DAG.

## Identities held fixed

`campaign.yaml` is the authoritative campaign record. It pins:

- the historical AutoSim generation and statistics commits;
- the original generator overrides and split sizes;
- the published AE, CRPS, and FM reference runs;
- the published CRPS and FM epoch budgets;
- the evaluation member count, windows, metrics, videos, snapshots, and
  trajectory-statistics outputs; and
- the Slurm resources for every stage.

AD uses the historical generator commit and the later historical statistics
commit, matching the original two-step procedure. GPE and GS use the inline
statistics code from their respective generation commits. The GPE generator
retains all original overrides, including its internal simulator
`random_seed=42`; only the top-level dataset seed changes to 43. GS retains the
six ordered pattern strata and exact per-stratum counts.

The fixed published AE always uses its original dataset's `stats.yml`. That
normalization is part of the learned AE coordinate system. The raw data path
alone changes to the new seed-43 dataset. Cache validation checks this
asymmetry explicitly and also checks the GPE `[1, 2]` real/imag channel subset.

Ambient CRPS uses `trainer=crps_main_comparison_rerun`. Its callback list is
exactly equal to the successful CNS rerun policy: 5% progress checkpoints,
best validation loss, early and late coverage/Winkler windows, one overall
best-Winkler checkpoint from 5%, validation plots, and EMA with decay 0.999.
FM uses the published `fm_main_comparison` checkpoint and EMA policy.

## Output layout

`prepare` calls AutoCast's existing `auto_run_name` function for both model
runs. It also assigns the same seven-character git-hash and seven-character
UUID suffix to campaign, cache, and evaluation identities. A prepared campaign
therefore records paths of this form:

```text
outputs/2026-08-13/
├── campaign_main_comparison_seed43_runs_<git7>_<uuid7>/
│   ├── state.yaml
│   └── slurm_logs/
├── cache_published_ae_ad64_<git7>_<uuid7>/
├── crps_ad64_vit_azula_large_<git7>_<uuid7>/
│   └── eval_crps_ad64_<git7>_<uuid7>/
└── diff_ad64_flow_matching_vit_<git7>_<uuid7>/
    └── eval_fm_ad64_<git7>_<uuid7>/
```

GPE and GS use `gpe64` and `gs64` in the same layout. The external raw dataset
paths are also dated and seed-labelled, but are not model run directories.
Prepared paths are immutable: workers refuse an existing dataset, cache, run,
or evaluation destination.

## Commands

Run from this checkout with its locked `uv` environment:

```bash
cd /home/u6eo/ltcx7228.u6eo/autocast-02
PIPELINE=slurm_scripts/comparison/main_comparison_seed43/pipeline.py

uv run --project . --frozen --no-sync python "$PIPELINE" plan
```

After the pipeline code is committed, reserve run identities and write a state
file. This still does not submit work:

```bash
uv run --project . --frozen --no-sync python "$PIPELINE" prepare
```

Use the printed state path in later commands. The first active phase is data
only:

```bash
STATE=outputs/2026-08-13/campaign_main_comparison_seed43_runs_<git7>_<uuid7>/state.yaml

uv run --project . --frozen --no-sync python "$PIPELINE" \
  submit --state "$STATE" --dataset all --stage data --yes-submit
```

Wait for all three jobs, inspect the example videos and data distributions,
then run the validator for each dataset:

```bash
uv run --project . --frozen --no-sync python "$PIPELINE" \
  validate --dataset ad --stage data
uv run --project . --frozen --no-sync python "$PIPELINE" \
  validate --dataset gpe --stage data
uv run --project . --frozen --no-sync python "$PIPELINE" \
  validate --dataset gs --stage data
```

Only after that review should cache and CRPS be submitted. They are independent
branches and can run concurrently:

```bash
uv run --project . --frozen --no-sync python "$PIPELINE" \
  submit --state "$STATE" --dataset all --stage cache --yes-submit
uv run --project . --frozen --no-sync python "$PIPELINE" \
  submit --state "$STATE" --dataset all --stage crps --yes-submit
```

Validate the caches before submitting FM:

```bash
uv run --project . --frozen --no-sync python "$PIPELINE" \
  validate --state "$STATE" --dataset ad --stage cache
uv run --project . --frozen --no-sync python "$PIPELINE" \
  validate --state "$STATE" --dataset gpe --stage cache
uv run --project . --frozen --no-sync python "$PIPELINE" \
  validate --state "$STATE" --dataset gs --stage cache
uv run --project . --frozen --no-sync python "$PIPELINE" \
  submit --state "$STATE" --dataset all --stage fm --yes-submit
```

After validating CRPS and FM checkpoints, submit their evaluations as two
separate phases:

```bash
uv run --project . --frozen --no-sync python "$PIPELINE" \
  submit --state "$STATE" --dataset all --stage eval_crps --yes-submit
uv run --project . --frozen --no-sync python "$PIPELINE" \
  submit --state "$STATE" --dataset all --stage eval_fm --yes-submit
```

`status --state "$STATE"` prints the immutable paths and recorded job IDs.
Workers execute the exact committed source recorded by `prepare` and refuse a
dirty or changed checkout. Failed or partial outputs are preserved for
diagnosis; the pipeline never removes or overwrites them.
