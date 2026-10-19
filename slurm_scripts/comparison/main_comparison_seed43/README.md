# Seed-43 main-comparison campaign

This is the reusable continuation of the completed CNS seed-43 rerun for
advection--diffusion (AD), the laser-wake GPE dataset, and Gray--Scott (GS).
It repeats the original data-generating procedures with data seed 43, keeps
training seed 42, trains ambient CRPS, encodes the new data with each
published dataset-specific autoencoder, trains latent flow matching, and then
runs the full trajectory-statistics evaluation.

Nothing is submitted by `plan` or `prepare`. Every submission requires both a
single named stage and `--yes-submit`. The pipeline intentionally has manual
gates instead of automatically queuing the whole DAG. Evaluation stages may
be queued after their training stage: the pipeline records an `afterany`
dependency so evaluation can use a checkpoint saved before a hard timeout.

## Identities held fixed

`campaign.yaml` is the authoritative campaign record. It pins:

- the clean current `../autosim` HEAD recorded by `prepare`;
- the current AutoSim generator presets and original scientific parameters;
- the original split sizes with a new top-level data seed;
- the published AE, CRPS, and FM reference runs;
- the published CRPS and FM epoch budgets;
- the evaluation member count, windows, metrics, videos, snapshots, and both
  aggregate and per-trajectory statistics outputs; and
- the Slurm resources for every stage.

Data workers run the current `../autosim` checkout directly with
`uv run --project ../autosim --frozen --no-sync`. `prepare` records its full
commit, and data submission and workers refuse a dirty checkout or a changed
HEAD. The pipeline never checks out another AutoSim commit. Current AutoSim
writes the normalization statistics inline for all three datasets.

The current simulator implementations retain the published physical
parameters. AD uses the vorticity-only wrapper around the same multichannel
solver, with exact sample counts and retry-safe split seeds. GPE retains all
original overrides, including its internal simulator `random_seed=42`; only
the top-level dataset seed changes to 43. GS retains `min_std=0.01`, the six
ordered pattern strata, and exact per-stratum counts. Validation permits only
the audited namespace, video-selection, and AD exact-count representation
changes relative to the published resolved configs.

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
uv run --project . --frozen --no-sync python "$PIPELINE" preflight
```

`preflight` is the cluster-local campaign audit. It compares the current
AutoSim compositions, published simulator parameters, epoch budgets, CNS
callback policy, and fixed-AE channel selection without creating outputs or
submitting work. It intentionally sits outside the portable pytest suite.

After both AutoCast and `../autosim` are clean and the pipeline code is
committed, reserve run identities and write a state file. This still does not
submit work:

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

If separate jobs queue poorly, cancel them before using one interactive
allocation for all three datasets sequentially:

```bash
srun --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=115G \
  --time=06:00:00 --pty /bin/bash --login

cd /home/u6eo/ltcx7228.u6eo/autocast-02
PIPELINE=slurm_scripts/comparison/main_comparison_seed43/pipeline.py
STATE=outputs/2026-08-13/campaign_main_comparison_seed43_runs_<git7>_<uuid7>/state.yaml
for DATASET in ad gpe gs; do
  uv run --project . --frozen --no-sync python "$PIPELINE" \
    run-stage --state "$STATE" --dataset "$DATASET" --stage data
done
```

The one-GPU data, cache, and evaluation stages request `115G` explicitly.
Four-GPU CRPS and FM training inherit `mem=0` from the established comparison
launcher preset, with one `srun` rank per GPU, four GPUs on one node, and 72 CPU
cores per GH200 rank. Their run directories and generated batch scripts appear
immediately after successful submission.

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

Once the CRPS and FM job IDs have been recorded, their evaluations can be
queued as two separate phases:

```bash
uv run --project . --frozen --no-sync python "$PIPELINE" \
  submit --state "$STATE" --dataset all --stage eval_crps --yes-submit
uv run --project . --frozen --no-sync python "$PIPELINE" \
  submit --state "$STATE" --dataset all --stage eval_fm --yes-submit
```

Each evaluation receives an `afterany` dependency on its corresponding
training job. CRPS selects the single overall-best multi-Winkler checkpoint.
FM prefers the finalized `processor.ckpt`; after a timeout it records and uses
the valid saved checkpoint with the highest global step.

Each evaluation writes the usual aggregate test, rollout, and benchmark CSVs
alongside the single-step and rollout per-trajectory tables (including
per-timestep trajectory rows). The validator requires both output families.

`status --state "$STATE"` prints the immutable paths and recorded job IDs.
Direct data, cache, and evaluation workers execute the exact AutoCast commit
recorded by `prepare`. CRPS and FM use the established AutoCast launcher from
that same clean checkout; keep the checkout at the recorded commit while they
are queued. Data submission and generation additionally require the exact
clean AutoSim commit; later stages use the immutable data-validation marker
that records that pin. The direct worker derives the repository from the
manifest argument, so `sbatch worker.sh ...` remains valid when Slurm stages
its copy in the node-local spool directory.
Failed or partial outputs are preserved for diagnosis; the pipeline never
removes or overwrites them.
