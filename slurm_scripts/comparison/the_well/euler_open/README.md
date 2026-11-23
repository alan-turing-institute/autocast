# Euler Multi-Quadrants open-BC comparison

This experiment uses the stock Well loader on the verified Flatiron dataset
`/projects/u6eo/autocast/datasets/euler_multi_quadrants_openBC`. Its
`well_base_path` is `/projects/u6eo/autocast/datasets`. Ten files per split hold
4,000 train, 500 validation and 500 test trajectories. Each trajectory has 101
frames. The published `euler_all` 80× f32c64 codec was trained on both open and
periodic Euler. This experiment freezes that codec and uses only open-BC data
for its cache, surrogate training, and evaluation.

The run pins LoLA revision `944ef5e877493a68cfec0564049c9335e5935d73`, the
published `s259b4l6` surrogate config, and the codec `n6g8kix8`. The archive
listed by `experiments/eval_all.sh` was downloaded from Flatiron in pilot job
8633077. Its recorded SHA-256 was rechecked on Isambard. Both ZIP members were
hashed directly and matched the staged files. The pins and staged paths are in
[the shared Hydra source config](../../../../local_hydra/local_experiment/the_well/euler_open/_sources.yaml).

Both cache paths use the checkpoint's training statistics and apply `log1p`
to energy, density and pressure before standardizing. Channels are ordered
energy, density, pressure, momentum_x, momentum_y. The label is gamma followed
by the four open boundary codes, in The Well's padding order. The codec maps
`512 × 512 × 5` to `16 × 16 × 64`. MiniWell stores `(N, 101, 64, 16, 16)` and
`(N, 5)` float32 datasets. Both backends store one unaugmented latent trajectory
per source trajectory. A window uses raw frame indices
`[t, t+4, t+8, t+12, t+16]`, with 85 possible starts per trajectory.

The CRPS model uses LoLA's 256-dimensional label embedding and joint ViT-Large
trunk, with 1,280 spatiotemporal tokens for a five-frame latent window. It
inherits the AutoCast/RB AdamW schedule and uses LoLA's published Euler context
mask settings (Poisson rate 1, prefix probability 0.66, minimum one frame).
AutoCast trains an eight-member, noise-modulated ensemble with masked alpha-fair
CRPS. The latent presets use FP32 training with high float32 matmul precision,
matching LoLA's training precision. LoLA's latent surrogate trains with MSE. The
paired evaluation uses a fixed one-frame prefix and scores 16 members against
raw targets after deterministic decoding, inverse standardization and
`expm1` on the three transformed channels. It selects 64 open test trajectories
and eight distinct windows per trajectory with seed 0. The resulting manifest
records source filenames, zero-based trajectory rows and raw context indices,
and can be reused across checkpoints and both cache backends.
The paired output includes 19 coverage levels, spatial energy, and
trajectory-level standard errors.
[Rayleigh–Bénard comparison protocol](https://github.com/alan-turing-institute/aiphys/issues/112).

### Latent and ambient parameter match

| Dataset | Latent CRPS | Matched ambient CRPS | Ambient increase |
| --- | ---: | ---: | ---: |
| Rayleigh–Bénard | 223,019,328 | 225,969,408 | 1.32% |
| Euler open BC | 223,019,072 | 226,755,840 | 1.68% |

The RB latent processor embeds six labels. Euler embeds five, which accounts
for the 256-parameter difference. Both use the 1024-wide, 16-block, eight-head
ViT. The published Euler surrogate has 222,756,672 parameters and the CRPS
noise projection adds 262,400. The counts cover trainable parameters and
exclude the frozen codec. The matched Euler ambient preset uses 16×16 patches
and activation checkpointing, as in RB. Its larger patch projections add
3,736,768 parameters. The five-frame Euler window has 1,280 latent tokens and
5,120 ambient tokens. The ambient preset uses the same field preprocessing,
stride, gamma and boundary labels, masked CRPS and paired test manifest on
the native grid.

The latent recipes use a fixed, seeded permutation of validation windows
across the full open split before Lightning applies the validation batch cap.
This gives each gamma shard representation without adding validation batches.
At 32 ranks and 16 batches, CRPS scores 512 windows and FM scores 4,096.
The CRPS windows are a subset of the FM windows at these batch sizes.

## Local inspection

```bash
euler=slurm_scripts/comparison/the_well/euler_open
preset=the_well/euler_open/crps_spatiotemporal_latent
workflow=slurm_scripts.comparison.the_well.euler_open.workflow
bash -n "$euler/job.sh"
uv run --no-sync python -m autocast.scripts.train.processor "local_experiment=$preset" --cfg job --resolve
uv run --no-sync python -m autocast.scripts.train.processor "local_experiment=${preset}_smoke" --cfg job --resolve
uv run --no-sync python -m autocast.scripts.train.processor \
    local_experiment=the_well/euler_open/fm_spatiotemporal_latent --cfg job --resolve
uv run --no-sync python -m autocast.scripts.train.processor \
    local_experiment=the_well/euler_open/fm_spatiotemporal_latent_smoke --cfg job --resolve
uv run --no-sync python -m autocast.scripts.train.encoder_processor_decoder \
    local_experiment=the_well/euler_open/crps_spatiotemporal_ambient_matched --cfg job --resolve
uv run --no-sync python -m "$workflow" plan
uv run --no-sync python -m "$workflow" --profile smoke plan
uv run --no-sync python -m "$workflow" windows /private/tmp/euler-open-paired-test.yaml
uv run --no-sync python -m pytest -q tests/scripts/test_euler_open_lola.py \
    tests/nn/test_spatiotemporal_vit.py tests/processors/test_masked_window.py \
    tests/processors/test_flow_matching.py \
    tests/scripts/test_rb_spatiotemporal.py tests/scripts/test_cache_latents.py \
    tests/data/test_cached_latent_dataset.py \
    tests/scripts/test_eval_encoder_processor_decoder.py
```

## Isambard launches for approval

Run from the prepared AutoCast checkout. Choose an approved shared directory
with at least 35 GiB per production cache and space for training checkpoints.
The commands assume a committed AutoCast checkout with a working CUDA `uv`
environment. The run records the checkout revision and marks local source
changes with `+dirty`. The `root` and `logs` directories must already exist.
Use separate run directories for smoke, production, retries from scratch and
each cache backend.

```bash
root=/absolute/approved/euler-run-root
logs="$root/logs"
job=slurm_scripts/comparison/the_well/euler_open/job.sh
common=(--account=brics.u6eo --partition=workq --chdir="$PWD" --nodes=1)

# AutoCast smoke: one trajectory per split, two CRPS optimizer steps,
# one validation and one test batch, two paired windows with 16 members each.
export EULER_CACHE_BACKEND=autocast
smoke_cache=$(sbatch --parsable "${common[@]}" --job-name=euler-cache-smoke \
    --array=0-2%3 --ntasks=1 --gpus=1 --cpus-per-task=4 --mem=32G --time=00:20:00 \
    --output="$logs/%x-%A_%a.out" "$job" cache smoke "$root/cache-smoke")
smoke_train=$(sbatch --parsable "${common[@]}" --dependency=afterok:"${smoke_cache%%;*}" \
    --job-name=euler-crps-smoke --ntasks=1 --gpus=1 --cpus-per-task=4 --mem=48G --time=00:25:00 \
    --output="$logs/%x-%j.out" "$job" train smoke "$root/cache-smoke" "$root/train-smoke")
sbatch "${common[@]}" --dependency=afterok:"${smoke_train%%;*}" \
    --job-name=euler-paired-smoke --ntasks=1 --gpus=1 --cpus-per-task=4 --mem=48G --time=00:20:00 \
    --output="$logs/%x-%j.out" "$job" eval smoke "$root/train-smoke" \
    "$root/eval-smoke" --expected-step 2
```

Job 8659936 supplies the earlier CRPS/FM GPU evidence. It used one cached
trajectory per split and the RB presets. Artifacts are under
`/projects/u6eo/autocast/datasets/.lola-well-stage/euler-lola-train-smoke-kEhbZU/attempt-8659936-CiS9Gx`.
The smoke above exercises the new canonical data path, Euler configuration,
cache verification and physical-space decoding. Confirm the production batch
size and throughput on the target GPUs before booking the full training budget.

### Pinned upstream LoLA cache

Once the upstream environment is ready, the following runs LoLA's unchanged
`experiments/cache_latents.py::cache_latent` for one complete shard. The upstream
script uses its own `dawgz` launcher when run as a CLI. Here Slurm uses the same
array mapping as AutoCast and calls the original shard function in a temporary
run directory with the verified codec. The pilot uses index 10, the first
validation shard with 50 trajectories. Use a separate checkout and `uv`
environment with importing `lola.autoencoder`, `dawgz`, and the stock Well loader
verified on Isambard. The old xFormers/Torch pins need an ARM CUDA-compatible
environment. `EULER_LOLA_ENV` can point to an already prepared environment
outside the LoLA checkout.

```bash
export EULER_LOLA_CHECKOUT=/absolute/pinned/lola
export EULER_CACHE_BACKEND=upstream
sbatch "${common[@]}" --job-name=euler-cache-lola-pilot \
    --array=10 --ntasks=1 --gpus=1 --cpus-per-task=4 --mem=48G --time=01:00:00 \
    --output="$logs/%x-%A_%a.out" "$job" cache production "$root/cache-lola"
```

Inspect the pilot shard's non-finite checks, labels and shape, and compare it
with the AutoCast shard before running the upstream backend across the other 29
shards. It uses all frames in a trajectory in one encode call. Budget and test
its memory separately from the four-frame chunks of the AutoCast path.
[Pinned LoLA cache source](https://github.com/PolymathicAI/lola/blob/944ef5e877493a68cfec0564049c9335e5935d73/experiments/cache_latents.py).

Use `cache-periodic production <absolute-cache-path>` to encode the canonical
periodic split with the same pinned upstream script and unaugmented settings.
The array still maps indices 0–9 to train, 10–19 to validation and 20–29 to
test. The periodic cache validates its own boundary labels. Use
`verify-periodic-cache production <absolute-cache-path>` after the full array.
Set `EULER_CODEC_DIR` to select a local copy of the published codec. The cache
checks both file hashes before encoding.

### Joint open and periodic Euler

The `the_well/euler_all` Hydra presets concatenate the canonical open and
periodic sources within each split. The two caches use the same frozen codec
and remain unaugmented. The ambient preset loads raw fields from both Well
datasets and applies the same Euler field transforms in the model.

```bash
uv run --no-sync python -m "$workflow" verify-joint-cache
uv run --no-sync python -m "$workflow" verify-joint-ambient
uv run --no-sync python -m autocast.scripts.train.processor \
    local_experiment=the_well/euler_all/crps_spatiotemporal_latent --cfg job --resolve
uv run --no-sync python -m autocast.scripts.train.processor \
    local_experiment=the_well/euler_all/fm_spatiotemporal_latent --cfg job --resolve
uv run --no-sync python -m autocast.scripts.train.encoder_processor_decoder \
    local_experiment=the_well/euler_all/crps_spatiotemporal_ambient_matched --cfg job --resolve
```

For paired evaluation, generate one manifest per boundary with
`workflow --source open windows` and `workflow --source periodic windows`.
Use the maintained `eval=the_well/euler_open` metric preset for both.
For a latent checkpoint, set `eval.mode=encode_once`,
`datamodule.data_paths=null` and `datamodule.data_path` to the selected
boundary's cache root. The evaluator infers the matching raw Well dataset
and codec from that root. For ambient evaluation, set `eval.mode=ambient`
and select one `datamodule.well_dataset_name`. Pass the matching manifest
as `eval.test_window_subset` in each run.

The [joint launcher](../euler_all/job.sh) uses the two-root preflight and the
same eight-node TF32 layout for both methods. Its `probe` profile runs
three 64-update epochs with a 16-batch validation pass. Its `production`
profile runs 4,096 epochs and validates every eight epochs. Use a separate
single-GPU allocation for the `eval` operation. It calls `autocast eval
--workdir` with explicit cache paths and sends paired results to `eval-open`
and `eval-periodic` under the training run. Each evaluation renders the
first scored window as `videos/test_window_0.mp4`. The probe scores two
windows per boundary while verifying the full paired manifest.

### Rollouts for the LoLA comparison

Use `eval=the_well/euler_all` for raw-space rollout curves from the final
latent checkpoints. It extends the RB reference preset with one observed
frame at raw time zero and 25 forecasts at raw steps 4, 8, …, 100. Step
100 is the last observed Euler frame. The RB diagnostics use stride one
and report 100- and 180-frame rollouts from frame 16. The Euler rollout
uses deterministic decoding and 16 members. FM uses AB3 with 16 solver
steps. See the [LoLA evaluation protocol](https://arxiv.org/html/2507.02608v4).

The joint launcher's `eval-rollout` operation generates one manifest per
boundary from LoLA's seed-0 test draw across the pooled split. It evaluates
each boundary separately. One GPU per method is sufficient for the pilot.
For each run directory, preview the outputs with two trajectories per
boundary before choosing a production walltime.

```bash
sbatch --partition=workq --account="$AUTOCAST_SLURM_ACCOUNT" \
    --nodes=1 --gpus-per-node=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
    --time=00:15:00 slurm_scripts/comparison/the_well/euler_all/job.sh \
    eval-rollout probe fm "$fm_run"
```

Set `AUTOCAST_EXPECTED_REVISION` to the reviewed checkout commit and use
`crps "$crps_run"` for CRPS. The production profile uses the whole
selected cohort. The rollout pass scores per-lead metrics in physical
units. It includes the RB VRMSE, CRPS, spatial energy, spread–skill ratio
and 19 coverage levels, plus the LoLA power-spectrum errors. It skips
benchmarks, next-window scoring and a second pass for window aggregates.
CSV column `i` represents lead `4 * (i + 1)` in raw steps. Use the existing
`eval` operation for the paired four-lead task. For a pooled LoLA curve,
use the same pooling and metric reductions for both models.

Each evaluation writes one MP4 under its `videos/` directory using the
first trajectory in the saved boundary manifest. It shows ground truth,
the ensemble mean and standard deviation, and member zero for all five
fields. FM and CRPS use the same selected trajectory for each boundary.
The pilot checks video rendering and scoring with two trajectories per
boundary.

### Production array, training and paired evaluation

Run this chain with `cache_path="$root/cache-lola"` and the upstream exports
above to use the official cache for training. Run the same chain with
`EULER_CACHE_BACKEND=autocast`, `cache_path="$root/cache-autocast"` and separate
outputs to use the AutoCast cache.

```bash
cache_path="$root/cache-lola"
run_path="$root/train-lola"
eval_path="$root/eval-lola"
cache_job=$(sbatch --parsable "${common[@]}" --job-name=euler-cache \
    --array=0-29%4 --ntasks=1 --gpus=1 --cpus-per-task=4 --mem=48G --time=03:00:00 \
    --output="$logs/%x-%A_%a.out" "$job" cache production "$cache_path")
train_job=$(sbatch --parsable "${common[@]}" --dependency=afterok:"${cache_job%%;*}" \
    --job-name=euler-crps --ntasks-per-node=4 --gpus=4 --cpus-per-task=16 --mem=0 --time=24:00:00 \
    --output="$logs/%x-%j.out" "$job" train production "$cache_path" "$run_path")
sbatch "${common[@]}" --dependency=afterok:"${train_job%%;*}" \
    --job-name=euler-paired --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
    --output="$logs/%x-%j.out" "$job" eval production "$run_path" "$eval_path"
```

Retry by submitting the same array or selected failed indices. A completed
shard is reused only if its provenance and HDF5 shapes and dtypes match. Each
shard's `provenance` attribute records its source file, split, zero-based source
row range, codec hashes, transform, statistics, label order, backend, strides
and latent shape. MiniWell sees a shard only after it is complete. Hidden
`.partial` directories left by hard termination can be inspected and removed
after confirming no job is using them. Training checks the full expected
inventory before starting. Attach a new training dependency to a successful
replacement array if the original array failed.

The training launcher inherits the 4,096-epoch cosine horizon and stops at the
24-hour allocation budget. Paired evaluation uses the raw weights in the
`processor.ckpt` endpoint for that allocation. Its provenance reports the
global step. Pass `--expected-step` to require a particular endpoint. The
paired output includes per-window, per-lead and per-field metrics, the window
manifest, resolved configuration and a completion marker.

### Latent flow matching

FM reuses the verified latent cache, ViT trunk, training context mask, and
paired test windows. It trains with masked velocity regression and draws 16
members for paired evaluation using 16 third-order Adams–Bashforth steps.
It has 222,888,256 trainable parameters. FM's time embedding accounts for
the small difference from the noise-modulated latent CRPS processor.
Following the RB batch convention, the four-GPU production run uses 256
distinct windows per update. CRPS uses 32 windows with eight members each.
This matches the number of member-window predictions per update and gives
FM eight times the distinct-window exposure at the same update count.

Run the FM smoke against the existing smoke cache first. The paired smoke
checks two windows with 16 members.

```bash
fm_smoke=$(EULER_CACHE_BACKEND=autocast sbatch --parsable "${common[@]}" --dependency=afterok:"${smoke_cache%%;*}" \
    --job-name=euler-fm-smoke --ntasks=1 --gpus=1 --cpus-per-task=4 --mem=48G --time=00:25:00 \
    --output="$logs/%x-%j.out" "$job" train-fm smoke "$root/cache-smoke" "$root/train-fm-smoke")
sbatch "${common[@]}" --dependency=afterok:"${fm_smoke%%;*}" \
    --job-name=euler-paired-fm-smoke --ntasks=1 --gpus=1 --cpus-per-task=4 --mem=48G --time=00:20:00 \
    --output="$logs/%x-%j.out" "$job" eval smoke "$root/train-fm-smoke" "$root/eval-fm-smoke" --expected-step 2
```

After checking memory and per-update time at the production batch size, use
the same `cache_path`, `cache_job` and backend as the approved latent CRPS run.

```bash
fm_train=$(sbatch --parsable "${common[@]}" --dependency=afterok:"${cache_job%%;*}" \
    --job-name=euler-fm --ntasks-per-node=4 --gpus=4 --cpus-per-task=16 --mem=0 --time=24:00:00 \
    --output="$logs/%x-%j.out" "$job" train-fm production "$cache_path" "$root/train-fm")
sbatch "${common[@]}" --dependency=afterok:"${fm_train%%;*}" \
    --job-name=euler-paired-fm --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
    --output="$logs/%x-%j.out" "$job" eval production "$root/train-fm" "$root/eval-fm"
```

### Matched ambient probe and paired evaluation

The ambient launcher reads the canonical raw shards. Start with a two-step
probe at the production batch and ensemble sizes to measure fit and
throughput on four GPUs. After reviewing the probe, use the same launcher
with a new run directory and the approved allocation budget for training.
The ambient paired evaluator uses the same window manifest and scoring code
as the latent evaluator.

```bash
ambient_probe=$(sbatch --parsable "${common[@]}" --job-name=euler-ambient-probe \
    --ntasks-per-node=4 --gpus=4 --cpus-per-task=16 --mem=0 --time=00:45:00 \
    --output="$logs/%x-%j.out" "$job" train-ambient production "$root/ambient-probe" \
    +trainer.max_steps=2 trainer.max_epochs=1 trainer.limit_train_batches=2 \
    trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=1)
```

After the probe has passed and the production allocation is approved, run
the training and evaluation chain.

```bash
ambient_train_job=$(sbatch --parsable "${common[@]}" --job-name=euler-crps-ambient \
    --ntasks-per-node=4 --gpus=4 --cpus-per-task=16 --mem=0 --time=24:00:00 \
    --output="$logs/%x-%j.out" "$job" train-ambient production "$root/train-ambient")
sbatch "${common[@]}" --dependency=afterok:"${ambient_train_job%%;*}" \
    --job-name=euler-paired-ambient --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
    --output="$logs/%x-%j.out" "$job" eval-ambient production "$root/train-ambient" "$root/eval-ambient"
```

Get separate approval for shared Isambard writes, GPU submissions, pushes and
GitHub posts.
