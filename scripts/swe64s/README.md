# Stochastic SWE 64x64 baselines

This directory is the entry point for the larger stochastic SWE baselines:
paper-scale global-noise afCRPS and latent flow matching (FM). The `s` suffix
means stochastic; a future deterministic counterpart can use `swe64d`.
Keep simulator-parameter variation and model conditioning as separate config
choices within each study.

The baseline workflow uses the standard AutoCast model, data and metric APIs.
Historical 32x32 runs and follow-on loss, noise and physics variants are separate
comparisons; they are not dependencies of these initial fits.

The current one-output comparisons are listed in
[Current noise and Fourier-loss controls](#current-noise-and-fourier-loss-controls)
and [Matched-update residual flow matching](#matched-update-residual-flow-matching).
The [experiment readiness inventory](#experiment-readiness-inventory) separates
existing fits, prepared follow-ups and decisions still needed before launch.
The original presets retain their four-output/time-budget setup.

## Data and scope

The [dataset manifest](../../local_hydra/local_experiment/swe64s/dataset.json)
records the generator identity, physical settings and split checksums. The
existing dataset has 200/20/20 trajectories, 321 frames, channels `h,u,v`,
saved every 0.25 from t=40 through t=120. Physical parameters are fixed across
trajectories; initial states and forcing realizations vary. Training
normalization comes from the existing `stats.yml`, not recomputed test data.

The original paper-scale presets use one input frame, four output frames and
window stride one, matching the paper-style setup. These are direct full-field targets,
not unrolled two-step or physics-residual training. The larger dataset, model
and output window mean this is not a resolution-only ablation of the earlier
32x32 fits.

## Presets and shared settings

All presets live in
[`local_hydra/local_experiment/swe64s`](../../local_hydra/local_experiment/swe64s).
`common.yaml` holds windowing and portable launch defaults; `data.yaml` adds the
raw dataset and normalization paths. `afcrps.yaml` pins width 568, 12 blocks,
8 heads, patch 4, 1024D global AdaLN noise, eight training members and
alpha=0.95. The fixed scalar metadata is excluded by `with_constants=false`
and `include_global_cond=false`; the current fields and latent noise still
condition the forecast.

The processor's standard global-noise sampling and gated transformer blocks
are used without extension-specific options. FM likewise uses its standard
single-draw training objective. This preserves the baseline settings without
requiring spatial-noise, sublayer-conditioning or multistep-training APIs.

The architectural basis is
[`epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`](../../local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml).
Published run and checkpoint selections are indexed in
`autocast_paper_02/provenance/2026-09-25_submission_outputs/SUMMARY.md`.
Use that inventory to distinguish the main afCRPS run from ordinary/fair-CRPS
and architecture ablations; preset names alone do not identify those runs.

`autoencoder.yaml` uses the paper DC autoencoder capacity and factor-four
compression: 64x64x3 fields become 16x16x8 latents. Its three levels have two
stride-two transitions; input patching is explicitly one. Circular padding is
a deliberate SWE adaptation to the periodic domain; the published CNS AE used nonperiodic
padding. `cache_latents.yaml` mirrors the encoder/decoder settings and saves
all 321 frames, one trajectory per encoding call. Do not reuse another dataset's
AE or silently change normalization between fitting and caching.

`flow_matching.yaml` follows the
[`processor/conditioned_navier_stokes/fm_vit_large.yaml`](../../local_hydra/local_experiment/processor/conditioned_navier_stokes/fm_vit_large.yaml)
capacity: width 704, 12 blocks, 8 heads, patch 1 on the latent grid and 50 Euler
steps at inference. FM retains field conditioning and the flow-time embedding,
but excludes simulator-parameter conditioning. Its flow-matching objective is
not comparable numerically with afCRPS. Batch 256 matches 32x8 afCRPS model
evaluations per GPU, not the number of independent training inputs; record
optimizer updates and sample counts when comparing training budgets.

## Local preparation and launch gate

### Current noise and Fourier-loss controls

All current CRPS controls use the original 200/20/20 raw trajectories and
training statistics, one input/output frame and stride one. They retain width
224, 12 blocks, 8 heads, patch 4, modulation width 256, eight members and batch
32 per GPU. Simulator-parameter conditioning is disabled. Predictions are
delta-normalized persistence corrections; both loss terms score the full
reconstructed states in dataset-normalized coordinates.

| Noise | afCRPS control | CRPS-FFT |
| --- | --- | --- |
| Global 256D | [Existing global](../../local_hydra/local_experiment/swe64s/afcrps_global256_residual_one_output_12block_256ep_8h.yaml) | [Global](../../local_hydra/local_experiment/swe64s/crps_fft_global256_residual_one_output_12block_256ep_8h.yaml) |
| IID spatial 1D/token | [Existing spatial](../../local_hydra/local_experiment/swe64s/afcrps_spatial1d_residual_one_output_12block_256ep_8h.yaml) | [Spatial](../../local_hydra/local_experiment/swe64s/crps_fft_spatial1d_residual_one_output_12block_256ep_8h.yaml) |
| Structured spatial 1D/token | [Structured](../../local_hydra/local_experiment/swe64s/afcrps_structured_spatial1d_residual_one_output_12block_256ep_8h.yaml) | [Structured](../../local_hydra/local_experiment/swe64s/crps_fft_structured_spatial1d_residual_one_output_12block_256ep_8h.yaml) |

CRPS-FFT composes the existing `ensemble_fourier_crps` preset: pointwise
afCRPS plus complex Fourier-coefficient afCRPS, weights `[1.0, 0.1]`, both
with alpha 0.95 and orthonormal spatial FFTs. Data, model and optimizer settings
otherwise match the corresponding afCRPS control. Native `val_loss` includes
both terms; use the retained common `val_afcrps` and physical evaluation
metrics for cross-loss comparisons.

An online auxiliary is available as
[`SpectralSpreadErrorLoss`](../../src/autocast/losses/anomaly.py), composed
through the existing `WeightedSumLoss` API. It compares batch-mean radial
anomaly power with observed residual power using the current ensemble mean,
stop-gradient on the observation reference and a finite-member moment
correction. It requires neither simulator redraws nor a frozen mean cache.
The [loss documentation](../../docs/reference/losses/spectral.md#online-anomaly-power-regularization)
defines its normalization, process-local pooling and limitations. The baseline
presets do not enable it. The separate control below fixes its bands but still
requires a reviewed calibration record to supply its scalar weight.

The older structured source uses modes `(3,0),(0,3),(2,2),(2,-2)` on the 16x16 token
grid. Four sine/cosine pairs use eight Gaussian coefficients in the single
noise channel, compared with 256 independent scalars in IID spatial noise.
The 1-to-256 modulation projection is unchanged. Each member/call redraws
independently; these presets introduce spatial correlation, with no temporal
persistence or output filtering.
Those narrowband one-channel presets remain unchanged for provenance; they
are not the new four-channel multiscale controls below.

Every CRPS preset keeps a fresh 256-epoch cosine schedule, LR `2e-4` to zero,
and no step cap. At four GPUs this is 128,000 updates. The inherited allocation
is eight hours with a 7h45m trainer safety cap, best-validation and last
checkpoints, and no automatic early-stopping callback. If the cap binds,
record the achieved budget and resume the full checkpoint before claiming
completion. Existing global/IID afCRPS fits remain the comparison controls.

The first intended pair is **ambient residual FM** and **IID spatial-1D
CRPS-FFT**. Both use the original raw dataset and require no AE/cache. FM uses
128,000 updates, batch 256/GPU and step-based cosine LR `1e-4` to zero; CRPS-FFT
uses the epoch-based protocol above. Compose the latter without fitting:

```bash
uv run --frozen python -m autocast.scripts.train.encoder_processor_decoder \
  local_experiment=swe64s/crps_fft_spatial1d_residual_one_output_12block_256ep_8h \
  --cfg job --resolve
```

The focused local tests are `tests/scripts/test_swe64_fourier_configs.py` and
`tests/scripts/test_swe64_residual_fm.py`. They exercise configuration, resolved
mode/grid compatibility, residual reconstruction, score arithmetic, gradients
and short synthetic rollouts. Target-machine data/W&B preflight remains a
separate preparation step.

The shared evaluation wrapper accepts saved one-input/one-output and
one-input/four-output tasks. It evaluates 100 saved-time leads in either case:
100 calls for one-output models, or 25 calls for four-output models. Both
teacher-forced and free-running evaluations use 16 members, the held-out test
split, central 90% coverage, and per-channel/per-trajectory metrics. Saved model
and normalization settings are retained. Outputs must use fresh directories.

### Four-channel multiscale noise controls

These opt-in presets use four independent coloured Gaussian fields with
smooth radial filters. They do not estimate the filters from simulator
redraws or field statistics. The source is FCN3-inspired spatial noise,
not a reproduction of FCN3's full architecture or temporally evolving
eight-field process. Every forecast call independently redraws the noise.

| Design | afCRPS preset | CRPS-FFT preset | Noise grid | Half-power k |
| --- | --- | --- | --- | --- |
| Spatial AdaLN | `afcrps_multiscale_spatial4d_residual_one_output_12block_256ep_8h` | `crps_fft_multiscale_spatial4d_residual_one_output_12block_256ep_8h` | 16x16 tokens | `[1, 2, 4, 6]` |
| Input concat | `afcrps_multiscale_concat4d_residual_one_output_12block_256ep_8h` | `crps_fft_multiscale_concat4d_residual_one_output_12block_256ep_8h` | 64x64 cells | `[2, 4, 8, 16]` |

For channel c the relative power envelope is
`P_c(k) = 2**(-(k / k_half[c])**2)`. FFT coefficients are multiplied by its
square root. `k` is a continuous radial magnitude, not a rounded shell
index. With `domain_lengths=[1,1]`, it counts cycles across the domain;
the source also supports explicit rectangular physical domain lengths.
Scales must be below both axial Nyquist frequencies of the consumer's grid.
The envelopes overlap rather than divide the spectrum into hard bands.
Expected pointwise variance is one in every channel, retaining DC and
Gaussian draw-to-draw amplitude fluctuations; draws are not standardized.

AdaLN uses the existing shared 4-to-256 conditioning projection and reuses
one draw across all blocks. Concat appends four input channels before the
encoder, keeping its physical inputs, targets and persistence baseline
clean. Its stochastic AdaLN path is disabled; blocks retain their default
256-wide deterministic modulation. Four-channel AdaLN draws 1,024 grid
values/member while native-grid concat draws 16,384. The designs intentionally
change injection, grid and scale range and are not a pure placement ablation
or a noise-dimension-matched comparison with global-256/spatial-1D.

Both retain the original 200/20/20 data and training statistics, one input
and output frame, stride one, delta-normalized persistence residuals,
width 224, 12 blocks, eight heads and patch four. Simulator-parameter
conditioning stays off. Training uses eight members, batch 32/GPU on four
GPUs, 256 epochs and cosine LR `2e-4` to zero, with the existing eight-hour
allocation and 7h45m trainer cap. The FFT versions change only the loss to
the existing `[1.0, 0.1]` physical/coefficient afCRPS composite.

The initial proposed pair uses **afCRPS alone** to assess each complete
noise design before adding the spectral objective. Prepare selected
commands without submitting, opening W&B or creating output directories:

```bash
uv run --frozen --no-sync python scripts/swe64s/prepare_followups.py \
  --datasets-root /projects/u6eo/autocast/datasets \
  --output-root /projects/u6eo/autocast/outputs/swe64s/NEW_STUDY \
  --run multiscale_adaln_afcrps --run multiscale_concat_afcrps
```

The helper prints standard CLI dry-run commands. Select
`multiscale_adaln_crps_fft` or `multiscale_concat_crps_fft` explicitly for the
spectral variants. Omitting `--run` preserves its original global-FFT/latent-FM
pair; adding these presets does not enlarge a default training queue.
All runs require separate target-machine data/code/W&B preflight and launch
approval. Common scores, anomaly power, divergence/vorticity balance and
teacher-forced/free-running evaluation remain required after fitting.

### Experiment readiness inventory

This is a local config/planning inventory, not a live scheduler status.
Existing-fit entries refer to the completed evaluation work discussed in
this study; no new submission is implied. All preset names below live in
`local_hydra/local_experiment/swe64s` with a `.yaml` suffix.

| Experiment | Preset | Preparation state |
| --- | --- | --- |
| Global-256 afCRPS reference | `afcrps_global256_residual_one_output_12block_256ep_8h` | Existing fit/evaluation; retain selected checkpoint |
| IID spatial-1D afCRPS reference | `afcrps_spatial1d_residual_one_output_12block_256ep_8h` | Existing fit/evaluation; retain selected checkpoint |
| IID spatial-1D CRPS-FFT | `crps_fft_spatial1d_residual_one_output_12block_256ep_8h` | Existing fit/evaluation; retain selected checkpoint |
| Ambient residual FM | `flow_matching_residual_ambient_12block_one_output_128k` | Existing completed fit/evaluation; use final raw checkpoint under the agreed selection rule |
| Global-256 CRPS-FFT | `crps_fft_global256_residual_one_output_12block_256ep_8h` | Prepared; `--run global_crps_fft` |
| Latent residual FM, using the 1,000-trajectory AE | `flow_matching_residual_12block_one_output_128k` | Prepared; `--run latent_residual_fm`; verify the pinned cache/AE identity on target |
| Multiscale AdaLN, afCRPS / CRPS-FFT | Two spatial-4D presets above | Prepared; explicit `--run multiscale_adaln_afcrps` / `multiscale_adaln_crps_fft` |
| Multiscale concat, afCRPS / CRPS-FFT | Two concat-4D presets above | Prepared; explicit `--run multiscale_concat_afcrps` / `multiscale_concat_crps_fft` |
| IID spatial-1D CRPS-FFT + online anomaly-power auxiliary | `crps_fft_anomaly_spatial1d_residual_one_output_12block_256ep_8h` | Config tested; **blocked on training-only weight calibration**, not ready to launch |
| afCRPS alone + online anomaly-power auxiliary | No dedicated preset yet | Discussed extension; prepare/calibrate separately if selected |
| IID spatial-4D noise control | No matched 12-block preset yet | Optional control for correlation at fixed noise width; not silently added to the queue |
| Older four-mode spatial-1D afCRPS / CRPS-FFT | Existing `*_structured_spatial1d_*` presets | Retained narrowband controls; not the multiscale replacement or an automatic launch choice |

CRPS variants retain the 256-epoch schedule (128,000 nominal updates on four
GPUs). Both FM presets use 128,000 updates, batch 256/GPU and step-cosine LR
`1e-4` to zero; this does not match runtime or independent condition exposure.
Shared evaluation remains redraw structure plus teacher-forced/free-running
VRMSE, central 90% coverage and SSR by channel/lead. Compare common afCRPS
scores across objectives, not raw afCRPS-versus-composite training losses.

The next launch review should select a bounded subset from the prepared
items rather than launch this whole inventory. Data-informed noise scales,
temporally persistent noise and a new anomaly-only supervision protocol are
not required by the initial multiscale pair.

### Prepared spatial CRPS-FFT + online-power control

The opt-in preset
`crps_fft_anomaly_spatial1d_residual_one_output_12block_256ep_8h` inherits the
IID spatial CRPS-FFT baseline. Its first two terms remain pointwise afCRPS plus
`0.1 x` Fourier-coefficient afCRPS. The new term uses the current eight-member
ensemble mean, stop-gradient on the observed-residual reference, and the
`1 + 1/8` sampling correction. It is a **fresh fit**, not a checkpoint
continuation. All data, model, normalization, optimizer and allocation settings
remain matched: 200/20/20 trajectories, one input/output frame, batch 32/GPU,
width 224, 12 blocks, patch 4, one IID noise scalar/token, no simulator-parameter
conditioning, and 256-epoch cosine LR `2e-4` to zero (128,000 nominal updates
on four GPUs). Best/last checkpoints and the common `val_afcrps` remain available.
The combined native validation objective is not directly comparable to the
baseline's native loss.

Use a few broad bands, specified before fitting, rather than tuning separate
weights around every visible peak:

| Bin | Integer Fourier radius at 64x64 | Motivation |
| --- | --- | --- |
| DC | 0 | Detect spatially uniform uncertainty separately. |
| Forcing scale | `0 < k < 6` | Contains the k=3 forcing ring and its nearby response. |
| Intermediate | `6 <= k < 14` | Separates resolved finer-scale spread. |
| Fine | `14 <= k < 22` | Contains the first k=16 patch-lattice harmonic. |
| Finest | `22 <= k <= sqrt(2)*32` | Includes diagonal harmonics and all Nyquist corners. |

The edges are divided by 64 in the config. The boundary at 22 is **not** an
exact simulator radial cutoff: its dealiasing mask acts per axis. Within-band
mode means are formed in dataset-normalized coordinates, separately for h/u/v
and each lead. Keep `power_scale=1` and avoid inverse truth-power weights, which
would amplify almost-zero simulator modes. Absolute-power discrepancies may
be dominated by energetic coarse bands; the calibration record exposes each
band's contribution. This is not a claim that relative fine-scale excess is
equally penalized.

Choose the scalar auxiliary weight from **training-only output gradients**, not
from the ratio of objective values or from held-out redraws. The helper
[`calibrate_anomaly_weight.py`](calibrate_anomaly_weight.py) uses the already
selected learned spatial CRPS-FFT control (not a nearly collapsed random
initialization), eight disjoint random batches of 32 training windows, eight
members, and fixed seed 153164. For batch i compute gradient RMS values
`b_i = RMS(dL_base/dpred)` and `a_i = RMS(dL_aux/dpred)`, then freeze
`lambda = min_i(0.1 * b_i / a_i)`. Here `L_base` includes BOTH baseline terms.
Thus the auxiliary/base output-gradient ratio is at most 10% on each calibration
batch; it is neither a 10% loss weight nor a guarantee during later training.
There is no dynamic reweighting, new teacher cache, or optimization in this step.

Preparation command for later, on the machine holding the trusted checkpoint:

```bash
uv run --frozen --no-sync python scripts/swe64s/calibrate_anomaly_weight.py \
  --source-config /path/to/spatial-fft/resolved_config.yaml \
  --checkpoint /path/to/spatial-fft/selected.ckpt \
  --data-path /path/to/shallow_water2d_stochastic_64_b43bd32 \
  --device cuda --output /path/to/new/anomaly-weight.json
```

The helper validates the source model/task, hashes the checkpoint, config,
training split and normalization, scores training windows only, and records
their indices, band powers/discrepancies and gradient ratios. It never opens
validation/test trajectories, restores optimizer state, trains, contacts W&B
or submits a job. Existing records cannot be overwritten. The output's printed
`model.loss_func.weights.2=...` override supplies the required third weight.
Save and review that record before a launch using the usual four-GPU/W&B
settings. The checked-in preset intentionally leaves this weight mandatory
(`???`); **no numerical weight has been calibrated or approved by local tests**.
The other prepared follow-up commands are unchanged.

### Matched-update residual flow matching

The current pair uses **128,000 optimizer updates** and **step-based cosine
decay from 1e-4 to zero** for both ambient and cached-latent FM. Each uses batch
256 per GPU, one input/output frame and stride one. The afCRPS controls retain
their 256-epoch schedule, which also gives 128,000 updates on four GPUs.
Neither FM preset has an additional epoch limit that binds before this budget.
Validation and best-checkpoint selection occur every 500 updates, with full
snapshots every 8,000. The trainer does not install automatic early stopping.

- `flow_matching_residual_12block_one_output_128k` trains the processor on the
  existing latent cache, using the `processor` training entry point.
- `flow_matching_residual_ambient_12block_one_output_128k` uses identity codecs
  and the original normalized raw fields, using the `epd` training entry point.
  Patch 4 on 64x64 fields and patch 1 on 16x16 latents both give 256 tokens.
  The shared transformer blocks, optimizer and checkpoint policy are matched.

Ambient FM's `model.train_in_latent_space=true` selects the processor-owned
flow objective; the identity encoder means those processor coordinates are
still normalized physical fields. Its reference is the last clean field and
its target is `y - x`. Residual statistics are fitted from training increments
in those normalized coordinates. Both presets keep outer
`model.residual_prediction=false`, since the processor adds the reference.
Only latent FM requires the fixed AE and cache; its target is `E(y) - E(x)`.
Both transport an IID Gaussian field through the learned flow, with flow time
conditioning AdaLN and no additional AdaLN noise draw.

The training loaders retain partial batches. At 64,000 windows, afCRPS has 500
full updates per epoch, while FM has 62 full updates and one half batch per
epoch (128 examples per GPU in the last batch). Thus 2,032 FM epochs would give
128,016 updates; the step limit avoids that rounding. At the exact update
budget FM reaches 2,031 full epochs plus 47 updates, with a partly completed
final epoch. No windows are dropped to force equal batch sizes.

Compose either preset locally without fitting or accessing real data:

```bash
uv run --frozen python -m autocast.scripts.train.encoder_processor_decoder \
  local_experiment=swe64s/flow_matching_residual_ambient_12block_one_output_128k \
  --cfg job --resolve

uv run --frozen python -m autocast.scripts.train.processor \
  local_experiment=swe64s/flow_matching_residual_12block_one_output_128k \
  datamodule.data_path=/path/to/shallow_water2d_stochastic_64_latents_ef93555 \
  --cfg job --resolve
```

The study evaluation wrapper reads the saved output window to keep the
100-lead horizon fixed. Ambient FM with identity codecs does not require an AE;
latent FM still requires the matching AE checkpoint. Evaluate both against raw
physical truth at matching leads, with the cached-latent source mapping and AE
hash verified when applicable. Automatic stopping patience is a separate
launch decision.

`flow_matching_residual_12block_one_output_128k.yaml` uses the merged residual FM
implementation from [PR #445](https://github.com/alan-turing-institute/autocast/pull/445).
Its training-only statistics use the canonical `residual_normalizer` interface.
It predicts the next latent's increment about the last input latent, then adds
that reference before decoding.
An IID Gaussian source is used; no additional AdaLN noise draw is introduced.
The per-example flow time conditions AdaLN, while the current latent fields
remain spatial input conditioning. Simulator-parameter conditioning stays off.

The backbone matches the current afCRPS global-256/spatial-1D pair: width 224,
12 blocks, 8 heads, FFN factor 4, modulation width 256, dropout zero, QK
normalization and relative positional bias, without RoPE or temporal mixing.
One input predicts one output at stride one. Patch 1 on the 16x16 latent grid
gives the same 256 tokens as physical 64x64 fields with patch 4. Transformer
block parameter names and shapes match exactly (10,109,376 parameters); FM's
time embedding and different input/output heads give 10,346,920 trainable
parameters overall, versus 10,231,504 for global afCRPS. The frozen AE is extra.

The budget is **128,000 optimizer updates**, batch **256 per GPU**, four GPUs,
and no accumulation. This matches the nominal 256-epoch afCRPS allocation's
updates and 32 conditions x 8 members per-GPU backbone batch. FM instead uses
256 conditions with one independent source/time draw each. It therefore sees
more distinct conditions per update; this is not matched epochs or exact
GPU-hours. At 64,000 training windows, FM has 63 updates per epoch, and its
epoch ceiling is deliberately nonbinding. Earlier afCRPS fits stopped early;
compare their actual achieved budgets, not an assumed completed 128,000 steps.
The loader retains its final half batch (128/GPU once per 63 updates), so
the match is to the nominal full-batch size, within about 0.8% in total examples.

AdamW retains FM's LR **1e-4 to zero**, betas 0.9/0.99, no weight decay or
warmup, with cosine indexed by optimizer update over the full 128,000 steps.
The 23h30m timer is only a safety cap, not the LR horizon; an interrupted fit
must resume the full checkpoint to complete the budget. Validation runs every
500 updates, snapshots every 8,000, and best-val and last checkpoints are kept.
EMA stays separately stored; main comparisons use raw weights. Training and
validation use the vector-field objective without ODE ensemble sampling;
50 Euler steps are retained for subsequent inference. FM loss and afCRPS are
different quantities: compare common physical evaluation metrics separately.

Residual mean/std are fitted once from **cached B training increments** only,
pooled spatially per lead/channel, and saved in checkpoint buffers. They are
not the three-channel physical delta statistics. The physical normalizer is
still B's original `stats.yml`, already applied by the fixed AE when caching.
Zero latent increment decodes to the current state's AE reconstruction, not
exact physical persistence; nonlinear encoding makes this a matched-backbone
model-family comparison rather than identical physical residual coordinates.

The verified cache is `shallow_water2d_stochastic_64_latents_ef93555` (UUID
`ef935550-fa55-4ddf-903b-c265e1b14c0b`): the original B/C/D 200/20/20 splits,
321 frames per trajectory, float32 `(321,16,16,8)` latents. It was produced
using the completed 1,000-trajectory AE's raw final checkpoint at step 514,048,
SHA256 `e514ebf07d4350521da8fd25c8875eb2842eb668a8b059bb207989361935573c`.
The cache's `cache_provenance.json` records source/checkpoint/statistics hashes.
The cache path is required at runtime: the preset does not select another
dataset or regenerate latents.

Preparation checks (CPU synthetic data only; no W&B run or job submission):

```bash
uv run --frozen python -m pytest -q tests/scripts/test_swe64_residual_fm.py

uv run --frozen python -m autocast.scripts.train.processor \
  local_experiment=swe64s/flow_matching_residual_12block_one_output_128k \
  datamodule.data_path=/path/to/shallow_water2d_stochastic_64_latents_ef93555 \
  --cfg job --resolve

# Preview only; launch and target-GPU preflight need separate approval.
uv run --frozen autocast processor --mode slurm --dry-run \
  --workdir /path/to/new/flow-matching-residual-128k \
  local_experiment=swe64s/flow_matching_residual_12block_one_output_128k \
  datamodule.data_path=/projects/u6eo/autocast/datasets/shallow_water2d_stochastic_64_latents_ef93555 \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:23:30:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  hydra.launcher.timeout_min=1440 \
  logging.wandb.enabled=true logging.wandb.mode=online \
  logging.wandb.entity=turing-core logging.wandb.project=autocast \
  logging.wandb.name=swe64s-fm-residual-12block-one-output-128k
```

### Global Fourier and latent residual follow-ups

The next pair reuses the presets above, without changing model defaults:

| Run | Data | Training budget | LR |
| --- | --- | --- | --- |
| Global-256 CRPS-FFT | Original raw 200/20/20 | 256 epochs; 128,000 updates on four GPUs | Epoch cosine, 2e-4 to zero |
| Latent residual FM | Cached original 200/20/20, encoded by the 1,000-trajectory AE | 128,000 updates | Step cosine, 1e-4 to zero |

Both use one input/output frame and stride one. CRPS retains batch 32/GPU and
eight members; FM uses batch 256/GPU and one source/time draw. These match the
nominal update budget and backbone batch, not condition exposure or GPU time.
The 1,000 trajectories train the frozen AE only: they do not enlarge the FM
forecasting split. No AE retraining or recaching is required.

[`followup_runs.json`](../../local_hydra/local_experiment/swe64s/followup_runs.json)
pins the run choices and the cache/AE identity. The preparation helper prints
the standard CLI commands, with four GPUs, explicit Slurm allocations, fresh
output paths and W&B enabled. It does not create outputs, contact W&B, submit
jobs, or verify remote data. The allocation ceilings remain eight hours for
CRPS and 24 hours for FM; they are not the learning-rate horizons.
Recreate the plan from the updated checkout on the target machine: its source
commit and Hydra search path refer to the checkout running the helper.

```bash
uv run --frozen python scripts/swe64s/prepare_followups.py \
  --datasets-root /projects/u6eo/autocast/datasets \
  --output-root /projects/u6eo/autocast/outputs/swe64s/global-fft-latent-fm-20261020 \
  --account brics.u6eo

uv run --frozen python -m pytest -q \
  tests/scripts/test_swe64_followup_runs.py \
  tests/scripts/test_swe64_fourier_configs.py \
  tests/scripts/test_swe64_residual_fm.py
```

Before launch, check the committed source, raw B statistics, cache provenance
and frozen AE hash on the target machine. Keep the saved training-only latent
residual statistics on resume. Evaluate final raw weights unless validation
shows clear sustained overfitting, in which case select best validation; retain
both checkpoints and report the actual achieved budget. This is a selection
rule, not an automatic early-stopping callback. Use the same conditional-redraw,
teacher-forced and free-running protocol as the current models, decoding
latent FM with the exact AE recorded in the cache.

### Original baseline preparation

Run commands from the AutoCast repository root with `uv`. Set
`AUTOCAST_DATASETS` to the existing dataset parent and `SWE64S_OUTPUTS` to an
approved scratch/output directory before training. With no output override,
Hydra uses the ignored `outputs/swe64s` directory. W&B is disabled by default.
The original baseline presets do not select a cluster launcher; the bounded
epoch profiles below do. Selecting a preset never submits jobs by itself.

The read-only helper composes all four job presets without loading the dataset
or allocating model weights. Without `--configs-only`, it checks the raw split
shapes, fixed parameters and normalization on CPU using memory-mapped tensors.
`--verify-hashes` additionally reads every split byte; `--cache /path/to/cache`
checks an existing cache's counts, first trajectory shapes and saved AE settings.

```sh
uv run --frozen python scripts/swe64s/check_inputs.py --configs-only
uv run --frozen python scripts/swe64s/check_inputs.py --verify-hashes
```

```sh
export AUTOCAST_DATASETS=/projects/u6eo/autocast/datasets
export SWE64S_OUTPUTS=/path/to/approved/scratch/swe64s

# Compose only: no dataset load or model fit.
uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps --cfg job

uv run --frozen train_autoencoder \
  local_experiment=swe64s/autoencoder --cfg job

uv run --frozen cache_latents \
  local_experiment=swe64s/cache_latents \
  autoencoder_checkpoint=/path/to/new-swe64/autoencoder.ckpt \
  cache_latents.output_dir=/path/to/new-empty-cache --cfg job

uv run --frozen python -m autocast.scripts.train.processor \
  local_experiment=swe64s/flow_matching \
  datamodule.data_path=/path/to/new-cache --cfg job
```

afCRPS and FM use the existing wall-clock cosine scheduler:
`scheduler=cosine`, `scheduler_interval=time` and zero warmup. LR decay follows
Lightning's resume-aware Timer, with a 23h30m training cap and no timing runs or
estimated epoch counts. Their finite ceiling of one million epochs only keeps
progress callbacks well-defined; elapsed time is the binding budget.

The AE retains the historical fixed schedule: `max_epochs=512`,
`scheduler_interval=epoch`, `cosine_epochs=512` and zero warmup. Its 23h30m
`max_time` is a safety cap, not the cosine horizon; if time expires early, record
the completed epochs rather than claiming the full 512. The original setting is
in [`submit_ae_large.sh`](../../slurm_scripts/comparison/ae/submit_ae_large.sh).
Learning rates remain 2e-4 for afCRPS, 1e-4 for FM and 1e-5 for the PSGD AE.

The allocation is nominally 24 hours on four GPUs per fit, leaving a 30-minute
buffer for startup and finalization. afCRPS/FM do not reuse CNS's 473/3223 epoch
estimates. The portable presets retain one-device defaults; use the four-GPU
launch settings below to match the intended compute allocation.

The study-local `trainer=swe64s_time` policy saves hourly snapshots and
`last.ckpt`, best validation loss and, when logged, overall/post-25% MultiWinkler
checkpoints. It uses existing callbacks, not a new scheduler implementation.
The post-25% window follows elapsed time. EMA remains stored separately;
the main evaluation uses raw weights.

Record batch/device count, updates, wall time, overrides, code/lock hashes,
dataset checksums and the selected checkpoint hash with each run. Evaluate raw
weights for the main comparison, even if the inherited callback stores EMA.

Fit and review the AE before caching and launching FM. Check
reconstruction error and each channel's spectrum, plus divergence/vorticity
balance: decoder artifacts must not be mistaken for FM uncertainty artifacts.
The cache must retain the raw split counts and `(321,16,16,8)` trajectory shape.
The existing cacher writes into its output directory, so use a fresh directory;
do not reuse a completed cache destination.

## Four GPU launch previews

Run these from the repository root on the target machine after environment
and data preflight. They are previews: `--dry-run` does not submit a job.
Use fresh work directories. Only remove `--dry-run` after approving launch.
The explicit time override prevents the distributed preset's 12-hour default
from replacing the study budget. The allocation is one node, four GPUs and
four tasks for up to 24 hours; the fit cap leaves a 30-minute allocation buffer
for startup and finalization.

```bash
swe64_launch_overrides=(
  '+distributed=ddp_4gpu_slurm'
  'trainer.max_time=00:23:30:00'
  '++hydra.launcher.nodes=1'
  '++hydra.launcher.cpus_per_task=72'
  'hydra.launcher.timeout_min=1440'
)

uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/afcrps" \
  local_experiment=swe64s/afcrps "${swe64_launch_overrides[@]}"

uv run --frozen autocast ae --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/autoencoder" \
  local_experiment=swe64s/autoencoder "${swe64_launch_overrides[@]}"

# Only after the matching AE has been reviewed and its fresh cache verified.
uv run --frozen autocast processor --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/flow_matching" \
  local_experiment=swe64s/flow_matching \
  datamodule.data_path="$SWE64S_OUTPUTS/cached_latents" \
  "${swe64_launch_overrides[@]}"
```

## Training pilots

Before the full allocations, run a short target-GPU smoke test of afCRPS and
the AE with the intended four-GPU model, batch, precision and data settings.
The `trainer=swe64s_pilot` profile uses a 30-minute fit cap inside a 60-minute
allocation, at most 200 optimizer updates, validation every 50 updates and four
validation batches. It keeps the model, optimizer, training data and batch size
unchanged, and enables anomaly detection. Step checkpoints are saved every 50
updates, alongside best validation loss and `last.ckpt`. Existing callbacks log
gradient norms/LR and each rank's GPU utilization; local metric plots and saved
callback histories include training/validation loss, gradients and LR. This is
a training check, not a timing calibration.

```bash
swe64_pilot_overrides=(
  'trainer=swe64s_pilot'
  '+distributed=ddp_4gpu_slurm'
  'trainer.max_time=00:00:30:00'
  'trainer.max_steps=200'
  '++hydra.launcher.nodes=1'
  '++hydra.launcher.cpus_per_task=72'
  'hydra.launcher.timeout_min=60'
)

uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/pilot_afcrps" \
  local_experiment=swe64s/afcrps "${swe64_pilot_overrides[@]}"

uv run --frozen autocast ae --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/pilot_autoencoder" \
  local_experiment=swe64s/autoencoder "${swe64_pilot_overrides[@]}"
```

Use fresh directories and remove `--dry-run` only for an approved submission.
The explicit cap/step overrides prevent the distributed/common presets from
restoring their full-run limits. Record the code/lock/dataset identity and Slurm
job ID, and check CUDA/driver visibility inside the allocated compute node, not
by trying GPU training on a login node.

Check finite losses/gradients, nonzero parameter updates, functioning DDP,
memory headroom, LR evolution, at least two validation passes and successful
checkpoint save/reload. Inspect initial versus final validation loss and a few
predictions/reconstructions, without treating a short pilot as evidence of
convergence or requiring calibrated uncertainty already. If the time cap is
reached before enough updates/validation, the pilot is inconclusive, not passed.

The shorter time cap compresses afCRPS's cosine schedule; the AE keeps its
512-epoch horizon. Start both full fits fresh rather than continuing these
diagnostic runs. FM needs its own pilot once a reviewed SWE AE and matching
cache are available; an afCRPS/AE pilot does not validate latent FM.

## Epoch-based one-output residual control

`afcrps_global_residual_one_output_256ep.yaml` starts a fresh fit on the
**original 200/20/20 trajectories**, all 321 regular frames. It inherits the
small one-output global-noise residual model: width 256, four blocks, eight
heads, patch four, 1024D global noise, eight training members and alpha-fair
CRPS with alpha 0.95. Persistence corrections use the existing one-step delta
statistics; the score is applied to the reconstructed state. Simulator-parameter
conditioning remains disabled. The model has 10,657,264 parameters. No AE,
latent cache, new data or changes to the existing normalization are required.

The binding training budget is **256 full forecasting-data epochs**, not a
fixed number of updates or elapsed hours. AdamW retains LR `2e-4`, betas
`[0.9,0.99]`, zero weight decay and no warmup, with epoch-wise cosine decay
to **zero** over `${trainer.max_epochs}`. Changing the epoch count also changes
the cosine horizon. `max_steps=-1` leaves no independent step cap.

One input and one output with stride one give 320 windows per trajectory,
64,000 training windows in total. On four GPUs at batch 32/GPU this means
500 updates per epoch, hence **128,000 updates** for the full schedule. Those
updates are derived, not a second stopping limit. Different data/window/batch
settings can change them. These are full-data epochs, unlike the AE's sampled
approximately 20% epochs; no frame or batch thinning is applied here.

The reusable `trainer=swe64s_epochs` policy validates the complete held-out
split every epoch, selects the overall best `val_loss` without a progress
window, saves snapshots every 64 epochs and maintains a full-state `last.ckpt`.
Gradient/LR and metric plots and the existing separate EMA storage are retained;
use raw weights for the main evaluation. Select the `best-val-*.ckpt` file
explicitly: the stable `encoder_processor_decoder.ckpt` export contains final
weights, not necessarily the best validation checkpoint. There is no
early-stopping callback.
Incorrect anomaly structure and late validation deterioration are separate
diagnostics; a complete cosine run does not guarantee either is corrected.

The inherited 2h30m limit is replaced by the standard **23h30m safety cap**.
At the earlier global one-output throughput, 256 epochs could take about
2--3 hours on four GPUs; this is an extrapolation, not a timing run. A fit
that hits the cap before epoch 256 is incomplete, even if it has a useful
best checkpoint. The safety cap does not accelerate or stretch the LR schedule.

The earlier `afcrps_global_residual_one_output_512ep.yaml` remains available
unchanged for reproducibility. The 256-epoch preset inherits it and changes
only the epoch count and experiment name; the cosine horizon is interpolated,
not copied as a literal. It cools earlier: at epoch 180, LR is about `4.04e-5`
rather than `1.45e-4` under the 512-epoch schedule. This is not a guarantee
against overfitting. Start the shorter fit fresh; do not resume a cancelled
512-epoch checkpoint under the shorter schedule. Existing partial outputs
should be retained separately, not overwritten or used as benchmark results.

The matched `afcrps_spatial_residual_one_output_256ep.yaml` inherits this
256-epoch preset and changes only its name, `noise_mode=spatial` and
`n_noise_input_channels=4`. Patch four gives 256 tokens, each receiving four
IID Gaussian values: 1,024 random scalars per member, matching the global
latent's total size. The same draw is reused across blocks; every prediction
call draws fresh noise. This is per patch token, not per grid cell.
The inherited modulation width remains 1,024. Its shared 4-to-1024 input
projection adds 5,120 parameters, giving 10,662,384 parameters in total.
Data, one-step residual task, eight members, batch size, LR schedule,
validation and checkpoint selection are unchanged. Start a fresh fit with
a separate output directory and W&B run; do not resume the global fit.
Per-token modulation can change throughput, so equal epochs/updates do not
imply equal wall time. Do not truncate the spatial fit to the global run's
elapsed time; record its actual cost separately.

Read-only composition and submission preview (no fit or submission):

```bash
uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps_global_residual_one_output_256ep --cfg job

uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps_spatial_residual_one_output_256ep --cfg job

uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir /path/to/new/afcrps-global-residual-one-output-256ep \
  local_experiment=swe64s/afcrps_global_residual_one_output_256ep \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:23:30:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  hydra.launcher.timeout_min=1440 \
  logging.wandb.enabled=true logging.wandb.entity=turing-core \
  logging.wandb.project=autocast \
  logging.wandb.name=swe64s-afcrps-global-residual-one-output-256ep
```

For the spatial submission preview, replace the preset with
`swe64s/afcrps_spatial_residual_one_output_256ep` and use a distinct spatial
work directory and W&B name. Keep the distributed/safety-cap overrides.

Use the approved scratch roots (`AUTOCAST_DATASETS` and `SWE64S_OUTPUTS`) and
a fresh output/W&B run. The explicit time override prevents the distributed
preset's default from replacing the safety cap. Retain the saved configuration,
code/lock/dataset identity, checkpoint hashes and actual epochs/updates/GPU-hours.
Before an eventual launch, verify the original dataset and delta statistics
on the target machine. The earlier time-bounded controls remain unchanged.

### Fresh 512-epoch matched noise comparison

The bounded `afcrps_global_residual_one_output_512ep_8h.yaml` and
`afcrps_spatial_residual_one_output_512ep_8h.yaml` presets extend this comparison
to **512 full epochs / 256,000 updates**. Both use an **eight-hour Slurm
allocation** (`timeout_min=480`) and **7h45m trainer safety cap**, leaving
15 minutes for final saving. Epochs remain the intended stopping condition;
the safety cap does not change the epoch-wise cosine from `2e-4` to zero.
Record a time-limited fit as incomplete rather than a full 512-epoch result.

These are fresh fits, not continuations of the completed 256-epoch schedule
or the earlier cancelled 512-epoch attempt. A fresh 512-epoch cosine follows
a different LR path from the outset. Preserve the earlier output directories
and W&B runs, and use new, distinct destinations for both fits. The original
512- and 256-epoch presets remain unchanged so their saved configurations
remain reproducible.

The global fit retains all scientific settings of its 256-epoch counterpart.
The spatial fit changes only the experiment name and noise layout relative
to the new global preset: four IID values per patch token, with the same
1,024 random scalars per member and inherited modulation width. Model, data,
residual task, batch 32/GPU, eight members, optimizer, validation and checkpoint
selection are otherwise matched. No AE, new dataset or normalization is needed.
The measured global 256-epoch runtime was 2h14m on four GPUs, suggesting roughly
4.5 hours for 512 epochs. Earlier spatial throughput suggests about six hours,
but was measured on a different output window. These are estimates, not timing
runs; allow both fits the same eight-hour allocation and record actual cost.

Compose and preview without training, W&B initialization or job submission:

```bash
uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps_global_residual_one_output_512ep_8h --cfg job

uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps_spatial_residual_one_output_512ep_8h --cfg job

uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir /path/to/new/afcrps-global-residual-one-output-512ep \
  local_experiment=swe64s/afcrps_global_residual_one_output_512ep_8h \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:07:45:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  logging.wandb.enabled=true logging.wandb.mode=online \
  logging.wandb.entity=turing-core logging.wandb.project=autocast \
  logging.wandb.name=swe64s-afcrps-global-residual-one-output-512ep
```

For the spatial preview, select
`swe64s/afcrps_spatial_residual_one_output_512ep_8h` and use a separate spatial
work directory and W&B name. Both presets set the eight-hour launcher default.
Keep the explicit `trainer.max_time` override when adding the distributed
preset, whose default would otherwise replace the trainer safety cap. Verify
the target checkout and original dataset/delta statistics before approving
either launch.

### Matched 256-dimensional noise on a twelve-block backbone

The new `afcrps_global256_residual_one_output_12block_512ep_8h.yaml` and
`afcrps_spatial1d_residual_one_output_12block_512ep_8h.yaml` presets retain the
main comparison's **12 blocks, eight heads, patch four and MLP expansion four**,
scaling hidden width to **224** and AdaLN conditioning width to **256**.
QK normalization, relative positional bias and zero dropout remain unchanged.
They contain **10,231,504 global / 10,232,016 spatial parameters**. The shared
spatial input projection accounts for the 512-parameter difference.

Both modes sample **256 independent Gaussian values per member and prediction
call**. The global vector is used at every token without a compression layer.
The spatial mode assigns one scalar to each of the 256 patch tokens and uses
a shared `1 -> 256` projection. Every block reuses that call's conditioning,
with separate learned modulation networks per block. This tests global versus
local noise access at equal total latent dimension. One scalar per token is a
local-conditioning choice, not a hard one-dimensional restriction on output
patches after attention mixes tokens; its adequacy remains an empirical question.

Data, one-input/one-output delta-normalized persistence residuals, eight members,
afCRPS alpha 0.95, batch 32/GPU, AdamW, seed and checkpoint policy are unchanged.
There is no simulator-parameter conditioning. The schedule is **512 full epochs /
256,000 updates on four GPUs**, with epoch-wise cosine LR **2e-4 to zero** and
no warmup. Both presets retain the **eight-hour allocation / 7h45m safety cap**.
Measured twelve-block throughput is described below; completing all 512 epochs
may require a longer allocation and a matching trainer-cap override. Equal
parameter counts do not imply the four-block fit's runtime. A time-limited fit
is incomplete.

Start both fresh with new output directories and W&B runs. Do not resume the
older four-block checkpoints: the architecture and noise dimensions differ.
The older presets remain unchanged. Comparison between this pair isolates the
noise access pattern, not the effects of every change from earlier experiments.

Read-only composition and submission preview (no fit, W&B run or submission):

```bash
uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps_global256_residual_one_output_12block_512ep_8h --cfg job

uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps_spatial1d_residual_one_output_12block_512ep_8h --cfg job

uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir /path/to/new/afcrps-global256-residual-12block-512ep \
  local_experiment=swe64s/afcrps_global256_residual_one_output_12block_512ep_8h \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:07:45:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  logging.wandb.enabled=true logging.wandb.mode=online \
  logging.wandb.entity=turing-core logging.wandb.project=autocast \
  logging.wandb.name=swe64s-afcrps-global256-residual-12block-512ep
```

For the spatial preview, select
`swe64s/afcrps_spatial1d_residual_one_output_12block_512ep_8h` with a separate
spatial work directory and W&B name. Keep the explicit distributed time override;
the eight-hour launcher default is pinned in each selected preset.

### Fresh 256-epoch controls on the twelve-block backbone

Use `afcrps_global256_residual_one_output_12block_256ep_8h.yaml` and
`afcrps_spatial1d_residual_one_output_12block_256ep_8h.yaml` to test an earlier
cosine decay on the same models. Each changes only the experiment name and
epoch horizon relative to its corresponding 512-epoch preset. The data,
normalization, one-output persistence-residual task, noise, architecture,
optimizer, validation and checkpoint settings remain unchanged.

Both use **256 full epochs / 128,000 updates on four GPUs**, with epoch-wise
cosine LR **2e-4 to zero**, no warmup, batch 32/GPU and eight ensemble members.
The 256D global vector and spatial one-scalar-per-token noise each supply
256 independent random values per member. These are the **12-block, width-224,
eight-head models with 256-wide conditioning**, not the earlier four-block
256-epoch presets. Their **eight-hour allocation / 7h45m safety cap** is retained;
the cap is a safety limit, not the intended training duration. Measured
twelve-block throughput suggests about five hours for global and six hours for
spatial, rather than the older four-block model's approximately 2h14m.

Start fresh with distinct output directories and W&B runs. Do not resume the
stopped 512-epoch fits or simply stop them at epoch 256: this control changes
the LR trajectory from the first epoch. Keep their existing checkpoints and
saved configurations intact. A time-limited fit is incomplete.

The latest 512-epoch global and spatial fits both developed validation upturns
while LR remained relatively high. This motivates testing an earlier decay;
it does not establish depth as the cause. The twelve-block setup also changed
hidden width, conditioning width and noise dimensions relative to the older
four-block model. The shorter schedule is a control, not a demonstrated fix or
an established common budget for future loss comparisons. Select best raw
validation checkpoints and assess conditional uncertainty separately.

Following Google's [tuning playbook](https://github.com/google-research/tuning_playbook#deciding-how-long-to-train-when-training-is-not-compute-bound),
use best-validation checkpoint selection, calibrate the budget before the main
comparison, then hold it fixed within that study. The interrupted 512-epoch
fits did not test the low-LR tail, so they do not exclude a later improvement.
The [training-curve guidance](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/scientific)
also distinguishes optimization from regularization. Shortening decay is not
a substitute for a regularization check if problematic overfitting persists.
Weight decay and dropout remain zero here to isolate the schedule change.

Compose and preview without training, W&B initialization or job submission:

```bash
uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps_global256_residual_one_output_12block_256ep_8h --cfg job

uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps_spatial1d_residual_one_output_12block_256ep_8h --cfg job

uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir /path/to/new/afcrps-global256-residual-12block-256ep \
  local_experiment=swe64s/afcrps_global256_residual_one_output_12block_256ep_8h \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:07:45:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  logging.wandb.enabled=true logging.wandb.mode=online \
  logging.wandb.entity=turing-core logging.wandb.project=autocast \
  logging.wandb.name=swe64s-global256-residual-12block-256ep
```

For the spatial preview, select
`swe64s/afcrps_spatial1d_residual_one_output_12block_256ep_8h` and a separate
work directory/W&B name. Keep the explicit `trainer.max_time` override when
adding `+distributed=ddp_4gpu_slurm`; its default otherwise replaces the cap.
W&B remains disabled in the presets until explicitly enabled for a launch.

## Full 1000-trajectory AE pretraining

`autoencoder_pretraining_1000.yaml` is a fresh AE fit using **all 321 frames of
1,000 new trajectories** as its sampling pool (321,000 states). Forecasting
remains on the original 200/20/20 dataset. There is no temporal thinning or
checkpoint initialization.
The model, PSGD, batch 16/GPU on four GPUs, seed 42, clipping and separately
stored EMA match the earlier fresh AE control. The selected checkpoint is best
validation loss, using raw weights for the main evaluation.

Each sampled epoch is limited to **1,004 optimizer updates**, using the existing
shuffled loader and Lightning's epoch-dependent distributed sampler. Four GPUs
at batch 16 present 64,256 distinct states per sampled epoch: approximately 20%
of the pool, close to the original 64,200-state AE epoch. Selection is a fresh
random prefix each epoch, without replacement within that epoch; states can
reappear in later epochs. This is not a fixed 20% subset, selection of only 200
trajectories, or a change to the forecasting cadence. Seeds and rank count fix
the sampling protocol. Validation/test loaders still traverse their full data.

The fit allows **512 sampled epochs / 514,048 optimizer updates**, retaining the
historical 512-epoch cosine indexed by `floor(global_step / 1004)`, base LR
`1e-5` and zero warmup. Both the trainer and stopping callback use this endpoint.
Validation is every 1,004 updates and at the endpoint. The full budget presents
32,899,072 states: approximately 102.5 passes through the new pool, not 512
passes through all 321,000 states. The final LR is zero; the selected checkpoint
remains best validation loss rather than necessarily the final checkpoint.

The trainer has a **23h30m safety cap**; allow 24 hours in a Slurm allocation,
explicitly overriding the distributed preset's shorter time default. Based on
the completed 77,308-update control, the full budget could take about 18--19
hours on four GPUs (roughly 74 GPU-hours), subject to loading/I/O and contention.
This is an extrapolation, not a timing job. If the safety cap binds, report the
completed updates/epochs rather than claiming the full cosine was completed.

The AutoSim preset `generate_data_swe_ae_pretraining_64` pins the long-window
physics, including `nu=1e-4`, `T=120`, `dt_save=0.25`, spin-up through t=40,
balanced random-PV starts, fixed amp 0.1 and white-in-time vortical k=3 forcing.
It uses seed 20280000, beyond the original and earlier additional-pool reserved
seed ranges. It produces training data only; the existing validation/test
trajectories and original `stats.yml` are reused, without recomputation.

Generation is serial in the current AutoSim CLI. The read-only planner emits
ten fixed **100-trajectory** shard commands. Each uses its own process, one
Torch/BLAS thread and disjoint split/retry seed ranges. Four simultaneous
workers are a reasonable starting point on a shared CPU host. Based on the
earlier 200-trajectory generation, the total is roughly 20 CPU-hours; four
workers could take about 5--6 hours plus I/O. This is an extrapolation, not a
new timing run. The output fields alone occupy about 15.8 GB; sharded assembly
also needs space for the merged file and a temporary mapped buffer.

```bash
# Prints JSON commands only: no directories, simulations or jobs are created.
PYTHONPATH=scripts/swe64s uv run --frozen --no-sync python \
  scripts/swe64s/prepare_ae_pretraining.py plan \
  --autosim-source /path/to/reviewed/autosim \
  --output /scratch/swe64-ae1000-shards --workers 4

# After generation and checksum verification, check one complete pool.
# For ten shards, repeat --source once per shard, in any order.
PYTHONPATH=scripts/swe64s uv run --frozen --no-sync python \
  scripts/swe64s/prepare_ae_pretraining.py prepare \
  --baseline /path/to/shallow_water2d_stochastic_64_b43bd32 \
  --source /path/to/generated-ae1000 --output /path/to/new-ae1000-view \
  --check-only
```

The planner does not launch the commands or implement a worker pool. Run the
reviewed commands with the agreed concurrency when generation is authorized.
Shard count/seeds are fixed independently of worker count; a sharded pool is
reproducible but differs from one serial 1,000-trajectory call. Record the plan
and exact generator commit with the dataset.

Removing `--check-only` assembles a new dataset directory. A single pool is
referenced directly; ten shards are merged in seed order without thinning.
Both modes link unchanged baseline validation/test data and normalization.
Preparation rejects changed physics, counts, seeds, nonfinite values, duplicate
retained starting states and existing/overlapping destinations. It hashes
source/output files and writes `ae_pretraining_manifest.json`. Never point an
AE fit at the raw generated training-only directory, whose held-out splits
are empty and whose stats describe the new pool.

```bash
# Configuration-only gate works before generation and does not start W&B.
PYTHONPATH=scripts/swe64s uv run --frozen --no-sync python \
  scripts/swe64s/check_ae_diversity.py --fresh --pool-size 1000 --configs-only

# Once assembled, verify the complete dataset and compose the intended fit.
PYTHONPATH=scripts/swe64s uv run --frozen --no-sync python \
  scripts/swe64s/check_ae_diversity.py --fresh --pool-size 1000 \
  --data /path/to/new-ae1000-view
```

The fresh preset requires `ae_diversity_data_path=/path/to/new-ae1000-view`.
Keep `scripts/swe64s` on `PYTHONPATH` for the existing LR callback. Preserve the
old 200-trajectory best AE as an ablation; each selected AE needs its own latent
cache and FM fit. The 400x160 control below remains unchanged. None of these
preparation/configuration commands submit jobs or push code.

## Fresh AE trajectory-diversity control

The selected control is a **fresh fit**, not a continuation. It asks whether
allocating a similar number of training examples across more independent
trajectories reduces early overfitting. The previously prepared continuation
was cancelled before starting; its configuration is retained below for reference.

- Original: 200 trajectories x 321 frames = 64,200 training examples.
- Fresh control: 400 trajectories x 160 frames = 64,000 examples, from the
  original 200 trajectories plus 200 new trajectories at the same physics.
- Keep original validation/test trajectories and normalization unchanged.
- Match initialization seed 42, model, loss, PSGD, clipping, batch 16 per GPU,
  four GPUs and EMA decay. Initialize weights, optimizer and EMA from scratch.
- Match the original saved endpoint: **77,308 updates from zero**. Preserve
  the original 512-epoch cosine LR using `floor(global_step / 1004)`, not the
  new loader's epoch count. Validate every 1,004 updates and at the endpoint.

The materialized, AE-only view on Isambard is
`/projects/u6eo/autocast/datasets/shallow_water2d_stochastic_64_ae400x160_d2f66a0`.
Its `valid/`, `test/` and `stats.yml` link to the unchanged baseline. The
approximately uniform frame thinning spans the original t=40--120 window;
it is not suitable as a constant-time-step forecasting dataset.

The original four-GPU job `7159768` ran for **2h47m40s** before cancellation,
with the selected last checkpoint at 77,308 updates. Expect roughly three
hours for this matched fresh fit, not twice the original duration. Its
four-hour fit cap and 4h15m allocation provide headroom; they do not set the
LR schedule. Only a checkpoint at 77,308 updates completes the comparison.

Read-only checks and submission preview (these launch nothing):

```sh
export PYTHONPATH="$PWD/scripts/swe64s:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
ae_view=/projects/u6eo/autocast/datasets/shallow_water2d_stochastic_64_ae400x160_d2f66a0

uv run --frozen python scripts/swe64s/check_ae_diversity.py --fresh --configs-only
uv run --frozen python scripts/swe64s/check_ae_diversity.py --fresh --data "$ae_view"

uv run --frozen autocast ae --mode slurm --dry-run \
  --workdir /path/to/new/ae-diversity-fresh-fit \
  local_experiment=swe64s/autoencoder_diversity_400_fresh \
  ae_diversity_data_path="$ae_view" \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:04:00:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  hydra.launcher.timeout_min=255 \
  logging.wandb.enabled=true logging.wandb.entity=turing-core \
  logging.wandb.project=autocast \
  logging.wandb.name=swe64s-ae-diversity-400-fresh
```

Use a fresh output directory and W&B run; do not supply a checkpoint, even
weights-only. The fresh preflight rejects the supported AE restore options.
Compare absolute validation error and the onset of worsening against updates,
then audit best/final checkpoints by h/u/v, simulation time and spectrum.
A smaller train--validation gap alone is insufficient if validation error
does not improve. This tests more trajectories with sparser temporal sampling
at a comparable example budget, not trajectory count in isolation or different
physics. One initialization seed is a control, not a replicated causal claim.

### Earlier continuation (not selected)

This earlier configuration is not the selected fresh-fit control. Keep the original
200/20/20 dataset byte-for-byte unchanged. Generate 200 additional training
trajectories with the same physics, then build a separate AE-only view:

- Original comparison: 200 trajectories x 321 frames = 64,200 examples.
- New view: original 200 plus additional 200 trajectories, each contributing
  160 approximately evenly spaced frames = 64,000 examples.
- Validation/test remain the original 20/20 trajectories, all 321 frames.
  Use the original training normalization, not the additional dataset's stats.

The frame selector rounds 160 positions spanning indices 0 through 320. This
keeps both endpoints without ragged trajectories; gaps are two or three saved
frames. **Use this view only for single-frame AE reconstruction, not forecasting.**
It changes training-state allocation as well as trajectory diversity, and starts
from an already fitted model; it is not a from-scratch dataset-size scaling test.

### Additional simulations (prepared, not launched)

AutoSim's `generate_data_swe_ae_diversity_64` preset starts from fetched main
`47376c8cf5695e59f82a779998634020404197b5`. It specifies 200/0/0 trajectories,
seed `20261014`, 321 frames over t=40--120, dt_save=0.25, and the original
physical parameters (including nu=1e-4 and forcing k=3/bandwidth=0.7).
The original solver and forcing source files match that main revision.
Integration remains float64 and saved fields float32. The new seed stream
does not overlap the original split/retry seed ranges.

Proposed Spark CPU destination (not yet created):
`/data/aiphys/scratch/swe-crps-issue-153/dataset-generation-64-ae-diversity-20261014/additional_200`.
From the AutoSim checkout, inspect without generating:

```sh
uv run --frozen autosim --config-name=generate_data_swe_ae_diversity_64 \
  dataset.output_dir=/data/aiphys/scratch/swe-crps-issue-153/dataset-generation-64-ae-diversity-20261014/additional_200 \
  --cfg job --resolve
```

Generation must be approved separately. `overwrite=false` refuses existing
data or metadata before simulation; Hydra logs are stored separately. Record
the actual AutoSim commit, resolved config and split hashes with the output.
Previous serial Spark timings suggest about 4 CPU-hours and 3.16 GB of new
training fields; these are estimates, not a new timing run.
Together, the additional data and AE view occupy about 6.3 GB; allow about
9.5 GB of free space while the view's temporary mapped buffer exists.

### Prepare the AE-only view and full-state checkpoint

Run the following only after the additional data have been generated and
placed on the training machine. All destinations must be fresh. The builder
checks the original file hashes, complete simulator settings, finite fields,
and duplicate retained start states across old/new training and held-out data.
It records source trajectory IDs, selected times, checksums and a verified
copy map. Validation, test and stats are symlinked to the original dataset.

```sh
export PYTHONPATH="$PWD/scripts/swe64s:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
ae_baseline=/projects/u6eo/autocast/datasets/shallow_water2d_stochastic_64_b43bd32
ae_additional=/path/to/additional_200
ae_view=/path/to/new/shallow_water2d_stochastic_64_ae400x160

# Read-only source preflight; remove --check-only only to build the new view.
uv run --frozen python scripts/swe64s/prepare_ae_diversity_data.py \
  --baseline "$ae_baseline" --additional "$ae_additional" \
  --output "$ae_view" --check-only

ae_original=/projects/u6eo/autocast/outputs/swe64s/full-20261008/autoencoder
ae_derived=/path/to/new/ae-diversity-resume.ckpt
uv run --frozen python scripts/swe64s/ae_diversity.py prepare \
  --source "$ae_original/autocast/fl7hrywf/checkpoints/best-val-0028-0.0002.ckpt" \
  --source-config "$ae_original/resolved_autoencoder_config.yaml" \
  --target "$ae_derived"
```

The trusted source checkpoint SHA256 is
`df27e12bb9dc46f44fa5d1b594d3f90b534ac6d0a0d2487eb1b6bad800f0824b`.
The preparation helper refuses another source or an existing target. It retains
raw weights, PSGD state/preconditioners and EMA, resets run-local diagnostics,
and normalizes completed-epoch bookkeeping for the new loader. The original
checkpoint is never rewritten. Do not use `prepare_ae_refinement.py` here:
that helper intentionally changes LR and would confound this control.
Lightning may warn that the cumulative step is not divisible by the new
loader length. The completed-epoch reset is deliberate; the CPU resume test
checks batch-zero restart without losing optimizer updates or EMA history.

### Matched budget and launch gate

`autoencoder_diversity_400` resumes at global step **29,116** and stops at
**77,308**: **48,192 additional updates**, matching the original final checkpoint.
Model, batch 16 per GPU, four GPUs, PSGD, clipping and seed remain matched.
The ordinary scheduler is disabled only because the study callback reproduces
the original 512-epoch cosine staircase exactly by update count:
`LR(g) = 1e-5 * (1 + cos(pi * floor(g / 1004) / 512)) / 2`.
The new loader has 1,000 updates per epoch, versus the original 1,004, so
reusing its epoch counter directly would subtly change the schedule.

Full validation runs every 1,004 batches and at the final partial-epoch
endpoint. Raw-weight validation, W&B, hourly snapshots, best-val and final
checkpoints are enabled; EMA remains separately stored. A 4h30m fit cap inside
a 4h45m allocation is only a safety limit. Previous rates imply about 1h45m to
3h40m on four GPUs (roughly 7--14.5 GPU-hours); record actual updates if the cap
is reached, and do not treat that as the complete comparison.

From the repository root, retain `PYTHONPATH` above so Hydra can import the
study-local callback. These checks and the preview submit nothing:

```sh
uv run --frozen python scripts/swe64s/check_ae_diversity.py --configs-only
# Once the data view and derived checkpoint actually exist:
uv run --frozen python scripts/swe64s/check_ae_diversity.py \
  --data "$ae_view" --checkpoint "$ae_derived"

uv run --frozen autocast ae --mode slurm --dry-run \
  --workdir /path/to/new/ae-diversity-fit \
  local_experiment=swe64s/autoencoder_diversity_400 \
  ae_diversity_data_path="$ae_view" resume_from_checkpoint="$ae_derived" \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:04:30:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  hydra.launcher.timeout_min=285
```

Before approving launch, verify the view and checkpoint manifests on the target
machine and inspect this preview. Afterwards compare raw best/final checkpoints
with the existing 200-trajectory audit by h/u/v, simulation time and spectrum.
Improvement would support limited independent training variation as a factor;
no improvement would not, by itself, establish a generator defect.

## Bounded capacity and AE refinement comparison

The separate `afcrps_large_2h` and `afcrps_small_2h` presets use a fresh
two-hour time-cosine fit at LR 2e-4. The large model retains width 568 and
12 blocks (80.85M parameters); the small model uses width 256 and four
blocks (10.69M). Both retain patch 4, eight heads, 1024D global AdaLN noise,
eight training members, zero dropout/weight decay and the original data and
one-input/four-output task. Full validation, half-hour checkpoints and
gradient/LR logging use `trainer=swe64s_short`. The original full-run presets
are unchanged. Equal wall time is a compute-budget comparison, not a pure
capacity ablation: compare curves against updates as well as elapsed time.

`autoencoder_refine` is a separate 30-minute, constant-LR 3e-6 continuation
from a selected best AE checkpoint. It preserves the architecture, optimizer
moments/preconditioners and epoch/step counters. Prepare a new full-state
checkpoint first; simply overriding the config LR would restore the old LR.
Only use trusted checkpoints. The helper refuses existing outputs, checks an
optional source SHA256 and writes source/derived hashes beside the result.
AE checkpoints need the original resolved training config supplied explicitly;
the helper checks its base LR against the checkpoint and records its hash.
Old checkpoint paths and diagnostic histories are removed; the existing timer
reset callback gives the continuation its own budget.

```sh
uv run --frozen python scripts/swe64s/prepare_ae_refinement.py \
  --source /path/to/best-ae.ckpt --target /path/to/new/resume.ckpt \
  --source-config /path/to/original/resolved_autoencoder_config.yaml \
  --expected-sha256 SELECTED_SOURCE_SHA256

uv run --frozen autocast ae --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/autoencoder_refine" \
  local_experiment=swe64s/autoencoder_refine \
  resume_from_checkpoint=/path/to/new/resume.ckpt \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:00:30:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  hydra.launcher.timeout_min=45
```

For each fresh afCRPS fit, use the same distributed overrides but
`trainer.max_time=00:02:00:00` and `hydra.launcher.timeout_min=135`.
The two 2h15 allocations, one 45-minute AE allocation and a reserved
45-minute matched evaluation allowance total at most six node-hours on
one four-GPU node per allocation. The short evaluation should be labelled
as a bounded preview, use identical test subsets/seeds and raw best-validation
checkpoints, and not be presented as the full test protocol below.

## Spatial noise control

`afcrps_spatial_small_2h` inherits `afcrps_small_2h` and changes only the
experiment name, AdaLN noise mode, input noise width and LR schedule.
Patch 4 on the 64x64 grid gives 256 tokens, each with an independent four-dimensional
Gaussian draw: 1,024 random scalars per member, matching the global baseline.
The shared 4-to-1024 projection adds 5,120 parameters (10,699,392 total).
The 1024-wide modulation layers, width 256, four blocks, afCRPS alpha 0.95,
eight members, data and parameter-conditioning settings remain the same.
Use constant LR 2e-4 (no scheduler) with a two-hour training cap to check for
overfitting while learning remains active. The earlier global runs used a
two-hour cosine schedule, so this is not a noise-only comparison against
their loss curves. One draw is reused across blocks and generates
all four output frames jointly; each prediction call draws fresh noise.

Start a fresh fit: the new input projection is absent from the global
checkpoint. Use the same data splits and validation protocol. Compare loss
curves by updates and elapsed time, since spatial modulation can change
throughput under the fixed time budget. Evaluate both the train/validation
gap and uncertainty structure; lower DC power alone could conceal an
increase in high-wavenumber anomaly power. Reuse the existing conditional
simulator redraws and report total spread, DC/high-wavenumber fractions and
anomaly divergence/vorticity balance alongside common forecast scores.

```sh
uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/afcrps_spatial_small_2h" \
  local_experiment=swe64s/afcrps_spatial_small_2h \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:02:00:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  hydra.launcher.timeout_min=135
```

This previews submission only. Use the common evaluator below with the new
run's saved config and raw best-validation checkpoint after an approved fit.
The 135-minute Slurm allocation leaves 15 minutes for setup and final saving;
the training timer itself is two hours. Save both best and last checkpoints.
Assess the loss curves and conditional uncertainty before approving a
continuation; do not automatically resume or start another fit.

## Persistence-residual control

`afcrps_global_residual_small_2h` inherits `afcrps_small_2h` and enables the
general EPD persistence-residual API from main. The processor predicts four
cumulative corrections from the same last clean observed frame. Reconstruction
adds each correction to that frame, not to the preceding predicted lead;
afCRPS, metrics and rollouts receive complete next-state fields.

Use the dataset normalizer's existing `mean_delta` and `std_delta` from
`stats.yml`, with no new statistics file or dataset generation. These are
one-step statistics reused at every lead, not lead-specific cumulative
statistics. A zero normalized correction represents the saved mean increment,
not necessarily exact persistence. Backbone initialization is unchanged.

Retain the small 10,694,272-parameter model, global 1024D AdaLN noise, patch 4,
eight members, alpha 0.95, one-input/four-output/stride-one windows, dataset,
parameter-conditioning settings and full validation protocol. Set constant
LR 2e-4 (no scheduler) and a two-hour training cap. The earlier global fit
used time-cosine scheduling: comparing against it changes both the prediction
form and scheduler, so it is not a residual-only ablation.

Start fresh, even though the parameter shapes are unchanged: the old output
weights represented states rather than increments. Keep best-validation,
30-minute snapshots and last checkpoints. Decide whether to continue only
after reviewing train/validation curves by updates and elapsed time, and the
conditional uncertainty diagnostics against the existing simulator redraws.
The older `afcrps_spatial_small_2h` control remains direct-state; it is not
the matched spatial-residual arm described below.

Use `afcrps_global_residual_small_2h30` for the extended decision pilot. It
inherits the two-hour residual preset and changes only the experiment name
and training cap to 2h30m. Constant LR 2e-4, the backbone and all scientific
settings remain unchanged. The two-hour preset is retained for provenance.
Previous effective throughput was approximately 14.6 optimizer updates/s,
giving about 130k updates in 2h30m, or 105k with a 20 percent slowdown. Check
throughput during the fit and aim for at least 100k updates before judging
late overfitting; a time cap cannot guarantee that count. No automatic
continuation is approved if the run falls short.

```sh
uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/afcrps_global_residual_small_2h30" \
  local_experiment=swe64s/afcrps_global_residual_small_2h30 \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:02:30:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  hydra.launcher.timeout_min=165
```

This command is a preview only. The 2h30m training timer is distinct from
the 165-minute Slurm allocation's setup/checkpoint buffer. No fit, evaluation
or continuation is launched by adding the preset. After an approved fit,
use its saved config with the common evaluator below so residual flags and
normalization remain intact for both teacher-forced and free-running tests.

### Matched spatial-4D residual control

`afcrps_spatial_residual_small_2h30` inherits the global two-hour-thirty-minute
residual preset. Only the experiment name, `noise_mode=spatial` and
`n_noise_input_channels=4` differ. It retains the backbone, patch 4, 1024-wide
modulation, eight members, alpha 0.95, delta statistics, one-input/four-output
task, full validation and constant LR 2e-4. Parameter conditioning stays off.

Four IID Gaussian values are sampled per patch token, not per grid cell.
At 64x64 with patch 4, 256 tokens use 1024 random scalars per member, the same
total as the global latent. A shared 4-to-1024 input projection adds 5120
parameters: 10,699,392 versus 10,694,272 for global (under 0.05 percent).
The draw is reused across blocks and generates the four leads jointly;
each new prediction call draws fresh noise. Start a fresh fit, not a global
checkpoint continuation.

```sh
uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/afcrps_spatial_residual_small_2h30" \
  local_experiment=swe64s/afcrps_spatial_residual_small_2h30 \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:02:30:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  hydra.launcher.timeout_min=165 \
  logging.wandb.enabled=true logging.wandb.mode=online
```

This remains a preview, with W&B enabled only by the explicit launch override.
The fit cap is 2h30m inside the same 165-minute allocation. Compare curves
against both updates and wall time: this is an equal-time control, not a
guaranteed equal-update comparison. Keep best-validation and last checkpoints
and inspect uncertainty against the existing redraws. No continuation is
approved automatically.

### One-output global residual control

`afcrps_global_residual_one_output_2h30` inherits the global 2h30m
residual preset and changes only its experiment name and
`datamodule.n_steps_output=1`. It predicts one persistence correction
instead of four joint corrections. The input length and stride remain one.
The same dataset, normalization and delta statistics, width-256 four-block
backbone, patch size four, global 1024D noise, eight members, afCRPS alpha
0.95 and constant LR 2e-4 are retained; parameter conditioning stays off.

The smaller output head gives 10,657,264 parameters versus 10,694,272 for
four outputs, a 0.35 percent reduction. Use a fresh fit, not a resumed
four-output checkpoint. This is an output-window control, not an exactly
equal-parameter or equal-wall-time training-budget comparison. Compare
curves by optimizer updates and training-window passes. One output creates
64,000 training windows rather than 63,400; both counts include overlapping
examples. Keep the 2h30m training cap and 165-minute allocation unchanged.

```sh
# Configuration preview only: no training, submission or W&B initialization.
uv run python -m autocast.scripts.train.encoder_processor_decoder \
  --cfg job --resolve \
  local_experiment=swe64s/afcrps_global_residual_one_output_2h30 \
  +distributed=ddp_4gpu_slurm \
  trainer.max_time=00:02:30:00 \
  hydra.launcher.timeout_min=165 \
  logging.wandb.enabled=true logging.wandb.mode=online
```

For evaluation, compare its one-step scores with the first lead of the
four-output controls, using the same targets, members and metric. Do not
compare the one-output objective directly with the four-output aggregate.
Saved configurations retain the correct one-output decoder and residual
flags for teacher-forced and free-running evaluation.

## Common evaluation

`evaluation.yaml` defines the common physical-field metric settings: 16 members,
raw weights, central 90 percent coverage through the existing `MultiCoverage`,
VRMSE and SSR with per-channel/per-lead output. Aggregate coverage-error scores
are distinct from empirical coverage; use the `coverage_0.90` trajectory columns
for the 90 percent coverage plots. The wrapper reuses the saved training config,
not a reconstructed generic architecture.

```sh
# Preview only; neither command is executed without --execute.
uv run --frozen python scripts/swe64s/evaluate.py \
  --run-dir /path/to/afcrps-run \
  --checkpoint /path/to/selected-checkpoint.ckpt \
  --output-root /path/to/new-evaluation
```

For latent FM also pass `--autoencoder-checkpoint /path/to/new-swe64/autoencoder.ckpt`.
The evaluator then restores the raw dataset and normalization from the saved
cache config and follows the existing encode-once path. Outputs use separate
`free_running` and `teacher_forced` directories; existing destinations are
rejected before execution. Teacher forcing supplies truth at each four-frame
block boundary, not at every frame inside a jointly predicted block. The
25-call cap covers 100 saved-time leads, or 25 simulator time units, on all test
trajectories. Review evaluation cost on the target GPU before executing.

The shared metric runner is not yet the complete structural evaluation.
Mean member power, mean member-anomaly power and divergence/vorticity balance
are evaluated separately against the existing 64x64 conditional simulator
redraws. Retain the reference's fixed states, generator identity and scoring
protocol when comparing another model; no new simulator draws are needed for
the spatial control. These diagnostics are required before drawing
uncertainty-structure conclusions. The generic diagnostic port is deferred.

## Environment before deployment

Use the repository's `uv.lock` and CUDA 12.6 source selection; the baseline
does not change dependencies. Check GPU driver/backend compatibility on the
target machine before syncing or launching. Record the installed AutoSim
revision separately from the dataset generator identity. Do not silently
regenerate this dataset or its redraw reference with whichever AutoSim
revision happens to import.

## Implementation sequence

1. Pin the data identity and prepare the afCRPS preset.
2. Add a new SWE autoencoder, matching latent cache and FM presets. Review
   reconstruction error, per-channel spectra and physical balance before FM.
3. Prepare common teacher-forced/free-running evaluation using the existing
   evaluator. Include VRMSE, central 90 percent coverage and SSR by lead time;
   mean member power and mean member-anomaly power remain distinct diagnostics.
4. Prepare a 64x64 conditional simulator-redraw reference and compare anomaly
   spectra and divergence/vorticity balance against it. Do not substitute the
   full state spectrum or the existing 32x32 reference.
5. Review the portable GPU environment and launch commands locally. Push and
   launch only with approval; datasets, caches, checkpoints and media stay
   outside Git.

Longer runs, spectral losses, parameter sweeps, structured-noise controls and
differentiable-physics variants are follow-on comparisons. They should not
change these baseline presets in place.
