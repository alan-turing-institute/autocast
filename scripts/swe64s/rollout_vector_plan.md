# SWE64 rollout and vector-score batch

The first batch contains six independent fits. Structured noise and detached
feedback experiments follow later.

| ID | Noise | Training | Initialization | Readiness |
| --- | --- | --- | --- | --- |
| G1 | Global 256D | One-step CRPS-FFT control | Global best, 59k | One-step preset prepared; shared validation pending |
| G2 | Global 256D | Two-step full gradient, average both lead losses | Same as G1 | Training-rollout API and preset pending |
| S1 | IID spatial 1D/token | One-step CRPS-FFT control | Spatial best, 36k | One-step preset prepared; shared validation pending |
| S2 | IID spatial 1D/token | Two-step full gradient, average both lead losses | Same as S1 | Training-rollout API and preset pending |
| V1 | IID spatial 1D/token | CRPS-FFT + local joint wind score | Fresh | Vector score, scaling and coefficient pending |
| V2 | IID spatial 1D/token | CRPS-FFT + rotational/divergent velocity scores | Same fresh seed as V1 | Helmholtz transform and coefficient pending |

## Fixed protocol

Keep the original 200/20/20 dataset, training statistics, width 224, 12 blocks,
eight heads, patch four, conditioning width 256, eight ensemble members,
batch 32 per GPU on four GPUs and one-step persistence residual outputs.
Simulator-parameter conditioning stays disabled. CRPS-FFT weights are [1, 0.1]
with alpha 0.95. Two-step training means two model calls, not two output frames
from one call. Global and spatial starts differ in achieved training budget;
the continuation comparison is paired within each noise type.

G/S runs load raw weights only and start fresh AdamW/scheduler counters for
32,000 new updates, one quarter of the original planned 128,000 updates.
Use LR 2e-5, step-based cosine to zero and 100-update warmup. Reuse the
existing tested scheduler convention: warmup multiplies the cosine curve
over the whole 32k horizon. Initial LR is 2e-5/101; the near-peak at update
100 is 1.9999518e-5, and LR is zero at 32k. This is not a separate 31,900-step
cosine segment. Preserve best, latest/final and 8k/16k intermediate snapshots.

The one-step presets are `crps_fft_global256_finetune_32k` and
`crps_fft_spatial1d_finetune_32k`, under `local_experiment=swe64s/`.
They require an explicit `resume_from_checkpoint`. They are not yet ready
for the matched batch: both controls must acquire the same eligible two-future
windows and predicted-feedback validation used by G2/S2.
The four-GPU distributed preset overrides the trainer cap to 12 hours; future
launch commands must explicitly set `trainer.max_time=00:07:45:00` to preserve
the prepared safety margin within the eight-hour allocation.

V1/V2 train from scratch for at most 256 epochs, cosine LR 2e-4 to zero.
Both retain the same base CRPS-FFT and use an ordinary local bivariate
ensemble score with alpha 0.95. V2 averages the rotational/divergent scores.
Use the same fixed auxiliary coefficient, selected by a bounded training-only
gradient check. Its value is not yet set. Restore physical u/v scaling before
projection and then divide by one common training-derived wind scale.
Match DC/harmonic treatment between the two auxiliary objectives before
calibration; do not independently standardize rot/div components.

## Starting checkpoints

Start the global pair from the best 59k checkpoint and the spatial pair
from the best 36k checkpoint. Verify each file against its launch manifest
and give both arms in each pair the same raw weights.

## Implementation sequence

1. **Training rollout and validation.** Extend
   `src/autocast/models/encoder_processor_decoder.py` and the ensemble model
   with separate train/validation horizons, lead weights and optional detached
   feedback. Keep the one-step default unchanged. Separate target-window length
   from model output shape in `scripts/setup.py` and decoder configuration.
   Expand members once, retain their paths, redraw noise per call and feed back
   normalized reconstructed states. Avoid the inference rollout's extra probe
   call. Add G2/S2 presets and update G1/S1 to the common target window.
2. **Vector scoring.** Add focused vector/Helmholtz transforms alongside the
   existing loss decomposition API, reusing `CoefficientCRPSLoss` for the
   ordinary two-vector distance score. Specify DC/Nyquist conventions and
   frozen training-stat scaling. Add V1/V2 configs after both score definitions
   and their common coefficient are fixed.
3. **Tests.** Extend persistence-residual, ensemble, optimizer and config tests.
   Verify one-step equivalence, two calls without member expansion, full and
   detached gradient paths, target alignment, fresh optimizer restoration and
   scheduler endpoints. Verify vector-score equivalence, pure rotational and
   divergent Fourier modes, projection/reconstruction identities, DC/Nyquist
   behavior, physical units and finite gradients. Use `uv run` throughout.
4. **Batch readiness.** Compose all six configs; run bounded memory/throughput
   checks, freeze validation selection/stopping rules and verify data/checkpoint
   hashes. Assign independent output folders and manifests. Set resource caps
   from measured throughput; inherited eight-hour ceilings are not estimates.
   The six fits can then be submitted independently after launch approval.

## Selection and evaluation

For G1/G2/S1/S2, select using the same validation average of lead-one and
lead-two CRPS-FFT under predicted feedback. For V1/V2 use the unchanged
one-step CRPS-FFT validation score, excluding the auxiliary term. Keep native
component losses for diagnostics and retain fixed-budget endpoints.

Reuse conditional redraws, teacher-forced and free-running evaluation:
member and anomaly power, DC, divergence/vorticity, common forecast scores,
coverage, SSR and lead curves. Never calibrate the loss on held-out redraws.
Report updates and GPU time separately because model calls per update differ.

Later choices: two-step detached feedback with both leads scored, then
second-lead-only supervision; multiscale afCRPS and a matched IID-4D control.
