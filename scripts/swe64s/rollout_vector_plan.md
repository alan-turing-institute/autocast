# SWE64 rollout and vector-score batch

The first batch contains six independent fits. Structured noise and detached
feedback experiments follow later.

| ID | Noise | Training | Initialization | Readiness |
| --- | --- | --- | --- | --- |
| R1 / G1 | Global 256D | One-step CRPS-FFT control | Global best, 59k | Locally tested |
| R2 / G2 | Global 256D | Two-step full gradient, average both lead losses | Same as G1 | Locally tested |
| R3 / S1 | IID spatial 1D/token | One-step CRPS-FFT control | Spatial best, 36k | Locally tested |
| R4 / S2 | IID spatial 1D/token | Two-step full gradient, average both lead losses | Same as S1 | Locally tested |
| V1 | IID spatial 1D/token | CRPS-FFT + local joint wind score | Fresh | Locally tested; coefficient check pending |
| V2 | IID spatial 1D/token | CRPS-FFT + solenoidal/divergent velocity scores | Same fresh seed as V1 | Locally tested; coefficient check pending |

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
They require an explicit `resume_from_checkpoint`. The two-step presets are
`crps_fft_global256_rollout2_finetune_32k` and
`crps_fft_spatial1d_rollout2_finetune_32k`. All four use the same eligible
two-future windows and two-lead predicted-feedback validation. One-step arms
score only the first lead during training; two-step arms average both leads.
The four-GPU distributed preset overrides the trainer cap to 12 hours; future
launch commands must explicitly set `trainer.max_time=00:07:45:00` to preserve
the prepared safety margin within the eight-hour allocation.

V1/V2 train from scratch for at most 256 epochs, cosine LR 2e-4 to zero.
Both retain the same base CRPS-FFT and use an ordinary local bivariate
ensemble score with alpha 0.95. V2 averages the rotational/divergent scores.
Use the same fixed auxiliary coefficient, selected by a bounded training-only
gradient check. Its value is not yet set. Restore physical u/v scaling before
projection and then divide by one common training-derived wind scale.
Do not independently standardize components. The solenoidal component retains
DC and joint derivative-null Nyquist modes under the real pseudospectral
derivative convention. No modes are discarded. V2 averages its two scores, so
a purely solenoidal/divergent error receives half the V1 auxiliary score.
One shared coefficient therefore does not equalize auxiliary gradient strength.

V1/V2 presets are `crps_fft_joint_velocity_spatial1d_256ep` and
`crps_fft_helmholtz_velocity_spatial1d_256ep`. Both require `vector_weight=...`;
there is no uncalibrated default. Use the same coefficient in both. W&B is
enabled in offline mode. All six retain one-output prediction heads.

## Overnight stopping

V1/V2 use `SustainedValidationRise` on base `val_loss`: after 10k updates,
stop gracefully if ten consecutive validations exceed the best by more than
1%. Validation is once per epoch (500 updates here). Returning inside the band
resets the count; a plateau alone does not trigger. Nonfinite validation scores
stop immediately. This does not add a per-training-batch numerical guard.
Thresholds are configurable in `trainer/swe64s_vector.yaml`.

Preserve full best/latest checkpoints every validation, 16-epoch snapshots,
and terminal `model.ckpt`, including optimizer, scheduler, EMA and callback
state. This is a compute-saving stop for review, not proof that later lower-LR
recovery is impossible. A full resume retains the stopping streak: deliberately
adjust or disable the callback after review if continuing. Do not describe a
nonfinite terminal checkpoint as healthy. R1--R4 retain their fixed 32k budget.

## Starting checkpoints

Start the global pair from the best 59k checkpoint and the spatial pair
from the best 36k checkpoint. Verify each file against its launch manifest
and give both arms in each pair the same raw weights.

## Implementation and remaining preflight

1. **Training rollout and validation: implemented.**
   `src/autocast/models/encoder_processor_decoder.py` and the ensemble model
   now support separate train/validation horizons and optional detached
   feedback, with unchanged one-step defaults. `src/autocast/scripts/setup.py`
   separates target-window length from the decoder output shape. Members expand
   once, retain their paths, redraw noise per call and feed back normalized
   reconstructed states. No extra probe call is made. The initial pair scores
   both leads equally; second-lead-only supervision is not part of this batch.
2. **Vector scoring: implemented.** `src/autocast/losses/vector.py` provides
   the velocity score and periodic Helmholtz transform, reusing the existing
   `CoefficientCRPSLoss`. Training-stat scaling and DC/Nyquist behavior are
   explicit and covered by tests. Both objectives compare members with the
   observed future, without subtracting the ensemble mean or scoring powers.
3. **Local validation: passed.** Tests cover one-step RNG/checkpoint
   compatibility, full/detached feedback, one member expansion, target alignment,
   actual tiny-ViT setup/backpropagation, fresh optimizer restoration, scheduler
   endpoints, vector distances, pure modes, projection identities, DC/Nyquist,
   scaling and gradients. Callback tests include a graceful Lightning stop with
   best/latest checkpoints and preserved AdamW/scheduler/EMA state.
4. **Vector coefficient: target-machine check pending.** On a bounded compute
   allocation, run `uv run --no-sync python scripts/swe64s/calibrate_vector_weight.py
   --data-path <raw-dataset> --output <new-json> --device cuda`. It uses four
   fresh-model training batches, verifies dataset/scaling, and chooses a shared
   weight limiting each auxiliary's initial output-gradient RMS to at most 10%
   of the base on those draws. It performs no training and opens no held-out
   splits. Apply its printed `vector_weight=...` to both V1/V2; this is not a
   guarantee of later stability or parameter-gradient balance.
5. **Batch readiness: target-machine checks pending.** Verify source/data and
   starting checkpoint hashes, and measure full-batch/member two-step GPU memory
   and throughput with a tiny bounded check. Use separate new output folders
   and manifests; preview standard `autocast epd --mode slurm --dry-run` commands
   with offline W&B, explicit trainer cap and source provenance. Set resource
   caps from measured throughput: eight-hour ceilings are not runtime estimates.
   Review the commands before approving launch. No remote changes, push or jobs
   are performed by this local preparation.

## Selection and evaluation

For G1/G2/S1/S2, select using the same validation average of lead-one and
lead-two CRPS-FFT under predicted feedback. For V1/V2 use the unchanged
one-step CRPS-FFT validation score, excluding the auxiliary term. Keep native
objectives for diagnostics (`train_loss` and `val_objective`) alongside
`train_base_loss` and `val_loss`, without additional model draws. Retain the
terminal checkpoints and report actual budgets when early stopping intervenes.

Reuse conditional redraws, teacher-forced and free-running evaluation:
member and anomaly power, DC, divergence/vorticity, common forecast scores,
coverage, SSR and lead curves. Never calibrate the loss on held-out redraws.
Report updates and GPU time separately because model calls per update differ.

Later choices: two-step detached feedback with both leads scored, then
second-lead-only supervision; multiscale afCRPS and a matched IID-4D control.
