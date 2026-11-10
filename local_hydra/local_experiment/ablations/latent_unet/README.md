# Matched latent U-Net generative controls

These draft controls retain the primary comparison's latent-space FM and
diffusion formulations, original datasets and frozen autoencoders. Only the
generative backbone changes from ViT to an Azula U-Net. Compare with the existing
ambient CRPS U-Nets under `ablations/arch_unet_fno_vit/` on the same test cases.

Configs cover advection-diffusion, conditioned Navier-Stokes, Gray-Scott and
GPE laser wake. Each has `fm_unet_matched` and `diffusion_unet_matched` overrides
under `ablations/latent_unet/<dataset>/`.

## Architecture

The four-level U-Net uses widths `[85,170,340,680]`, three blocks per level,
FFN factor one, LayerNorm, no dropout, no attention and nonperiodic boundaries.
The structural settings follow the CRPS U-Net control. Generative time modulation
retains width 256; the hidden widths match parameter count rather than copying
the CRPS widths, whose 1024-dimensional noise modulation contributes substantial
capacity. For eight latent channels, four output frames and four scalar
conditions, this is 80,901,221 parameters versus 81,289,744 for the reference
Gray-Scott CRPS U-Net (a 0.48% difference). Configuration/CPU checks compare each
system against its own CRPS processor, including its raw fields and conditioning,
and enforce a parameter-count difference below 1%.

| System | Latent FM/diffusion backbone | Ambient CRPS processor |
| --- | ---: | ---: |
| AD | 80,900,709 | 81,285,834 |
| CNS | 80,901,221 | 81,292,538 |
| GS | 80,901,221 | 81,289,744 |
| GPE | 80,906,341 | 81,300,904 |

The published caches have 16×16 grids with eight latent channels. Scalar
conditioning dimensions are 2 for AD, 4 for CNS and GS, and 24 for GPE. CPU
forward checks cover all four layouts; these dimensions are inferred from the
selected cache during model setup.

FM retains the reference flow objective and 50-step Euler sampler. Diffusion
retains its reference Karras denoiser, schedule and 50-step Euler sampler.
Both retain the reference optimizer, conditioning, seed, batch size and frozen
AE cache. Evaluate decoded forecasts with each model's own AE and training
normalization, using the reference ensemble size, metric windows and rollout
settings.

## Inputs and budgets

Set `AUTOCAST_UNET_<SYSTEM>_CACHE` to the original cache, with system identifiers
`AD`, `CNS`, `GS` and `GPE`. Do not use the enlarged Gray-Scott data-size cache.

Set `AUTOCAST_UNET_<METHOD>_<SYSTEM>_EPOCHS` to the validated 24-hour epoch
horizon for that formulation, backbone and dataset. Method identifiers are
`FM` and `DIFFUSION`. These required values prevent an unmeasured U-Net horizon
from silently inheriting ViT timing. The same horizon drives the cosine schedule.
The existing CRPS U-Net timings cannot establish latent generative throughput.

Diffusion pins the original diffusion-ablation trainer policy. FM inherits the
main-comparison trainer. The production preview helper sets its checkpoint
cadence to `max(1, max_epochs // 4)`, with `save_top_k=-1` and a
`quarter-{epoch:04d}` filename as in the reference FM launcher. Direct use of
the draft FM configs inherits the five-epoch default until these overrides
are supplied. Both policies retain the final checkpoint and EMA tracking;
the comparison evaluates final raw processor weights, with EMA disabled.

## Preview commands

From the checkout root, with the four cache variables exported:

```bash
# Print five-epoch timing commands; production budgets are not needed.
bash slurm_scripts/ablations/preview_latent_unet.sh timing

# After setting all eight measured epoch budgets, print production commands.
bash slurm_scripts/ablations/preview_latent_unet.sh train
```

Both modes always pass `--dry-run`; neither can submit a job. Timing uses the
existing `autocast time-epochs --kind processor` workflow with five epochs,
a 24-hour target and 2% margin. Its allocation preview is four hours.
Production requests one node, four GPUs/tasks and a 24-hour allocation, with
the trainer capped at 23h59m. `RUN_GROUP` can override the dated output group;
the eight W&B names distinguish method and system. Timing disables W&B.
The workflow's dry-run output prints training commands, not complete sbatch
scripts; the tests separately check the generated sbatch resource arguments.

Before eventual submission, revalidate cache/AE identity with the existing
`validate_cached_latents_against_ae` helper, inspect resolved configs and
callbacks, and regenerate previews in the final checkout/environment.

For example, after supplying the cache and budget variables:

```bash
uv run python -m autocast.scripts.train.processor \
  local_experiment=ablations/latent_unet/gray_scott/fm_unet_matched \
  --cfg job --resolve
```

These configs do not submit jobs. Budget evidence, cache identities and full
parameter/shape checks remain prerequisites for production submission. This
control preserves the latent/ambient comparison and does not isolate training
space as a separate factor.

## Decoded raw-test evaluation

After a completed production run, use its saved resolved config, final
`processor.ckpt` and the original cache's sibling `autoencoder.ckpt`.
The encode-once evaluation path restores the AE's raw datamodule, channel
selection and normalization from its saved config. Keep each system's original
test cohort. Do not substitute the enlarged Gray-Scott data or a different AE.

For each of the eight run directories, prepare a preview as follows. Set
`RUN_DIR` to that completed run and `ORIGINAL_AE_CHECKPOINT` to its verified AE:

```bash
uv run --frozen --no-sync autocast eval --mode slurm --dry-run \
  --workdir "${RUN_DIR}" --output-subdir eval_encode_once \
  eval.checkpoint=processor.ckpt \
  ++autoencoder_checkpoint="${ORIGINAL_AE_CHECKPOINT}" \
  ++eval.mode=encode_once ++eval.use_ema=false \
  ++eval.n_members=10 ++eval.batch_size=4 ++eval.devices=1 \
  ++eval.latent_space_metrics=false ++eval.free_running_only=true \
  ++eval.max_rollout_steps=25 \
  ++eval.max_test_batches=null ++eval.max_rollout_batches=null \
  ++eval.trajectory_statistics.enabled=true \
  ++eval.trajectory_statistics.overwrite_existing=false \
  ++eval.trajectory_statistics.sampling_seed=42 \
  ++eval.trajectory_statistics.include_per_timestep=true \
  '++eval.metric_windows=[null]' \
  '++eval.metric_windows_rollout=[[0,1],[0,4],[6,12],[13,30],[31,99]]' \
  '++eval.metrics=[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]' \
  hydra.launcher.additional_parameters.nodes=1 \
  hydra.launcher.gpus_per_node=1 hydra.launcher.tasks_per_node=1 \
  hydra.launcher.timeout_min=360
```

This preserves the reference ten-member, 25-step, decoded metric protocol.
Trajectory statistics provide per-case outputs and fixed evaluation sampling
seeds, matching the data-size controls. Re-evaluate comparison checkpoints with
the same settings when paired statistics are required.
Compare against the established validation-selected CRPS U-Net weights, using
their own normalization and the same test cases. Evaluation previews become
concrete once production run directories and completed checkpoints exist.
