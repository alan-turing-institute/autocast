# CNS diffusion matched to flow-matching epochs

Prepare exactly one new CNS latent diffusion run, trained from scratch for
3223 epochs. This matches the main-comparison latent flow-matching run
`outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3`, which
completed 3223 epochs and 199826 optimizer steps. No new timing run is needed:
the reference epoch count, rather than a runtime estimate, sets this budget.

The experiment config is
`local_hydra/local_experiment/ablations/fm_vs_diffusion/conditioned_navier_stokes/diffusion_vit_large_epoch_matched.yaml`.
The adjacent `diffusion_cns_epoch_matched.yaml` manifest records this single
run and both reference output directories. It is separate from the six-run
diffusion/U-Net extension manifest.

## Controlled change

Relative to the original CNS diffusion run
`outputs/2026-05-01/planned_03/diff_cns64_diffusion_vit_0c75022_80967c4`,
change both `trainer.max_epochs` and `optimizer.cosine_epochs` from 2639 to
3223. Keep seed 42, the same latent cache, batch size 256 per GPU, four GPUs
and four DDP tasks, AdamW at 1e-4 without warmup, and the 704/12/8 ViT.
Keep Karras diffusion, VPSchedule and the 50-step Euler sampler.

Retain `original_diffusion_ablation`: the same 13 configured callbacks,
EMA decay 0.999, plotting, validation loss and 5% progress snapshots.
Diffusion still has `val_metrics: []`; its optional MultiWinkler and
MultiCoverage checkpoint monitors remain inactive. This matches the
diffusion callback policy; the historical flow-matching run had a different
checkpoint policy. The ablation matches epochs and optimizer steps, not
every callback or exact wall-clock time across model families.

With 62 optimizer steps per epoch, the target is 199826 updates and the
5% snapshot interval becomes 9992 updates. This follows the original
fractional cadence over the longer horizon.

## Preview only

From `~/autocast-02`, this command previews one job without submitting it:

```bash
uv run --frozen --no-sync autocast processor --mode slurm --dry-run \
  --run-group "$(date +%F)/diffusion_cns_epoch_matched" \
  local_experiment=ablations/fm_vs_diffusion/conditioned_navier_stokes/diffusion_vit_large_epoch_matched \
  processor@model.processor=diffusion_vit \
  datamodule=cached_latents \
  datamodule.data_path=/lus/lfs1aip2/projects/u6eo/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/cached_latents \
  logging.wandb.name=latent_diffusion_cns_epoch_matched \
  hydra.launcher.timeout_min=1439
```

The future output group is `outputs/YYYY-MM-DD/diffusion_cns_epoch_matched/`,
using the actual launch date. The workflow generates a CNS diffusion run
name containing the source commit and a unique suffix. Do not supply an
existing work directory or resume checkpoint. No launch is part of this
preparation.

For later comparison, use the final raw `processor.ckpt` and the original
`eval_encode_once` protocol: the same published CNS autoencoder, 10 ensemble
members and evaluation settings as the original diffusion result. Training
keeps `output.skip_test: true`; evaluation is a separate step.
