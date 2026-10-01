# Diffusion and U-Net extension evaluations

Prepare seven evaluations from `diffusion_unet_evals.yaml`: AD, GS and GPE
for both model families, plus the CNS epoch-matched diffusion run. The
launcher previews by default and never evaluates an unfinished training run.

The manifest pins the evaluation settings from the original CNS diffusion
`eval_encode_once/resolved_eval_config.yaml` and U-Net
`eval_best_multiwinkler_from0p25/resolved_eval_config.yaml`. Their scientific
evaluation settings agree except for batch size and evaluation mode.

- Diffusion: final raw `processor.ckpt`, `encode_once`, the matching published
  autoencoder, batch size 4, and the trained 50-step Euler sampler.
- CRPS U-Net: overall-best MultiWinkler checkpoint, `ambient`, batch size 8.
  This uses the overall tracker requested for the extensions; the original
  CNS result used its from-25% tracker. No claim is made that those selection
  windows are identical.
- Both: 10 ensemble members, original 25-metric list including AFCRPS and
  energy, automatic coverage, full test/rollout loaders, 25 rollout blocks,
  the original metric windows and snapshot indices, and original inference
  benchmarks (5 warmups and 50 measurements). The seed configuration is 42
  and float32 matmul precision is high. EMA evaluation is disabled.
- Every job: one GPU, one task, 72 CPUs, one node, `--mem=115000M`, without an
  exclusive-node request. Evaluation also explicitly uses `eval.devices=1`.
  This matches the workq defaults per GPU and requests a quarter of the
  node's 4 GPUs, 288 CPUs and 460000M configured memory. Slurm expands `115G`
  to 117760M, which exceeds a quarter; therefore use 115000M explicitly.
  Host memory allocation is separate from GPU memory; actual usage can be lower.
- Time limits remain 360 minutes for diffusion and 240 minutes for U-Net,
  as in the original ablation evaluation scripts.

Outputs stay beside each training run, under `eval_encode_once` or
`eval_best_multiwinkler_overall`, preserving its original date prefix.

## Checkpoint review

Use the final diffusion checkpoint unless the validation history shows
overfitting. An earlier single-epoch loss minimum is insufficient evidence
because the diffusion validation loss is stochastic. Review the history,
then change the individual manifest row from `checkpoint: last` to
`checkpoint: best_val` when justified. Both selections use raw weights.

The completed AD and GS runs continue improving late; GPE reaches a plateau
without a clear sustained late rise. Their prepared selections are `last`.
Review CNS again after it completes all 3223 epochs / 199826 updates.
The final-checkpoint completion check applies even when selecting an earlier
best checkpoint. A still-running `processor.ckpt` symlink does not establish
completion.

## Preview and later submission

From `~/autocast-02`:

```bash
uv run --frozen --no-sync python -m slurm_scripts.ablations.submit_diffusion_unet_evals
```

After review and committing the prepared files, the six completed runs can
be submitted explicitly:

```bash
uv run --frozen --no-sync python -m slurm_scripts.ablations.submit_diffusion_unet_evals \
  --run-id latent_diffusion_ad latent_diffusion_gs latent_diffusion_gpe \
  unet_m8_crps_ad unet_m8_crps_gs unet_m8_crps_gpe --submit
```

Once CNS training and its validation-history review are complete:

```bash
uv run --frozen --no-sync python -m slurm_scripts.ablations.submit_diffusion_unet_evals \
  --run-id latent_diffusion_cns_epoch_matched --submit
```

The launcher validates checkpoint completion, cache/autoencoder identity,
active training jobs, existing evaluation outputs and a clean checkout before
submission. It checks every selected run before submitting any. Existing
results are never overwritten automatically. Running it without `--submit`
does not allocate GPUs, start inference or submit jobs.
