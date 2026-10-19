# GS U-Net full-state recovery

Continue job 6675381 from its last checkpoint, at global step 182070 after
306 completed epochs. Keep the original 541-epoch cosine schedule, seed 42,
eight ensemble members, almost-fair CRPS, batch size 32 per GPU and four
GPUs/tasks. About one epoch of work after the last save must be repeated.

The original output directory is
`outputs/2026-09-18/diffusion_unet_extensions/crps_gs64_unet_azula_large_138dd8c_03eb2bf`.
Reuse this directory and W&B run `turing-core/autocast/k1d3p5ge` so existing
best-checkpoint selection and metric history remain associated with the run.

Before submission, preserve `last.ckpt`, `resolved_config.yaml`, the original
Hydra config and launch script under the run's recovery directory. Verify the
checkpoint copy with SHA-256. The backed-up checkpoint and resolved config
must share a directory. Submission records and verification results belong
there, outside Git.

Run the checked-in entry point under four Slurm tasks:

```bash
uv run --frozen --no-sync python -m slurm_scripts.ablations.resume_unet_gs \
  --checkpoint /absolute/path/to/recovery/resume_source.ckpt \
  --original-run-dir /absolute/path/to/original/run \
  --verify-only
```

`--verify-only` disables W&B, uses a separate preflight output directory and
stops before any optimizer update. Each rank checks the actual restored model,
AdamW states, scheduler, EMA, checkpoint-selection state, plot history and
remaining timer budget. All ranks must pass; any mismatch aborts the job.
The original checkpoint directory is only read during this check.

For production continuation, omit `--verify-only` and retain the original
Slurm resource settings: one node, four GPUs, four tasks, workq, and 1439
minutes. The same startup checks run before optimization. Lightning's
restored timer retains roughly 11h37m of the original training allowance;
the time budget and optimizer state are never reset.

The callback fixes resolve fractional checkpoint intervals before Lightning
matches saved state, and retain cumulative training time across jobs. Saved
epoch-duration entries are preserved. A legacy checkpoint's unclosed timing
interval remains part of cumulative elapsed time, without inventing a full
epoch duration for the incomplete interval. Lost work after the checkpoint
and restart overhead should be reported separately using the Slurm records.

Lightning may warn about the fractional callback key before running its
restore hook; the explicit startup verification checks the state after that
hook and rejects an actual restoration failure.

This recovery preserves the training schedule and saved state, but cannot
promise bitwise stochastic replay because the original checkpoint did not
save per-rank RNG states. The original rank-0 segmentation fault's underlying
cause is still unknown.
