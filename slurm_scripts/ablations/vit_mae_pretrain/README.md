# ViT MAE pretraining

Deterministic MAE pretraining run for the CNS ambient ViT baseline. The model
keeps the CRPS ViT architecture from
`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`
but trains without the ensemble path:

- `model.n_members=1`, which instantiates `EncoderProcessorDecoder` instead of
  `EncoderProcessorDecoderEnsemble`.
- `model.loss_func=torch.nn.L1Loss`, so `train_loss` and `val_loss` are MAE in
  normalized space.
- deterministic `MAE` and `RMSE` metrics replace ensemble CRPS metrics.
- `datamodule.batch_size=256` preserves the CRPS baseline's effective per-GPU
  batch size (`32 x 8 = 256`) now that there is no ensemble expansion.

**Status:** staged - run timing first, then launch the 24h production script.

## Files

| file | purpose |
|---|---|
| `local_hydra/local_experiment/ablations/vit_mae_pretrain/conditioned_navier_stokes/vit_azula_large_mae_no_ensemble.yaml` | CNS deterministic MAE preset |
| `submit_vit_mae_pretrain_timing.sh` | 5-epoch timing run -> `timing.ckpt` |
| `submit_vit_mae_pretrain_large.sh` | 24h production run, keeping and W&B-logging all progress checkpoints |

## Workflow

1. Submit timing:

   ```bash
   bash slurm_scripts/ablations/vit_mae_pretrain/submit_vit_mae_pretrain_timing.sh
   ```

2. After the timing job finishes, collect the schedule:

   ```bash
   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
   ```

3. Paste the emitted `trainer.max_epochs` value into
   `COSINE_EPOCHS_BY_DATASET` in `submit_vit_mae_pretrain_large.sh`, or leave it
   blank and let the script derive it from the newest matching timing checkpoint.

4. Submit the 24h pretraining run:

   ```bash
   bash slurm_scripts/ablations/vit_mae_pretrain/submit_vit_mae_pretrain_large.sh
   ```

The production script intentionally runs dry-run first and then submits the real
job, following the other ablation submitters.

## Checkpoints and CRPS fine-tuning

The 24h script saves local progress checkpoints every ~5% of optimizer-step
progress with `save_top_k=-1`, keeps `last.ckpt`, and sets
`logging.wandb.log_model=all` so W&B logs every checkpoint artifact emitted by
the checkpoint callbacks.

For the follow-up shortened CRPS fine-tune, start from one of these checkpoints
with `resume_weights_only=true` rather than full-state resume. That loads the
deterministic MAE weights into the CRPS ensemble model while starting a fresh
optimizer, scheduler, and time budget.
