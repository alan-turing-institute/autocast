# MC dropout

Replace the CRPS baseline's sampled conditional-normalization modulation with
Monte Carlo dropout in the Azula ViT feed-forward blocks. Dropout uses
`p=0.1`, remains active during evaluation, and resamples independently on
every forward call and autoregressive rollout step.

This first run deliberately excludes attention-projection dropout, stochastic
depth, and rollout-locked masks. It is the simplest Azula-native MC-dropout
comparison.

## Comparison controls

The four local experiment configs inherit the corresponding ambient CRPS
baseline. They retain:

- AlphaFair CRPS training with `n_members=8`.
- Batch size 32/GPU across four GPUs.
- Twelve transformer blocks and eight attention heads.
- AdamW with learning rate `2e-4`, no warmup, and the same validation metrics.

Removing the 1024-channel modulation projection reduces the original
`hidden_dim=568` model from approximately 80.8M to 53.3M parameters. These
configs use `hidden_dim=704`, producing approximately 79.8M parameters while
preserving depth and head count. This follows the parameter-matching approach
used by the noise-channel ablation.

## Files

| file | purpose |
|---|---|
| `../submit_planned_updates_01_timing.sh` | Five-epoch timing jobs for all four datasets |
| `../submit_planned_updates_01_large.sh` | Timing-derived 24h production jobs |
| `local_hydra/local_experiment/ablations/mc_dropout/<dataset>/crps_vit_azula_mc_dropout_large.yaml` | Dataset-specific parameter-matched experiment |

## Submission workflow

1. Run `slurm_scripts/ablations/submit_planned_updates_01_timing.sh`.
2. Retrieve the timing outputs using the command printed by that script.
3. Run `slurm_scripts/ablations/submit_planned_updates_01_large.sh`. It
   locates each timing checkpoint and derives the 24h cosine schedule with a
   2% margin.

The production script performs a dry-run submission before each real
submission, matching the main CRPS comparison workflow. Set `TRAINING_SEED`
to create an independent repeat; it defaults to 42.
