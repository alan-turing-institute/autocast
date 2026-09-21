# Four-dataset extension plots

Run these commands from the `autocast-runs` repository root. The six new
training runs are linked into the existing comparison collection; the main
flow-matching runs already contain their EMA evaluation directories.
The setup script validates all targets and refuses to replace existing paths.

```bash
bash scripts/link_extension_results.sh
export PLOTS_PATH=2026-09-21_final_plots
mkdir "outputs/2026-07-24_collated/$PLOTS_PATH"
bash scripts/plots_final_results.sh --paper-only
```

Use a fresh `PLOTS_PATH` for a new collection. The default is now
`2026-09-21_final_plots`; earlier figures remain in their dated directories.
`--paper-only` generates all paper layouts and result tables. Use
`--paper-figures` instead to also generate the standard diagnostic plots.
The script uses the existing environment with `uv run --frozen --no-sync`.

All four extension comparisons cover AD, CNS, GS and GPE, in that order:

| Output directory | Comparison | Appearance |
| --- | --- | --- |
| `ablation_vit_unet_m8` | Ambient CRPS: ViT vs U-Net | Existing blue vs green |
| `ablation_dm_latent` | Latent flow matching vs diffusion | Existing orange vs purple |
| `ablation_fm_ema` | Raw vs EMA latent flow-matching weights | Orange solid raw, green dashed EMA |
| `ablation_fcrps_afcrps` | fCRPS vs alpha-fair CRPS training loss | Existing green fCRPS vs blue alpha-fair CRPS |

Each comparison produces `paper_four_ds_ablation.png` and `.pdf` when paper
figures are enabled, plus the usual single-step and rollout summary tables.
They are also included in the shared `paper_figures/{png,pdf,tables}` collection.
The script checks all required metric and coverage exports before plotting
each new comparison, and reports missing files rather than silently drawing
an incomplete four-dataset comparison.

## Additional paper exports

- `ablation_fcrps_vs_main_crps` is a clearly named copy of the four-dataset
  fCRPS versus main-CRPS loss comparison, with legend labels `CRPS (main, αfCRPS loss)`
  and `CRPS (fCRPS loss)`. Main CRPS is trained with `AlphaFairCRPSLoss`.
  Only these two methods are included; the existing loss folders are retained.
- `reviewer_training_data_published_test_comparison_mean_only` contains the
  five paper layouts for original versus alternate-training-data CRPS and
  latent FM models, all evaluated on the original published test datasets.
  It uses the same trajectory statistics and mean curves as
  `reviewer_training_data_published_test_comparison`, with shaded regions
  hidden via `--no-error-bands`. Table standard errors remain available.
  This differs from the older `reviewer_training_data_comparison_mean_only`,
  which uses the earlier evaluation campaign rather than consistently using
  the original test datasets.

Both exports are collected in `paper_figures/png`, `paper_figures/pdf` and
`paper_figures/tables`, with the output directory name as the filename prefix.

## fCRPS versus alpha-fair CRPS

`ablation_fcrps_afcrps` compares only the fCRPS and alpha-fair CRPS training
losses across all four datasets. The existing `ablation_crps_variants` and
`ablation_cns_crps_variants` directories and launcher blocks are preserved,
including the ordinary-CRPS training-loss result available only for CNS.
The new comparison keeps the same evaluation metrics as the other ablations.

The saved training and evaluation configs confirm `FairCRPSLoss` versus
`AlphaFairCRPSLoss`, eight training members and ten evaluation members.
The datamodule and evaluation settings match within each dataset pair,
except for checkpoint and output paths. All eight evaluations have the
single-step, rollout, per-timestep and coverage CSVs required by the plots.
The existing run symlinks already use relative `../...` targets.

All evaluation directories are named `eval_best_multiwinkler_from0p25`,
but the saved evaluation configs identify these actual fCRPS checkpoints:

| Dataset | fCRPS checkpoint selection | Epoch in filename |
| --- | --- | --- |
| AD | `best-multiwinkler-overall` | 0422 |
| CNS | `best-multiwinkler-from0p25` | 0277 |
| GS | `best-multiwinkler-overall` | 0353 |
| GPE | `best-multiwinkler-pre0p25` | 0095 |

The alpha-fair CRPS checkpoints use `best-multiwinkler-from0p25` for all
four datasets. These plots retain the existing evaluated selections;
they do not establish identical checkpoint-selection windows.

## Evaluation selection

- ViT uses `eval_best_multiwinkler_from0p25` for all four datasets. The original
  CNS U-Net also retains `eval_best_multiwinkler_from0p25`; the three U-Net
  extensions use `eval_best_multiwinkler_overall`. These selection windows
  differ and should be stated when interpreting this comparison.
- Diffusion uses `eval_encode_once`. CNS deliberately retains the original
  `diff_cns64_diffusion_vit_0c75022_80967c4` result. The new epoch-matched CNS
  run is excluded: its evaluation is pending and will form a separate
  one-dataset ablation later.
- Flow-matching raw/Euler-50 curves use the same `eval` results as the main
  comparison. EMA uses the historical `eval_encode_once_ema` evaluation of
  the EMA weights corresponding to the same selected checkpoint.


No training, checkpoint selection, evaluation metrics or evaluation outputs
are changed by these plotting commands.

The plotting CLI also accepts per-series styles, for example
`--run RUN "FM (EMA)" 2 eval=eval_encode_once_ema linestyle=dashed`.
Styles are keyed by evaluation-specific run reference, so choosing a style
for EMA does not change the raw-weight curve from the same training run.
