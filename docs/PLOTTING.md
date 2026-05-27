# Plotting Evaluation Results

Use `autocast plot` to make small comparison plots from a directory containing
multiple evaluated runs.

The results directory can contain runs directly or under grouping folders. Each
run should include a `resolved_config.yaml` and one or more evaluation
subdirectories with metrics CSVs:

```text
outputs/2026-05-27/results/
  2026-05-20/
    crps_ad64_vit_large_abcd123/
      resolved_config.yaml
      eval/
        evaluation_metrics.csv
        rollout_metrics.csv
        rollout_metrics_per_timestep_channel_all.csv
    fm_cns64_unet_large_ef45678/
      resolved_config.yaml
      eval_encode_once/
        evaluation_metrics.csv
```

List the runs that would be plotted:

```bash
uv run autocast plot \
  --results-dir outputs/2026-05-27/results \
  --list
```

Generate overall metric bars and optional lead-time curves:

```bash
uv run autocast plot \
  --results-dir outputs/2026-05-27/results \
  --output-dir outputs/2026-05-27/results/plots/comparison \
  --metrics vrmse crps ssr \
  --lead-time-metrics vrmse crps
```

The command writes `selected_runs.csv` plus figures such as
`overall_vrmse.png`, `overall_crps.png`, and `lead_time_vrmse.png`.

To plot specific runs or use a non-default eval directory:

```bash
uv run autocast plot \
  --results-dir outputs/2026-05-27/results \
  --run 2026-05-20/crps_ad64_vit_large_abcd123 "CRPS large" \
  --run 2026-05-20/fm_cns64_unet_large_ef45678 "FM large" eval=eval_encode_once \
  --metrics vrmse crps
```

When discovering runs automatically, `autocast plot` selects one eval directory
per run to avoid duplicate entries. It prefers `eval/` when present; otherwise
it uses the first available `eval*` directory.
