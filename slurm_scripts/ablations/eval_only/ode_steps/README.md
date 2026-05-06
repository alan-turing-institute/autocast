# ODE Steps / Solver

Eval-only ablation: re-evaluate already-trained FM runs with different
`model.processor.flow_ode_steps`, without retraining.

## Script

- `submit_eval_fm_encode_once_ode_steps.sh`

## Source Runs

Main 2026-04-20 FM cached-latent runs:

- `outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331`
- `outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a`
- `outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3`
- `outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382`

## Sweep

- `model.processor.flow_ode_steps={1,5,10,25,100}`
- Baseline training and canonical eval used `flow_ode_steps=50`.
- `eval.mode=encode_once`, so metrics are computed in raw data space after
  decoding while avoiding the ambient path's per-step decode/encode loop.

Each step value writes to a separate output subdir:

- `eval_encode_once_ode001/`
- `eval_encode_once_ode005/`
- `eval_encode_once_ode010/`
- `eval_encode_once_ode025/`
- `eval_encode_once_ode100/`

The submitter also overrides `eval.csv_path` and `eval.video_dir` inside each
subdir so repeated workdir evals do not clobber each other.

To queue only a subset, set `ODE_STEPS_OVERRIDE` as a space-separated list, e.g.
`ODE_STEPS_OVERRIDE="100" bash submit_eval_fm_encode_once_ode_steps.sh`.
