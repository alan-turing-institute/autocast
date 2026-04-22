# ODE steps / solver (eval-only)

Eval-only ablation: re-evaluate an already-trained FM run with
different `model.processor.flow_ode_steps` and/or a different ODE
solver, without retraining.

**Status:** stub — no scripts yet.

## Source runs

FM ambient and FM cached-latent runs under `outputs/2026-04-18/diff_*`.

## Knob

- `model.processor.flow_ode_steps` — e.g. `{10, 25, 50, 100}` (baseline
  trained with 50).
- Solver family if more than one is supported (check
  `src/autocast/processors/flow_matching.py`).

## Implementation sketch

Copy `slurm_scripts/comparison/eval/submit_eval_fm_ambient.sh` and sweep
an extra dimension (ODE steps) inside the run-dir loop. Each sweep step
gets its own `eval/` subdir suffix (e.g. `eval/ode50`, `eval/ode25`) via
a non-default `hydra.sweep.dir` override — need to verify how
`autocast eval` names output dirs when the same workdir is reused.

## Outstanding decisions

- Step values.
- How to prevent the 4 eval runs from clobbering each other's
  `evaluation_metrics.csv` (likely via a csv_path override per sweep).
