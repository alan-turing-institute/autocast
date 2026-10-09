# Weights & Biases logging

AutoCast contains built-in support for logging to Weights & Biases (W&B), a popular tool for tracking machine learning experiments.
This allows you to monitor your training progress and visualise metrics as your experiment is running (or after it has completed).

To enable W&B logging you need to first log in with the `wandb` command-line app:

```bash
uv run wandb login
```

Follow the instructions there to log in to your W&B account.

Then, apply the `++logging.wandb.enabled=false` override when training a model.
You can additionally also supply

- `++logging.wandb.project=<project_name>` to specify the W&B project name (defaults to `autocast`)
- `++logging.wandb.name=<run_name>` to specify the run name

By default the run name is inferred from the `--run-id` command-line option, so you can also set that if you prefer:

```bash
uv run autocast ... --run-id=my_id ...
```

Since `--run-id` also controls the output directory for AutoCast (see [the Output paths page](./output-paths.md) for more information), this allows you to use a single flag to set both the output directory as well as the W&B run name.
