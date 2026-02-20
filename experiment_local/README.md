# Local experiment presets (repo-level)

Use this folder for local/private Hydra experiment presets that should not be committed.

Create files like `experiment_local/my_private_run.yaml` with:

```yaml
# @package _global_
defaults:
  - _self_

experiment_name: my_private_run
trainer:
  max_epochs: 5
```

Run with:

```bash
uv run train_encoder_processor_decoder experiment_local=my_private_run
```

Because `hydra.searchpath` includes `${hydra:runtime.cwd}`, this repo-level folder is discoverable automatically when running from the repository root.
