# Training model ensembles

`autocast` also provides functionality for training model ensembles.
This can be done with a very simple override.
Consider our [original processor training command](./processor.md):

```
uv run autocast processor \
    --workdir ../proc_output \
    datamodule=cached_latents \
    ++datamodule.data_path=/path/to/parent_folder/ae_output/cached_latents \
    ++trainer.max_epochs=10
```

We just need to augment this with `++model.n_members` which specifies the ensemble size, as well as an ensemble-aware loss function.
The [CRPS loss function](https://en.wikipedia.org/wiki/Scoring_rule#Continuous_ranked_probability_score) is a common choice for probabilistic forecasts as it explicitly rewards diversity in the ensemble members, instead of just forcing them towards the mean prediction (which the original mean-squared error loss does).

```
uv run autocast processor \
    --workdir ../ensemble_proc_output \
    datamodule=cached_latents \
    ++datamodule.data_path=/path/to/parent_folder/ae_output/cached_latents \
    ++trainer.max_epochs=10
    ++model.n_members=10 \
    ++model.loss_func._target_=autocast.losses.ensemble.CRPSLoss
```

Similar overrides can be applied to the `epd` command.
