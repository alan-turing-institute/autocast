# Configurations

The AutoCast CLI is built on top of [Hydra](https://hydra.cc/).
This means that configurations are specified in YAML files and can be composed together to quickly switch between different datasets and model architectures.

## Command-line overrides

Hydra allows you to add or override any configuration value from the command line.
See [the Hydra documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) for more details.
As an example, to _override_ the number of training epochs for the `ae` command, you can run:

```bash
uv run autocast ae trainer.max_epochs=5
```

Note that this only works if the `trainer.max_epochs` key is defined in the default configuration for that command.
If the key is not defined, you have to prefix it with `+` to tell Hydra to add it:

```bash
uv run autocast ae +trainer.max_epochs=5
```

If you want to specify the option regardless of whether it is defined in the default config or not, you can use `++`.
It can be difficult to know whether an option has already been defined, so using `++` is a safe way to ensure that the option is set:

```bash
uv run autocast ae ++trainer.max_epochs=5
```

## Making your own configurations

If you find yourself making the same overrides repeatedly, it is probably worth it to make a new YAML configuration that specifies these overrides.
If they are generally useful for the package, these can be stored in `src/autocast/configs/experiments/<myexpt>.yaml` and specified on the command-line with `+experiment=<myexpt>`.

These largely follow the same pattern as the command-line overrides.
If you are specifying a *configuration group*, for example you want to override the entire `model` configuration group to point to a different model, you need to specify this in what is called the "defaults list" of your configuration file.

For example, let's say you want to create a new experiment configuration that is stored in `src/autocast/configs/experiments/my_experiment.yaml`.
You want this to largely be the same as a pre-existing experiment configuration called `parent_experiment`, but you want to override the model to use a different architecture called `my_model`.
You could then create a new configuration file with the following contents:

```yaml
defaults:
  - experiment: parent_experiment
  - model: my_model
  - _self_
```

The defaults list is processed from top to bottom, so `parent_experiment`'s configuration is loaded first, and then `my_model` will be applied on top of that.
Finally, `_self_` is a special directive that tells Hydra to load the rest of the configuration from this file after the defaults have been applied.

This file doesn't yet have any additional configuration, but you could add more options below the defaults list if you wanted to.
This is how you could override *configuration values* (rather than groups).
For example if you wanted to override the number of training epochs, you could do:

```yaml
defaults:
  - experiment: parent_experiment
  - model: my_model
  - _self_

trainer:
    max_epochs: 500
```

### Local experiments

Some of the configurations we used for our experiments are stored in the `local_hydra/local_experiment` folder.
These are not in the main `src/autocast/configs` folder because they are not meant to be distributed as part of the package.
These configuration files can, however, still be used by setting `local_experiment=<name>` on the command line.

(This works because AutoCast makes sure to add `local_hydra` to Hydra's search path, allowing you to load configurations from there even though they are outside the package.)
