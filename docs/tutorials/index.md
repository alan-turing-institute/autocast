# Focused notebooks

Start with the [Python quickstart](quickstart.ipynb) for a first forecast, or
follow the self-contained [CLI walkthrough](../walkthrough/index.md) for a
complete experiment with saved configurations and checkpoints.

These notebooks explore individual choices with small AutoSim datasets, CPU
training and plots. No dataset downloads or tracking accounts are needed.
Each notebook can run on its own. The short runs demonstrate behaviour, not
forecasting benchmarks.

| Tutorial | What it shows |
| --- | --- |
| [Autoencoder and latent data](autoencoder_and_latents.ipynb) | Train with Lightning, inspect latents, reload and cache |
| [Diffusion and flow matching](diffusion_and_flow_matching.ipynb) | Train Python models, switch UNet/ViT, resample and plot |
| [Deterministic ensembles](deterministic_ensembles.ipynb) | Train with CRPS, control noise and plot ensemble members |

After training, [evaluation and results](evaluation_and_results.ipynb) covers
physical-space metrics, uncertainty and comparison of saved runs. Evaluate once,
then select runs and metrics interactively from the saved CSVs. Its runnable
example can reuse the outputs from both processor notebooks above.

## Run locally

From the repository root, install the notebook environment:

```bash
uv sync --extra dev --extra docs
```

Open the notebooks in `docs/tutorials/` using that environment's Python kernel.
Run each notebook from its own directory. Data and checkpoints are written to
`docs/tutorials/outputs/`. The generative and evaluation notebooks have a short
setup cell: missing examples are prepared by running the original training
notebooks in separate kernels, without duplicating their code. Complete runs
are reused; incomplete existing runs are left untouched and need an explicit
rerun of their source notebook. After changing data settings, rerun the affected
training notebooks too.

The shared {download}`support file <_support.py>` supplies output paths and CLI
settings for evaluation, plus this optional preparation helper. Keep the other
notebooks alongside it when running locally. The feature each notebook teaches
is still implemented directly in its Python cells. Training logs are collapsible
in the rendered book.

## Build and check all notebooks

```bash
uv run python docs/build.py
```

The build runs the quickstart, the three notebooks above, then evaluation and
results, each in its own temporary directory and kernel. Every page starts
without data or checkpoints, exercising standalone setup in CI. It renders the
executed outputs to `docs/_build/html/`, without changing source notebooks or
reusing old checkpoints. Each cell has a
180-second timeout; CI caps execution plus rendering at ten minutes. Dependency
downloads are cached, but notebook execution is repeated after library changes.

CLI walkthrough pages are explanatory Markdown and are not executed by this build.
