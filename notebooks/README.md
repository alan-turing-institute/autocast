# AutoCast notebooks

The maintained notebooks now live in [docs/tutorials](../docs/tutorials/index.md),
with unnumbered filenames and a reading order in the Jupyter Book TOC. Each can
run independently; optional setup cells prepare any missing example runs.

Run all five and build their plotted documentation with:

```bash
uv sync --extra dev --extra docs
uv run python docs/build.py
```

## Replacing the exploratory notebooks

| Previous notebooks | Maintained tutorial |
| --- | --- |
| `00_quickstart`, `04_e2e` | [Python API](../docs/tutorials/quickstart.ipynb), [autoencoder](../docs/tutorials/autoencoder_and_latents.ipynb), and [end-to-end CLI](../docs/walkthrough/epd.md) |
| `01_encoder_decoder` | [Autoencoder and latent data](../docs/tutorials/autoencoder_and_latents.ipynb) |
| `05_ViT`, `06_00_processor_train`, `wip/02_diffusion` | [Diffusion, flow matching, and backbone choices](../docs/tutorials/diffusion_and_flow_matching.ipynb) |
| `07_ViT_ensemble`, `07b_UNet_ensemble` | [Deterministic ensembles](../docs/tutorials/deterministic_ensembles.ipynb) |
| `06_01_processor_eval_latent`, `06_02_processor_eval_ambient`, `08_collating_multiple_results` | [Decoded previews](../docs/tutorials/diffusion_and_flow_matching.ipynb) and [evaluation](../docs/tutorials/evaluation_and_results.ipynb) |

The old notebooks remain recoverable from Git history. Their exploratory
variants (such as direct learned-autoencoder/EPD assembly and additive or
concatenated input noise) are not all maintained as separate runnable examples.
Research-scale workflows belong in Markdown walkthroughs, not in notebook CI.
