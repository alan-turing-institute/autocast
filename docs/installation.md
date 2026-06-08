# Installation

`AutoCast` is a Python package that can be installed using [uv](https://github.com/astral-sh/uv).

## Prerequisites

**Python Version:** `AutoCast` requires Python `>=3.11` and `<3.13`.

**uv:** We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management and running scripts.

**ffmpeg** (optional): Required for video generation during evaluation. Install via your system package manager (e.g. `brew install ffmpeg` on macOS).

## Install from source

Clone the repository:

```bash
git clone https://github.com/alan-turing-institute/autocast.git
cd autocast
```

Install with uv:

```bash
uv sync
```

## Install for development

For development, install with dev dependencies:

```bash
uv sync --extra dev
```

If contributing to the codebase, set up pre-commit hooks:

```bash
prek install
```

This will setup the pre-commit checks so any pushed commits will pass the CI.

## Building the documentation

To build the documentation locally, install development dependencies and run:

```bash
uv sync --extra docs
jupyter-book build docs --all
```

Then open `docs/_build/html/index.html` in your browser to preview.
