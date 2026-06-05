# Contributing to the Docs

We welcome all documentation contributions, from fixing small typos to adding comprehensive tutorials. This guide will help you get started.

## Prerequisites

Before contributing, please read our [contributing guide](contributing.md) to understand our workflow.

To test the changes you make to the docs, install the package with documentation dependencies which includes the Jupyter Book package. This will allow you to build the documentation locally.

```bash
uv sync --extra docs
```

## Building the docs locally

From the top level of the repo, build the docs:

```bash
jupyter-book build docs --all
```

Open the generated file `docs/_build/html/index.html` in your browser to preview.

Drop the `--all` flag to build only the changed files:

```bash
jupyter-book build docs
```

## Types of documentation contributions

### 1. Fixing typos and small changes

1. Navigate to the relevant file in the `docs/` directory
2. Make your changes
3. Build the docs locally to verify your changes
4. Open a pull request

### 2. Adding tutorials

1. Create a new Markdown file or Jupyter notebook in `docs/tutorials/`
2. Include:
   - Clear introduction and objectives
   - Step-by-step instructions
   - Code examples
3. Add your tutorial to the table of contents:
   - Open `_toc.yml` in the `docs/` directory
   - Add an entry for your new tutorial
4. Build and verify the docs

### 3. Updating API documentation

The API documentation is generated from source code docstrings. There are two scenarios:

#### Modifying existing API docs

Simply update the docstring in the source code and rebuild:

```bash
jupyter-book build docs --all
```

#### Adding new API docs

1. Create a new `.rst` file in the appropriate `docs/reference/` subdirectory
2. Add the file to `_toc.yml`
3. Ensure your source code has comprehensive docstrings
4. Build the documentation
