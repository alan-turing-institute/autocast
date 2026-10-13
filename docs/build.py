"""Execute the tutorials in a fresh copy, then render the Jupyter Book.

Run with ``uv run python docs/build.py``. Source notebooks stay output-free;
generated HTML goes to ``docs/_build/html``.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient


def tutorial_paths(book: Path) -> list[Path]:
    """Find notebooks across TOC chapters, preserving reading order."""
    toc = yaml.safe_load((book / "_toc.yml").read_text())

    def collect(entries: list[dict]) -> list[Path]:
        paths = []
        for entry in entries:
            path = book / f"{entry['file']}.ipynb"
            if path.is_file():
                paths.append(path)
            paths.extend(collect(entry.get("sections", [])))
        return paths

    return collect(toc["chapters"])


def execute_tutorials(book: Path, timeout: int = 180) -> None:
    """Run every page independently, without earlier notebooks' artifacts."""
    started = time.perf_counter()
    for path in tutorial_paths(book):
        notebook = nbformat.read(path, as_version=4)
        notebook_started = time.perf_counter()
        print(f"Executing {path.name}", flush=True)
        with tempfile.TemporaryDirectory(
            prefix=f"{path.stem}-", dir=book.parent
        ) as work:
            tutorials = Path(work) / "tutorials"
            shutil.copytree(
                path.parent,
                tutorials,
                ignore=shutil.ignore_patterns("outputs", "__pycache__"),
            )
            NotebookClient(
                notebook,
                timeout=timeout,
                allow_errors=False,
                kernel_name="python3",
                resources={"metadata": {"path": str(tutorials)}},
            ).execute()
        nbformat.write(notebook, path)
        print(f"  Passed in {time.perf_counter() - notebook_started:.1f}s", flush=True)
    print(f"All tutorials passed in {time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    """Build with fresh data/checkpoints; never rewrite the source notebooks."""
    docs = Path(__file__).resolve().parent
    repo = docs.parent
    build_dir = docs / "_build"
    build_dir.mkdir(exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    # Keep staging inside the checkout so the CLI can find local Hydra configs.
    with tempfile.TemporaryDirectory(prefix="execution-", dir=build_dir) as staging:
        stage = Path(staging)
        book = stage / "docs"
        shutil.copytree(
            docs,
            book,
            ignore=shutil.ignore_patterns("_build", "outputs", "__pycache__"),
        )
        shutil.copy2(repo / "AC.png", stage / "AC.png")
        execute_tutorials(book)
        subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "jupyter-book",
                "build",
                str(book),
                "--path-output",
                str(docs),
                "--all",
            ],
            cwd=repo,
            check=True,
        )


if __name__ == "__main__":
    main()
