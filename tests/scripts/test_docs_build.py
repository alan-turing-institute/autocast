"""Cheap checks for standalone setup, isolation, and checkpoint selection."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import yaml

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("docs_build", REPO / "docs" / "build.py")
assert SPEC is not None
assert SPEC.loader is not None
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def make_book(tmp_path):
    book = tmp_path / "docs"
    (book / "tutorials").mkdir(parents=True)
    (book / "_toc.yml").write_text(
        yaml.safe_dump(
            {
                "chapters": [
                    {"file": "tutorials/z_first"},
                    {
                        "file": "walkthrough/index",
                        "sections": [{"file": "walkthrough/train"}],
                    },
                    {
                        "file": "tutorials/index",
                        "sections": [
                            {
                                "file": "tutorials/topic",
                                "sections": [{"file": "tutorials/a_second"}],
                            },
                        ],
                    },
                ]
            }
        )
    )
    for name in ("z_first", "a_second"):
        nbformat.write(
            nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1 + 1")]),
            book / "tutorials" / f"{name}.ipynb",
        )
    return book


def test_notebooks_can_span_chapters_and_nested_sections(tmp_path):
    book = make_book(tmp_path)
    assert [p.stem for p in build.tutorial_paths(book)] == ["z_first", "a_second"]


def test_navigation_keeps_walkthrough_before_focused_examples_and_results():
    toc = yaml.safe_load((REPO / "docs" / "_toc.yml").read_text())
    chapters = [chapter["file"] for chapter in toc["chapters"]]
    assert chapters[:5] == [
        "installation",
        "tutorials/quickstart",
        "walkthrough/index",
        "tutorials/index",
        "tutorials/evaluation_and_results",
    ]


def test_toc_lists_every_tutorial_in_reading_order():
    paths = build.tutorial_paths(REPO / "docs")
    assert set(paths) == set((REPO / "docs" / "tutorials").glob("*.ipynb"))
    assert [p.stem for p in paths] == [
        "quickstart",
        "autoencoder_and_latents",
        "diffusion_and_flow_matching",
        "deterministic_ensembles",
        "evaluation_and_results",
    ]


def test_execution_uses_toc_order_separate_clients_and_timeout(tmp_path, monkeypatch):
    book = make_book(tmp_path)
    (book / "tutorials" / "outputs").mkdir()
    (book / "tutorials" / "outputs" / "existing.ckpt").write_text("user checkpoint")
    clients = []

    class Client:
        def __init__(self, notebook, **options):
            self.notebook = notebook
            self.options = options
            clients.append(self)

        def execute(self):
            working_dir = Path(self.options["resources"]["metadata"]["path"])
            assert not (working_dir / "outputs").exists()
            assert (working_dir / "z_first.ipynb").is_file()
            assert (working_dir / "a_second.ipynb").is_file()
            (working_dir / "outputs").mkdir()
            (working_dir / "outputs" / "generated.ckpt").touch()
            self.notebook.cells[0].execution_count = len(clients)

    monkeypatch.setattr(build, "NotebookClient", Client)
    build.execute_tutorials(book, timeout=17)
    paths = build.tutorial_paths(book)
    assert [nbformat.read(p, as_version=4).cells[0].execution_count for p in paths] == [
        1,
        2,
    ]
    assert len(clients) == 2
    working_dirs = []
    for client in clients:
        assert client.options["timeout"] == 17
        assert client.options["allow_errors"] is False
        working_dir = Path(client.options["resources"]["metadata"]["path"])
        working_dirs.append(working_dir)
        assert working_dir != book / "tutorials"
        assert not working_dir.exists()
    assert working_dirs[0] != working_dirs[1]
    assert (
        book / "tutorials" / "outputs" / "existing.ckpt"
    ).read_text() == "user checkpoint"


def test_execution_stops_on_first_failure(tmp_path, monkeypatch):
    book = make_book(tmp_path)
    calls = []

    class Client:
        def __init__(self, notebook, **options):
            calls.append(notebook)

        def execute(self):
            message = "cell failed"
            raise RuntimeError(message)

    monkeypatch.setattr(build, "NotebookClient", Client)
    with pytest.raises(RuntimeError, match="cell failed"):
        build.execute_tutorials(book)
    assert len(calls) == 1


def test_build_stages_fresh_outputs_and_preserves_sources(tmp_path, monkeypatch):
    source = make_book(tmp_path)
    (tmp_path / "AC.png").write_bytes(b"logo")
    artifacts = source / "tutorials" / "outputs"
    artifacts.mkdir()
    (artifacts / "existing.ckpt").write_text("user checkpoint")
    source_bytes = build.tutorial_paths(source)[0].read_bytes()
    stages = []

    def execute(book):
        stages.append(book)
        assert not (book / "tutorials" / "outputs").exists()
        notebook = nbformat.read(build.tutorial_paths(book)[0], as_version=4)
        notebook.cells[0].execution_count = 1
        nbformat.write(notebook, build.tutorial_paths(book)[0])

    def render(command, **kwargs):
        book = Path(command[5])
        assert (
            nbformat.read(build.tutorial_paths(book)[0], as_version=4)
            .cells[0]
            .execution_count
            == 1
        )
        assert command[-3:] == ["--path-output", str(source), "--all"]
        assert kwargs["check"] is True

    monkeypatch.setattr(build, "__file__", str(source / "build.py"))
    monkeypatch.setattr(build, "execute_tutorials", execute)
    monkeypatch.setattr(build.subprocess, "run", render)
    build.main()
    build.main()
    assert stages[0] != stages[1]
    assert not any(stage.exists() for stage in stages)
    assert build.tutorial_paths(source)[0].read_bytes() == source_bytes
    assert (artifacts / "existing.ckpt").read_text() == "user checkpoint"


@pytest.mark.parametrize("suffix", ["", "_vit"])
def test_evaluation_uses_recorded_backbone(tmp_path, monkeypatch, suffix):
    selected = {
        "Flow matching": f"generative_processors/flow_matching{suffix}",
        "Diffusion": f"generative_processors/diffusion{suffix}",
    }
    (tmp_path / "processor_runs.json").write_text(json.dumps(selected))
    support = ModuleType("_support")
    support.__dict__.update(
        OUTPUT_ROOT=tmp_path,
        EVALUATION_OPTIONS=[],
        run_autocast=lambda *args: None,
    )
    monkeypatch.setitem(sys.modules, "_support", support)
    notebook = nbformat.read(
        REPO / "docs/tutorials/evaluation_and_results.ipynb",
        as_version=4,
    )
    selection = next(c.source for c in notebook.cells if "selected_runs =" in c.source)
    namespace = {}
    exec(compile(selection, "evaluation_selection", "exec"), namespace)
    for name, relative in selected.items():
        assert namespace["checkpoints"][name] == tmp_path / relative / "processor.ckpt"


def test_result_analysis_filters_runs_without_reevaluating(tmp_path, monkeypatch):
    selected = {
        "Flow matching": "generative_processors/flow_matching_vit",
        "Diffusion": "generative_processors/diffusion_vit",
    }
    (tmp_path / "processor_runs.json").write_text(json.dumps(selected))

    def unexpected_io(*args, **kwargs):
        pytest.fail("Interactive comparison must not rerun evaluation or reload CSVs")

    support = ModuleType("_support")
    support.__dict__.update(
        OUTPUT_ROOT=tmp_path,
        EVALUATION_OPTIONS=[],
        run_autocast=unexpected_io,
        prepare_tutorial=unexpected_io,
    )
    monkeypatch.setitem(sys.modules, "_support", support)
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.switch_backend("Agg")
    for relative in [*selected.values(), "deterministic_ensemble", "older_run"]:
        run = tmp_path / relative
        (run / "eval").mkdir(parents=True)
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": {"processor": {"sampler_steps": 4}},
                    "datamodule": {"batch_size": 4},
                    "optimizer": {"learning_rate": 0.003},
                }
            )
        )
        for filename in ("evaluation_metrics.csv", "rollout_metrics.csv"):
            pd.DataFrame(
                [
                    {
                        "window": "all",
                        "batch_idx": "all",
                        "rmse": 0.2,
                        "crps": 0.1,
                        "ssr": 1.1,
                    }
                ]
            ).to_csv(run / "eval" / filename, index=False)
        pd.DataFrame(
            {0: [0.2, 0.1, 1.1], 1: [0.3, 0.2, 1.2]},
            index=pd.Index(["rmse", "crps", "ssr"]),
        ).to_csv(run / "eval/rollout_metrics_per_timestep_channel_all.csv")
        pd.DataFrame(
            {"coverage_level": [0.5, 0.8], "observed_mean": [0.4, 0.7]}
        ).to_csv(run / "eval/rollout_coverage_window_all.csv", index=False)

    notebook = nbformat.read(
        REPO / "docs/tutorials/evaluation_and_results.ipynb", as_version=4
    )
    namespace = {}
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for source in sources:
        if (
            '"eval.mode=ambient"' not in source
            and "prepare_tutorial(tutorial)" not in source
        ):
            exec(compile(source, "result_analysis", "exec"), namespace)
    assert list(namespace["runs"].index) == [*selected, "Deterministic ensemble"]
    assert list(namespace["runs"]["sampling_steps"]) == [4, 4, 4]
    assert namespace["lead_time"]["Flow matching"].index.tolist() == [0, 1]

    try:
        with monkeypatch.context() as comparison:
            comparison.setattr(pd, "read_csv", unexpected_io)
            for source in sources:
                edited_source = source
                if source.startswith("compare ="):
                    edited_source = source.replace(
                        "compare = list(checkpoints)", 'compare = ["Diffusion"]'
                    )
                elif source.startswith("import matplotlib.pyplot"):
                    edited_source = source.replace(
                        'metrics = ("rmse", "crps")', 'metrics = ("ssr",)'
                    )
                elif not source.startswith(("fig, axis", "runs.loc[compare].to_csv")):
                    continue
                exec(
                    compile(edited_source, "interactive_comparison", "exec"), namespace
                )
        assert list(namespace["scores"].index) == ["Diffusion"]
        assert len(namespace["axes"].flat) == 1
        exported = pd.read_csv(tmp_path / "collated_results.csv")
        assert exported["Run"].tolist() == ["Diffusion"]
        assert exported["run_path"].tolist() == [selected["Diffusion"]]
        assert exported["overall_crps"].tolist() == [0.1]
    finally:
        plt.close("all")


@pytest.fixture
def tutorial_support(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "tutorial_support", REPO / "docs/tutorials/_support.py"
    )
    assert spec is not None
    assert spec.loader is not None
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    monkeypatch.setattr(support, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(support, "__file__", str(tmp_path / "_support.py"))
    return support


@pytest.fixture
def example_files(tutorial_support):
    def create(name):
        selected = {
            "Flow matching": "generative_processors/flow_matching_vit",
            "Diffusion": "generative_processors/diffusion_vit",
        }
        if name == "autoencoder_and_latents":
            files = [
                "autoencoder/autoencoder.ckpt",
                "autoencoder/cached_latents/autoencoder_config.yaml",
                "autoencoder/cached_latents/metadata.json",
                *[
                    f"autoencoder/data/{split}/data.pt"
                    for split in ("train", "valid", "test")
                ],
                *[
                    f"autoencoder/cached_latents/{split}/traj_000000.pt"
                    for split in ("train", "valid", "test")
                ],
            ]
        elif name == "deterministic_ensembles":
            files = [
                "deterministic_ensemble/encoder_processor_decoder.ckpt",
                "deterministic_ensemble/resolved_config.yaml",
                *[
                    f"deterministic_ensemble/data/{split}/data.pt"
                    for split in ("train", "valid", "test")
                ],
            ]
        else:
            files = [
                "processor_runs.json",
                *[
                    f"{run}/{filename}"
                    for run in selected.values()
                    for filename in ("processor.ckpt", "resolved_config.yaml")
                ],
            ]
        paths = [tutorial_support.OUTPUT_ROOT / filename for filename in files]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("existing artifact")
        if name == "diffusion_and_flow_matching":
            paths[0].write_text(json.dumps(selected))
        return paths

    return create


@pytest.mark.parametrize(
    "name",
    [
        "autoencoder_and_latents",
        "diffusion_and_flow_matching",
        "deterministic_ensembles",
    ],
)
def test_preparation_runs_source_once_then_reuses_outputs(
    tutorial_support, example_files, tmp_path, monkeypatch, name
):
    source = tmp_path / f"{name}.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("original_training_code")]
        ),
        source,
    )
    source_bytes = source.read_bytes()
    calls = []

    class Client:
        def __init__(self, notebook, **options):
            calls.append(options)
            self.notebook = notebook

        def execute(self):
            assert self.notebook.cells[0].source == "original_training_code"
            self.notebook.cells[0].execution_count = 1
            example_files(name)

    monkeypatch.setattr(tutorial_support, "NotebookClient", Client)
    tutorial_support.prepare_tutorial(name)
    tutorial_support.prepare_tutorial(name)
    assert len(calls) == 1
    assert calls[0] == {
        "timeout": 180,
        "allow_errors": False,
        "kernel_name": "python3",
        "resources": {"metadata": {"path": str(tmp_path)}},
    }
    assert source.read_bytes() == source_bytes
    if name == "diffusion_and_flow_matching":
        assert not (
            tutorial_support.OUTPUT_ROOT / "generative_processors/diffusion"
        ).exists()


@pytest.mark.parametrize(
    ("name", "missing"),
    [
        ("autoencoder_and_latents", "autoencoder/autoencoder.ckpt"),
        (
            "autoencoder_and_latents",
            "autoencoder/cached_latents/autoencoder_config.yaml",
        ),
        ("autoencoder_and_latents", "autoencoder/cached_latents/metadata.json"),
        ("autoencoder_and_latents", "autoencoder/data/test/data.pt"),
        ("autoencoder_and_latents", "autoencoder/cached_latents/test/traj_000000.pt"),
        ("diffusion_and_flow_matching", "processor_runs.json"),
        (
            "diffusion_and_flow_matching",
            "generative_processors/diffusion_vit/processor.ckpt",
        ),
        (
            "diffusion_and_flow_matching",
            "generative_processors/flow_matching_vit/resolved_config.yaml",
        ),
        ("deterministic_ensembles", "deterministic_ensemble/resolved_config.yaml"),
        ("deterministic_ensembles", "deterministic_ensemble/data/valid/data.pt"),
    ],
)
def test_preparation_does_not_replace_incomplete_runs(
    tutorial_support, example_files, monkeypatch, name, missing
):
    paths = example_files(name)
    (tutorial_support.OUTPUT_ROOT / missing).unlink()
    before = {path: path.read_bytes() for path in paths if path.is_file()}
    monkeypatch.setattr(
        tutorial_support,
        "NotebookClient",
        lambda *args, **kwargs: pytest.fail("Do not retrain"),
    )
    with pytest.raises(RuntimeError, match="rerun that notebook explicitly"):
        tutorial_support.prepare_tutorial(name)
    assert {path: path.read_bytes() for path in before} == before


def test_preparation_propagates_training_failure(
    tutorial_support, tmp_path, monkeypatch
):
    source = tmp_path / "autoencoder_and_latents.ipynb"
    nbformat.write(nbformat.v4.new_notebook(), source)

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self):
            message = "training failed"
            raise RuntimeError(message)

    monkeypatch.setattr(tutorial_support, "NotebookClient", Client)
    with pytest.raises(RuntimeError, match="training failed"):
        tutorial_support.prepare_tutorial("autoencoder_and_latents")


@pytest.mark.parametrize(
    ("notebook", "preparation_call", "expected"),
    [
        (
            "diffusion_and_flow_matching",
            'prepare_tutorial("autoencoder_and_latents")',
            ["autoencoder_and_latents"],
        ),
        (
            "evaluation_and_results",
            "prepare_tutorial(tutorial)",
            [
                "autoencoder_and_latents",
                "diffusion_and_flow_matching",
                "deterministic_ensembles",
            ],
        ),
    ],
)
def test_notebooks_prepare_examples_in_separate_optional_cells(
    notebook, preparation_call, expected, monkeypatch
):
    prepared = []
    support = ModuleType("_support")
    support.__dict__["prepare_tutorial"] = prepared.append
    monkeypatch.setitem(sys.modules, "_support", support)
    page = nbformat.read(REPO / f"docs/tutorials/{notebook}.ipynb", as_version=4)
    source = next(
        cell.source
        for cell in page.cells
        if cell.cell_type == "code" and preparation_call in cell.source
    )
    exec(
        compile(source, "prepare_examples", "exec"),
        {"prepare_tutorial": prepared.append},
    )
    assert prepared == expected
