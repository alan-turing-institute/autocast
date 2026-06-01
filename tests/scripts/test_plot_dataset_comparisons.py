from __future__ import annotations

from pathlib import Path

import pandas as pd

from autocast.scripts import plot_dataset_comparisons as pdc


def _write_run(
    results_dir: Path,
    run_id: str,
    *,
    eval_subdir: str = "eval",
    dataset: str = "ad64",
) -> Path:
    run_dir = results_dir / run_id
    eval_dir = run_dir / eval_subdir
    eval_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        f"datamodule:\n  dataset: {dataset}\n"
    )
    pd.DataFrame(
        [
            {
                "window": "all",
                "batch_idx": "all",
                "vrmse": 0.1,
                "coverage": 0.2,
                "crps": 0.3,
                "ssr": 1.1,
            }
        ]
    ).to_csv(eval_dir / "evaluation_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"window": "0-4", "batch_idx": "all", "vrmse": 0.4},
            {"window": "5-9", "batch_idx": "all", "vrmse": 0.5},
        ]
    ).to_csv(eval_dir / "rollout_metrics.csv", index=False)
    pd.DataFrame(
        [[0.1, 0.2]],
        index=pd.Index(["vrmse"], name="metric"),
        columns=pd.Index(["0", "1"]),
    ).to_csv(eval_dir / "rollout_metrics_per_timestep_channel_all.csv")
    return run_dir


def test_default_plot_metrics_are_small_package_surface():
    assert pdc.DEFAULT_PLOT_METRICS == ("vrmse", "coverage", "crps", "ssr")
    help_text = pdc.build_parser().format_help()

    assert "--lead-time-metrics" in help_text
    assert "paper" not in help_text
    assert "notebook" not in help_text


def test_run_target_preserves_nested_relative_paths(tmp_path: Path):
    results_dir = tmp_path / "outputs"
    run_id = "2026-05-01/crps_ad64_vit_azula_large_abcd123_ef45678"
    run_dir = _write_run(results_dir, run_id)

    [target] = pdc._discover_run_targets(results_dir)
    row = pdc.load_single_run_metrics(
        target.path,
        eval_subdir=target.eval_subdir,
        run_ref=target.ref,
        run_path=target.relative_path,
    )

    assert run_dir.exists()
    assert target.relative_path == run_id
    assert row["run_path"] == run_id
    assert row["run_name"] == "crps_ad64_vit_azula_large_abcd123_ef45678"


def test_discover_run_targets_defaults_to_one_eval_per_run(tmp_path: Path):
    results_dir = tmp_path / "outputs"
    run_dir = results_dir / "2026-05-01" / "crps_ad64_vit_abcd123_ef45678"
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text("datamodule: {}\n")
    for eval_subdir in [
        "eval",
        "eval_best_multiwinkler_from0p25",
        "eval_encode_once_ode001",
    ]:
        eval_dir = run_dir / eval_subdir
        eval_dir.mkdir()
        pd.DataFrame([{"window": "all", "batch_idx": "all", "vrmse": 0.1}]).to_csv(
            eval_dir / "evaluation_metrics.csv", index=False
        )
    artifact_eval_dir = results_dir / "plots" / "copied" / "eval_extra"
    artifact_eval_dir.mkdir(parents=True)
    pd.DataFrame([{"window": "all", "batch_idx": "all", "vrmse": 0.1}]).to_csv(
        artifact_eval_dir / "evaluation_metrics.csv", index=False
    )

    targets = pdc._discover_run_targets(results_dir)

    assert [target.eval_subdir for target in targets] == ["eval"]
    assert targets[0].relative_path == str(run_dir.relative_to(results_dir))
    assert targets[0].ref == run_dir.name
    assert pdc._available_eval_subdirs(run_dir) == [
        "eval",
        "eval_best_multiwinkler_from0p25",
        "eval_encode_once_ode001",
    ]


def test_discover_run_targets_uses_eval_postfix_when_default_missing(tmp_path: Path):
    results_dir = tmp_path / "outputs"
    run_dir = _write_run(
        results_dir,
        "2026-05-01/crps_ad64_vit_abcd123_ef45678",
        eval_subdir="eval_encode_once_ode001",
    )

    [target] = pdc._discover_run_targets(results_dir)

    assert target.eval_subdir == "eval_encode_once_ode001"
    assert target.ref == f"{run_dir.name}::eval=eval_encode_once_ode001"


def test_format_available_eval_subdirs_truncates_long_lists():
    assert (
        pdc._format_available_eval_subdirs(
            ["eval", "eval_0p50", "eval_0p75", "eval_encode_once"]
        )
        == "eval, eval_0p50, eval_0p75, ... (+1)"
    )


def test_load_single_run_metrics_reads_overall_and_rollout_metrics(tmp_path: Path):
    run_dir = _write_run(tmp_path, "run_abc", dataset="conditioned_navier_stokes")

    row = pdc.load_single_run_metrics(run_dir)

    assert row["dataset_label"] == "CNS64"
    assert row["overall_vrmse"] == 0.1
    assert row["overall_crps"] == 0.3
    assert row["vrmse_0-4"] == 0.4
    assert row["vrmse_5-9"] == 0.5


def test_filter_runs_supports_dataset_model_and_column_filters():
    df = pd.DataFrame(
        {
            "dataset_label": ["AD64", "CNS64"],
            "run_name": ["small_model", "large_model"],
            "eval_subdir": ["eval", "eval_extra"],
        }
    )

    filtered = pdc.filter_runs(
        df,
        datasets=["cns"],
        models=["large"],
        filters=["eval_subdir=eval_extra"],
    )

    assert filtered["run_name"].tolist() == ["large_model"]


def test_plot_overall_metric_writes_requested_format(tmp_path: Path):
    df = pd.DataFrame(
        {
            "dataset_label": ["AD64", "AD64"],
            "plot_group": ["model_a", "model_b"],
            "overall_vrmse": [0.2, 0.1],
        }
    )

    fig = pdc.plot_overall_metric(df, "vrmse", tmp_path, figure_formats=("png",))

    assert fig is not None
    assert (tmp_path / "overall_vrmse.png").exists()


def test_plot_lead_time_metric_writes_requested_format(tmp_path: Path):
    results_dir = tmp_path / "outputs"
    _write_run(results_dir, "run_abc")
    df = pd.DataFrame(
        {
            "run_path": ["run_abc"],
            "eval_subdir": ["eval"],
            "plot_group": ["model_a"],
        }
    )

    fig = pdc.plot_lead_time_metric(
        df,
        results_dir,
        "vrmse",
        tmp_path,
        figure_formats=("png",),
    )

    assert fig is not None
    assert (tmp_path / "lead_time_vrmse.png").exists()


def test_main_writes_table_and_plots(tmp_path: Path):
    results_dir = tmp_path / "outputs"
    _write_run(results_dir, "run_abc")
    output_dir = tmp_path / "plots"

    pdc.main(
        [
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
            "--metrics",
            "vrmse",
            "--lead-time-metrics",
            "vrmse",
        ]
    )

    assert (output_dir / "selected_runs.csv").exists()
    assert (output_dir / "overall_vrmse.png").exists()
    assert (output_dir / "lead_time_vrmse.png").exists()
