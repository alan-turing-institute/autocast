from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from autocast.scripts import plot_dataset_comparisons as pdc


def test_default_plot_metrics_include_overall_crps_and_ssr():
    assert pdc.DEFAULT_PLOT_METRICS == ("vrmse", "coverage", "crps", "ssr")

    err_metric, cov_metric = pdc._derive_lead_time_metrics(
        list(pdc.DEFAULT_PLOT_METRICS)
    )

    assert err_metric == ["vrmse", "crps", "rmse"]
    assert cov_metric == ["coverage", "coverage_0.9", "coverage_0.5", "ssr"]


def test_overall_ssr_bars_are_linear_and_ignore_error_ylim():
    error_ylim = (1e-5, 1.0)

    assert pdc._overall_or_window_bar_yscale("ssr") == "linear"
    assert pdc._overall_or_window_bar_ylim("ssr", error_ylim) is None
    assert pdc._overall_or_window_bar_ref_value("ssr") == 1.0
    assert pdc._overall_or_window_bar_ylim("crps", error_ylim) == error_ylim
    assert pdc._overall_or_window_bar_ref_value("crps") is None


def test_panel_figure_renders_all_requested_overall_metrics(
    monkeypatch,
    tmp_path: Path,
):
    calls: list[dict[str, Any]] = []

    def fake_grouped_bar(
        _df_in: pd.DataFrame,
        metric: str,
        title: str,
        _ylabel: str,
        _out_dir: Path,
        _styles: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        calls.append(
            {
                "metric": metric,
                "y_scale": kwargs.get("y_scale"),
                "ylim": kwargs.get("ylim"),
                "ref_value": kwargs.get("ref_value"),
            }
        )
        kwargs["ax"].set_title(title)

    def noop(*args: Any, **kwargs: Any) -> None:
        return None

    def fake_save_fig(fig: Figure, _out_dir: Path, _name: str) -> None:
        plt.close(fig)

    monkeypatch.setattr(pdc, "grouped_bar", fake_grouped_bar)
    monkeypatch.setattr(pdc, "plot_training_curves", noop)
    monkeypatch.setattr(pdc, "plot_coverage_calibration_panel", noop)
    monkeypatch.setattr(pdc, "plot_lead_time_panel", noop)
    monkeypatch.setattr(pdc, "save_fig", fake_save_fig)

    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "run_path": ["run"],
            "eval_subdir": ["eval"],
        }
    )
    styles = {"model": {"color": "black", "label": "model", "linestyle": "-"}}

    pdc.plot_panel_figure(
        df,
        tmp_path,
        tmp_path,
        styles,
        overall_metrics=("vrmse", "coverage", "crps", "ssr"),
        error_metrics=["vrmse"],
        coverage_metrics=["coverage_0.9", "ssr"],
        training_metrics=[],
        error_ylim=(1e-5, 1.0),
    )

    assert [c["metric"] for c in calls] == [
        "overall_vrmse",
        "overall_coverage",
        "overall_crps",
        "overall_ssr",
    ]
    assert calls[-1]["y_scale"] == "linear"
    assert calls[-1]["ylim"] is None
    assert calls[-1]["ref_value"] == 1.0


def test_grouped_bar_draws_requested_reference_line(tmp_path: Path):
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "overall_ssr": [0.8],
        }
    )
    styles = {"model": {"color": "black", "label": "model", "linestyle": "-"}}
    fig, ax = plt.subplots()

    pdc.grouped_bar(
        df,
        "overall_ssr",
        "Overall SSR",
        "SSR",
        tmp_path,
        styles,
        y_scale="linear",
        ax=ax,
        save=False,
        ref_value=1.0,
    )

    ref_lines = [
        line
        for line in ax.lines
        if np.asarray(line.get_ydata(), dtype=float).tolist() == [1.0, 1.0]
        and line.get_linestyle() == ":"
    ]
    assert len(ref_lines) == 1
    plt.close(fig)


def test_grouped_bar_can_scale_axis_labels_with_ticks(tmp_path: Path):
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "overall_vrmse": [0.1],
        }
    )
    styles = {"model": {"color": "black", "label": "model", "linestyle": "-"}}
    fig, ax = plt.subplots()

    pdc.grouped_bar(
        df,
        "overall_vrmse",
        "Overall VRMSE",
        "VRMSE",
        tmp_path,
        styles,
        ax=ax,
        save=False,
        tick_label_scale=1.5,
        axis_label_scale=1.5,
    )

    assert ax.yaxis.label.get_size() == 15.0
    assert ax.xaxis.get_ticklabels()[0].get_size() == 15.0
    plt.close(fig)


def test_single_step_results_table_uses_grouped_bar_means():
    df = pd.DataFrame(
        {
            "dataset_label": ["AD", "AD", "AD"],
            "plot_group": ["crps", "crps", "fm"],
            "overall_vrmse": [1.0, 3.0, 2.0],
            "overall_crps": [0.1, 0.3, 0.2],
            "overall_ssr": [0.8, 1.0, 1.1],
            "model_latency_ms_per_sample": [10.0, 14.0, 8.0],
            "train_total_s": [3600.0, 7200.0, 1800.0],
            "train_mean_epoch_s": [100.0, 200.0, 50.0],
        }
    )
    styles = {
        "crps": {"label": "CRPS", "color": "tab:blue"},
        "fm": {"label": "FM", "color": "tab:orange"},
    }

    table = pdc.build_single_step_results_table(
        df,
        styles,
        dataset_order=["AD"],
        hue_order=["CRPS", "FM"],
    )

    assert table["Model"].tolist() == ["CRPS", "FM"]
    assert table.loc[0, "VRMSE"] == 2.0
    assert table.loc[0, "CRPS"] == 0.2
    assert table.loc[0, "SSR"] == 0.9
    assert table.loc[0, "Inference latency (ms/sample)"] == 12.0
    assert "Training time (h)" not in table.columns
    assert table.loc[0, "Training time (s/epoch)"] == 150.0


def test_single_step_results_latex_uses_two_sig_figs(tmp_path: Path):
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["crps"],
            "overall_vrmse": [0.012345],
            "overall_crps": [0.098765],
        }
    )
    styles = {"crps": {"label": "CRPS", "color": "tab:blue"}}

    pdc.write_single_step_results_table(df, tmp_path, styles)

    tex = (tmp_path / "single_step_overall_results.tex").read_text()
    assert r"\begin{tabularx}{\linewidth}" in tex
    assert r"\shortstack{Inference\\latency\\(ms/sample)}" in tex
    assert "1.2e-02" in tex
    assert "9.9e-02" in tex
    assert "0.012345" not in tex
    assert "0.098765" not in tex


def test_single_step_results_latex_bolds_best_values_by_dataset(tmp_path: Path):
    df = pd.DataFrame(
        {
            "dataset_label": ["AD", "AD", "CNS", "CNS"],
            "plot_group": ["crps", "fm", "crps", "fm"],
            "overall_vrmse": [0.2, 0.1, 0.3, 0.4],
            "overall_crps": [0.05, 0.06, 0.08, 0.07],
            "overall_ssr": [0.7, 1.2, 1.1, 0.6],
            "model_latency_ms_per_sample": [12.0, 8.0, 9.0, 11.0],
            "train_total_s": [3600.0, 7200.0, 5400.0, 1800.0],
            "train_mean_epoch_s": [100.0, 200.0, 150.0, 50.0],
        }
    )
    styles = {
        "crps": {"label": "CRPS", "color": "tab:blue"},
        "fm": {"label": "FM", "color": "tab:orange"},
    }

    pdc.write_single_step_results_table(
        df,
        tmp_path,
        styles,
        dataset_order=["AD", "CNS"],
        hue_order=["CRPS", "FM"],
    )

    tex = (tmp_path / "single_step_overall_results.tex").read_text()
    assert r"\textbf{1.0e-01}" in tex
    assert r"\textbf{5.0e-02}" in tex
    assert r"\textbf{1.2e+00}" in tex
    assert r"\textbf{7.0e-02}" in tex
    assert r"\textbf{1.1e+00}" in tex


def test_coverage_calibration_panel_uses_publication_axis_labels(tmp_path: Path):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    coverage = pd.DataFrame(
        {
            "coverage_level": [0.1, 0.5, 0.9],
            "observed_mean": [0.05, 0.4, 0.8],
        }
    )
    coverage.to_csv(eval_dir / "test_coverage_window_all.csv", index=False)
    coverage.to_csv(eval_dir / "rollout_coverage_window_0-4.csv", index=False)
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "run_path": ["run1"],
            "eval_subdir": ["eval"],
        }
    )
    styles = {"model": {"color": "black", "label": "model", "linestyle": "-"}}

    fig = pdc.plot_coverage_calibration_panel(
        df,
        tmp_path,
        tmp_path,
        styles,
        window_rows=["all", "0-4"],
        save=False,
    )

    assert fig is not None
    axes = fig.axes
    assert axes[0].get_ylabel() == "Empirical coverage"
    assert axes[1].get_ylabel() == "Empirical coverage (0:4)"
    assert axes[1].get_xlabel() == r"Expected coverage (1 - $\alpha$)"
    plt.close(fig)


def test_coverage_calibration_panel_can_use_shared_xlabel_and_taller_height(
    tmp_path: Path,
):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    coverage = pd.DataFrame(
        {
            "coverage_level": [0.1, 0.5, 0.9],
            "observed_mean": [0.05, 0.4, 0.8],
        }
    )
    coverage.to_csv(eval_dir / "test_coverage_window_all.csv", index=False)
    coverage.to_csv(eval_dir / "rollout_coverage_window_0-4.csv", index=False)
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "run_path": ["run1"],
            "eval_subdir": ["eval"],
        }
    )
    styles = {"model": {"color": "black", "label": "model", "linestyle": "-"}}

    fig = pdc.plot_coverage_calibration_panel(
        df,
        tmp_path,
        tmp_path,
        styles,
        window_rows=["all", "0-4"],
        shared_axis_labels=True,
        height_scale=1.5,
        save=False,
    )

    assert fig is not None
    assert fig.get_size_inches()[1] == 2.3 * 2 * 1.5
    assert fig.axes[1].get_xlabel() == ""
    assert r"Expected coverage (1 - $\alpha$)" in [
        text.get_text() for text in fig.texts
    ]
    plt.close(fig)


def test_lead_time_coverage_delta_is_proportional(tmp_path: Path):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        [[0.25, 0.75]],
        index=pd.Index(["coverage_0.5"], name="metric"),
        columns=["0", "1"],
    ).to_csv(eval_dir / "rollout_metrics_per_timestep_channel_0.csv")
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "run_path": ["run1"],
            "eval_subdir": ["eval"],
        }
    )
    styles = {"model": {"color": "black", "label": "model", "linestyle": "-"}}

    fig = pdc.plot_lead_time_panel(
        df,
        ["coverage_0.5"],
        tmp_path,
        tmp_path,
        "coverage_delta.png",
        styles,
        coverage_delta=True,
        save=False,
    )

    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_ylabel() == ""
    assert r"$\Delta$ empirical coverage (proportional)" in [
        text.get_text() for text in fig.texts
    ]
    assert np.asarray(ax.lines[0].get_ydata(), dtype=float).tolist() == [-0.5, 0.5]
    plt.close(fig)


def test_short_axis_labels_use_compact_shared_coverage_delta(tmp_path: Path):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        [[0.25, 0.75]],
        index=pd.Index(["coverage_0.5"], name="metric"),
        columns=["0", "1"],
    ).to_csv(eval_dir / "rollout_metrics_per_timestep_channel_0.csv")
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "run_path": ["run1"],
            "eval_subdir": ["eval"],
        }
    )
    styles = {"model": {"color": "black", "label": "model", "linestyle": "-"}}

    fig = pdc.plot_lead_time_panel(
        df,
        ["coverage_0.5"],
        tmp_path,
        tmp_path,
        "coverage_delta.png",
        styles,
        coverage_delta=True,
        short_axis_labels=True,
        save=False,
    )

    assert fig is not None
    assert r"Rel. $\Delta$ empirical coverage" in [
        text.get_text() for text in fig.texts
    ]
    plt.close(fig)


def test_lead_time_error_labels_are_uppercase(tmp_path: Path):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.Index(["vrmse"], name="metric"),
        columns=["0", "1"],
    ).to_csv(eval_dir / "rollout_metrics_per_timestep_channel_0.csv")
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "run_path": ["run1"],
            "eval_subdir": ["eval"],
        }
    )
    styles = {"model": {"color": "black", "label": "model", "linestyle": "-"}}

    fig = pdc.plot_lead_time_panel(
        df,
        ["vrmse"],
        tmp_path,
        tmp_path,
        "lead_time.png",
        styles,
        save=False,
    )

    assert fig is not None
    assert fig.axes[0].get_ylabel() == "VRMSE"
    plt.close(fig)
