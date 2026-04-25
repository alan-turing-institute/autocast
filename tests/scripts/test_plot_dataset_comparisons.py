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
