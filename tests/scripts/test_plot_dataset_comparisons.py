from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from autocast.scripts import plot_dataset_comparisons as pdc


def test_cli_linestyles_distinguish_evaluations_of_same_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    run = "diff_ad64_flow_matching_vit_example"
    for subdir, values in [("eval", [0.1, 0.2]), ("eval_ema", [0.2, 0.3])]:
        eval_dir = tmp_path / run / subdir
        eval_dir.mkdir(parents=True)
        pd.DataFrame({"vrmse": values}).T.to_csv(
            eval_dir / "rollout_metrics_per_timestep_channel_all.csv"
        )

    captured = []

    def capture_ablation(df, results_root, out_dir, styles, *_args, **_kwargs):
        assert df["eval_subdir"].tolist() == ["eval", "eval_ema"]
        fig = pdc.plot_lead_time_panel(
            df,
            ["vrmse"],
            results_root,
            out_dir,
            "unused.png",
            styles,
            save=False,
        )
        assert isinstance(fig, Figure)
        lines = fig.axes[0].lines
        assert len(lines) == 2
        assert [line.get_linestyle() for line in lines] == ["-", "--"]
        assert lines[0].get_color() == lines[1].get_color()
        np.testing.assert_allclose(np.asarray(lines[0].get_ydata()), [0.1, 0.2])
        np.testing.assert_allclose(np.asarray(lines[1].get_ydata()), [0.2, 0.3])
        legend = fig.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == [
            "FM (no EMA)",
            "FM (EMA)",
        ]
        assert [line.get_linestyle() for line in legend.get_lines()] == ["-", "--"]
        captured.append(True)
        plt.close(fig)

    monkeypatch.setattr(pdc, "plot_four_ds_ablation_figure", capture_ablation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autocast-plots",
            "--results-dir",
            str(tmp_path),
            "--run",
            run,
            "FM (no EMA)",
            "1",
            "eval=eval",
            "dataset=AD",
            "linestyle=solid",
            "--run",
            run,
            "FM (EMA)",
            "1",
            "eval=eval_ema",
            "dataset=AD",
            "linestyle=dashed",
            "--uniform-run-hue-color",
            "--paper-only",
            "--four-ds-ablation",
        ],
    )
    pdc.main()
    assert captured == [True]


@pytest.mark.parametrize(
    "tokens",
    [
        ["linestyle=invalid"],
        ["linestyle="],
        ["linestyle=solid", "linestyle=dashed"],
    ],
)
def test_cli_rejects_invalid_linestyle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    tokens: list[str],
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autocast-plots",
            "--results-dir",
            str(tmp_path),
            "--list",
            "--run",
            "example",
            *tokens,
        ],
    )
    with pytest.raises(SystemExit, match="1"):
        pdc.main()
    assert "linestyle" in capsys.readouterr().out


def test_base_first_run_hue_keeps_main_color_and_darkens_variant():
    main_pg = "vit__crps__large__main_run"
    variant_pg = "vit__crps__large__training_data_run"
    df = pd.DataFrame({"plot_group": [main_pg, variant_pg]})

    styles = pdc.build_family_style(
        df,
        custom_label_by_run={
            "main_run": "CRPS (main)",
            "training_data_run": "CRPS (training data)",
        },
        uniform_run_hue_color=True,
        base_first_run_hue_color=True,
        hue_group_by_run={"main_run": 0, "training_data_run": 0},
    )

    base_color = plt.get_cmap("tab10")(0)
    assert styles[main_pg]["color"] == pytest.approx(base_color)
    assert pdc._rgb_luminance(styles[variant_pg]["color"]) < pdc._rgb_luminance(
        base_color
    )


def test_run_hue_uses_explicit_fm_label_for_semantic_run_style():
    crps_pg = "semantic__unknown__large__crps_run"
    fm_pg = "semantic__unknown__large__fm_run"
    df = pd.DataFrame({"plot_group": [crps_pg, fm_pg]})

    styles = pdc.build_family_style(
        df,
        custom_label_by_run={
            "crps_run": "CRPS (training data)",
            "fm_run": "FM (training data)",
        },
        hue_group_by_run={"crps_run": 0, "fm_run": 1},
    )

    assert styles[crps_pg]["linestyle"] == "--"
    assert styles[fm_pg]["linestyle"] == "-"


def test_skill_is_backfilled_without_inventing_spread_or_overwriting_direct_values():
    legacy = pd.DataFrame({"rmse": [2.0, 4.0], "ssr": [0.5, 1.5]})

    derived = pdc._with_spread_skill_metrics(legacy)

    assert derived["skill"].tolist() == pytest.approx([2.0, 4.0])
    assert "spread" not in derived.columns
    assert derived["ssr"].tolist() == legacy["ssr"].tolist()
    assert "skill" not in legacy.columns
    assert "spread" not in legacy.columns

    direct = pd.DataFrame(
        {
            "rmse": [2.0],
            "ssr": [0.5],
            "skill": [3.0],
            "spread": [7.0],
        }
    )
    preserved = pdc._with_spread_skill_metrics(direct)
    assert preserved["skill"].tolist() == pytest.approx([3.0])
    assert preserved["spread"].tolist() == pytest.approx([7.0])


@pytest.mark.parametrize("trajectory_statistics", [False, True])
def test_lead_time_spread_requires_direct_export(
    tmp_path: Path,
    trajectory_statistics: bool,
):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    data = pd.DataFrame({"rmse": [2.0, 4.0], "ssr": [0.5, 1.5]})
    row = {
        "run_path": "run1",
        "eval_subdir": "eval",
        "dataset_label": "AD",
        "plot_group": "model",
    }
    if trajectory_statistics:
        path = eval_dir / "rollout_metrics_per_timestep_per_trajectory.csv"
        data["trajectory_id"] = ["test_0", "test_0"]
        data["lead_time"] = [0, 1]
        row["trajectory_statistics_dir"] = str(eval_dir)
    else:
        path = eval_dir / "rollout_metrics_per_timestep_channel_all.csv"

    def write_metrics():
        if trajectory_statistics:
            data.to_csv(path, index=False)
        else:
            data.T.to_csv(path)

    write_metrics()
    df = pd.DataFrame([row])
    styles = {"model": {"color": "black", "label": "model", "linestyle": "-"}}

    # The same legacy export remains usable for existing SSR figures.
    fig = pdc.plot_lead_time_panel(
        df,
        ["ssr"],
        tmp_path,
        tmp_path,
        "unused.png",
        styles,
        save=False,
    )
    assert isinstance(fig, Figure)
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [0.5, 1.5])
    plt.close(fig)

    with pytest.raises(
        ValueError, match="Direct ensemble spread is unavailable"
    ) as exc:
        pdc.plot_lead_time_panel(
            df,
            ["spread"],
            tmp_path,
            tmp_path,
            "unused.png",
            styles,
            save=False,
        )
    assert str(path) in str(exc.value)

    # These measured values intentionally differ from SSR * RMSE ([1, 6]).
    data["spread"] = [0.25, 0.75]
    write_metrics()
    fig = pdc.plot_lead_time_panel(
        df,
        ["spread"],
        tmp_path,
        tmp_path,
        "unused.png",
        styles,
        save=False,
    )
    assert isinstance(fig, Figure)
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [0.25, 0.75])
    plt.close(fig)


@pytest.mark.parametrize("paper_only", [False, True])
@pytest.mark.parametrize("spread_skill", [False, True])
def test_cli_spread_summary_requires_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    paper_only: bool,
    spread_skill: bool,
):
    run = "diff_ad64_flow_matching_vit_example"
    eval_dir = tmp_path / run / "eval"
    eval_dir.mkdir(parents=True)
    metrics = [
        "vrmse",
        "crps",
        "ssr",
        "energy",
        "psrmse_high",
        "psrmse_mid",
        "psrmse_low",
        "rmse",
        "spread",
    ]
    pd.DataFrame({metric: [0.1, 0.2] for metric in metrics}).T.to_csv(
        eval_dir / "rollout_metrics_per_timestep_channel_all.csv"
    )
    names: list[str] = []

    def capture_figure(fig: Figure, _out_dir: Path, name: str):
        names.append(name)
        plt.close(fig)

    monkeypatch.setattr(pdc, "save_fig", capture_figure)
    argv = [
        "autocast-plots",
        "--results-dir",
        str(tmp_path),
        "--run",
        run,
        "--paper-main-figures",
        "--metric-groups",
        "none",
    ]
    if paper_only:
        argv.append("--paper-only")
    if spread_skill:
        argv.append("--paper-spread-skill")
    monkeypatch.setattr(sys, "argv", argv)

    pdc.main()

    assert "paper_lead_time_panel_summary.png" in names
    assert ("paper_lead_time_panel_summary_spread_skill.png" in names) == spread_skill


def test_paper_summary_can_add_separate_spread_and_skill_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured_metrics: list[str] = []

    def fake_plot_lead_time_panel(
        _df_in: pd.DataFrame,
        metrics: list[str],
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        captured_metrics.extend(metrics)

    monkeypatch.setattr(pdc, "plot_lead_time_panel", fake_plot_lead_time_panel)
    monkeypatch.setattr(pdc, "_paper_legend", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pdc, "save_fig", lambda *_args, **_kwargs: None)

    pdc.plot_paper_lead_time_summary_figure(
        pd.DataFrame({"dataset_label": ["AD"]}),
        tmp_path,
        tmp_path,
        {},
        dataset_order=["AD"],
        include_spread_skill=True,
    )

    assert captured_metrics == [
        "vrmse",
        "crps",
        "ssr",
        "spread",
        "skill",
        "energy",
        "psrmse_high",
        "psrmse_mid",
        "psrmse_low",
    ]
    plt.close("all")


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

    assert ax.yaxis.label.get_fontsize() == 15.0
    assert ax.xaxis.get_ticklabels()[0].get_fontsize() == 15.0
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


def test_trajectory_standard_errors_use_independent_trajectories_and_gs_strata():
    ordinary = pd.DataFrame(
        {"dataset": ["advection_diffusion"] * 4, "vrmse": [1.0, 2.0, 3.0, 4.0]}
    )
    mean, se = pdc._trajectory_mean_se(ordinary, "vrmse")
    assert mean == pytest.approx(2.5)
    assert se == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1) / 2)

    gs = pd.DataFrame(
        {
            "dataset": ["gray_scott"] * 24,
            "cs0": np.repeat(np.arange(6), 4),
            "cs1": np.repeat(np.arange(6) + 10, 4),
            "vrmse": np.tile([0.0, 2.0, 0.0, 2.0], 6),
        }
    )
    mean, se = pdc._trajectory_mean_se(gs, "vrmse")
    assert mean == pytest.approx(1.0)
    assert se == pytest.approx(np.sqrt(2) / 6)

    coverage = ordinary.assign(**{"coverage_0.90": [0.7, 0.8, 0.9, 1.0]})
    summary = pdc._summarize_trajectory_metrics(coverage, ["coverage_0.9"])
    assert summary.loc[0, "metric"] == "coverage_0.9"
    assert summary.loc[0, "mean"] == pytest.approx(0.85)


def test_pooled_coverage_mae_se_uses_cross_level_covariance_and_gs_strata():
    coverage_vectors = np.array(
        [
            [0.1, 0.5],
            [0.2, 0.7],
            [0.3, 0.6],
            [0.4, 0.8],
        ]
    )
    ordinary = pd.DataFrame(
        {
            "dataset": ["advection_diffusion"] * 4,
            "trajectory_id": [f"test_{i}" for i in range(4)],
            "coverage_0.20": coverage_vectors[:, 0],
            "coverage_0.80": coverage_vectors[:, 1],
        }
    )
    projected = coverage_vectors @ np.array([0.5, -0.5])
    expected_se = projected.std(ddof=1) / np.sqrt(4)

    mean, se = pdc._pooled_coverage_mae_se(ordinary)

    assert mean == pytest.approx(0.1)
    assert se == pytest.approx(expected_se)

    gs = pd.concat(
        [
            ordinary.assign(
                dataset="gray_scott",
                trajectory_id=[f"test_{4 * h + i}" for i in range(4)],
                cs0=h,
                cs1=h + 10,
            )
            for h in range(6)
        ],
        ignore_index=True,
    )

    mean, se = pdc._pooled_coverage_mae_se(gs)

    assert mean == pytest.approx(0.1)
    assert se == pytest.approx(expected_se / np.sqrt(6))


def test_trajectory_aggregates_use_pooled_coverage_mae(tmp_path: Path):
    stats_dir = tmp_path / "trajectory_statistics"
    stats_dir.mkdir()
    coverage_vectors = np.array(
        [
            [0.1, 0.5],
            [0.2, 0.7],
            [0.3, 0.6],
            [0.4, 0.8],
        ]
    )
    nominal = np.array([0.2, 0.8])
    trajectory_coverage_mae = np.abs(coverage_vectors - nominal).mean(axis=1)
    data = pd.DataFrame(
        {
            "dataset": ["advection_diffusion"] * 4,
            "trajectory_id": [f"test_{i}" for i in range(4)],
            "mse": np.arange(4, dtype=float),
            "coverage": trajectory_coverage_mae,
            "coverage_mae": trajectory_coverage_mae,
            "coverage_0.20": coverage_vectors[:, 0],
            "coverage_0.80": coverage_vectors[:, 1],
            "window": ["all"] * 4,
        }
    )
    data.to_csv(stats_dir / "single_step_metrics_per_trajectory.csv", index=False)
    data.assign(window="[0:4)").to_csv(
        stats_dir / "rollout_metrics_per_trajectory.csv", index=False
    )
    row: dict[str, object] = {}

    pdc._add_trajectory_aggregate_metrics(row, stats_dir)

    assert np.mean(trajectory_coverage_mae) != pytest.approx(0.1)
    assert row["overall_coverage"] == pytest.approx(0.1)
    assert row["coverage_0-4"] == pytest.approx(0.1)
    assert row["overall_coverage_se"] == pytest.approx(row["coverage_0-4_se"])


def test_results_table_formats_mean_with_standard_error():
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["crps"],
            "overall_vrmse": [0.012345],
            "overall_vrmse_se": [0.001234],
            "overall_coverage": [0.04],
            "overall_coverage_se": [0.01],
        }
    )
    styles = {"crps": {"label": "CRPS", "color": "tab:blue"}}

    table = pdc.build_single_step_results_table(df, styles)

    assert table.loc[0, "VRMSE"] == "1.2e-02 (1.2e-03)"
    assert table.loc[0, "Coverage MAE"] == "0.04 (0.01)"


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
    assert r"\begin{tabular}{@{}l" in tex
    assert r"Latency\\(ms/sample)" in tex
    assert r"\small" not in tex
    assert "1.2e-02" in tex
    assert "9.9e-02" in tex
    assert "0.012345" not in tex
    assert "0.098765" not in tex


def test_output_suffix_applies_to_figures_and_tables(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(pdc, "OUTPUT_NAME_SUFFIX", ["_se"])
    monkeypatch.setattr(pdc, "FIGURE_FORMATS", ["png"])
    fig, _ = plt.subplots()

    pdc.save_fig(fig, tmp_path, "result.png")
    pdc.write_single_step_results_table(
        pd.DataFrame(
            {
                "dataset_label": ["AD"],
                "plot_group": ["crps"],
                "overall_vrmse": [0.1],
            }
        ),
        tmp_path,
        {"crps": {"label": "CRPS", "color": "tab:blue"}},
    )

    assert (tmp_path / "result_se.png").is_file()
    assert (tmp_path / "single_step_overall_results_se.csv").is_file()
    assert (tmp_path / "single_step_overall_results_se.tex").is_file()
    assert (tmp_path / "single_step_overall_results_se.md").is_file()
    assert not (tmp_path / "result.png").exists()
    assert not (tmp_path / "single_step_overall_results.csv").exists()


def test_single_step_results_latex_midrule_aligns_with_dataset_boundaries():
    """Booktabs lines: header midrule + one midrule between dataset blocks."""
    df = pd.DataFrame(
        {
            "dataset_label": ["AD", "AD", "CNS", "CNS"],
            "plot_group": ["crps", "fm", "crps", "fm"],
            "overall_vrmse": [0.2, 0.1, 0.3, 0.4],
            "overall_coverage": [0.05, 0.06, 0.07, 0.08],
            "overall_crps": [0.05, 0.06, 0.08, 0.07],
            "overall_ssr": [0.7, 1.2, 1.1, 0.6],
            "model_latency_ms_per_sample": [12.0, 8.0, 9.0, 11.0],
            "train_mean_epoch_s": [100.0, 200.0, 150.0, 50.0],
        }
    )
    styles = {
        "crps": {"label": "CRPS", "color": "tab:blue"},
        "fm": {"label": "FM", "color": "tab:orange"},
    }
    table = pdc.build_single_step_results_table(
        df,
        styles,
        dataset_order=["AD", "CNS"],
        hue_order=["CRPS", "FM"],
    )
    tex = pdc.render_single_step_results_latex(table)
    assert tex.count(r"\midrule") == 2
    assert tex.index(r"\toprule") < tex.index(r"\midrule")
    last_mid = tex.rindex(r"\midrule")
    assert last_mid < tex.index(r"\bottomrule")
    assert "\\midrule\nCNS & CRPS" in tex


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
    # siunitx S columns use \bfseries (not \textbf{...}) for detect-weight cells
    assert r"\bfseries 1.0e-01" in tex
    assert r"\bfseries 5.0e-02" in tex
    assert r"\bfseries 1.20" in tex
    assert r"\bfseries 7.0e-02" in tex
    assert r"\bfseries 1.10" in tex


def test_single_step_results_markdown_is_reviewer_ready(tmp_path: Path):
    df = pd.DataFrame(
        {
            "dataset_label": ["AD", "AD"],
            "plot_group": ["crps", "fm"],
            "overall_vrmse": [0.012345, 0.023456],
            "overall_coverage": [0.04, 0.03],
            "overall_crps": [0.098765, 0.087654],
            "overall_ssr": [0.9, 1.2],
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
        dataset_order=["AD"],
        hue_order=["CRPS", "FM"],
    )

    markdown = (tmp_path / "single_step_overall_results.md").read_text()
    assert (
        "| Dataset | Model | VRMSE ↓ | Coverage MAE ↓ | CRPS ↓ | SSR → 1 |" in markdown
    )
    assert "| AD | CRPS | **1.2e-02** | 0.04 | 9.9e-02 | **0.90** |" in markdown
    assert "| AD | FM | 2.3e-02 | **0.03** | **8.8e-02** | 1.20 |" in markdown


def test_rollout_window_summary_markdown_is_reviewer_ready(tmp_path: Path):
    df = pd.DataFrame(
        {
            "dataset_label": ["AD", "AD"],
            "plot_group": ["cln", "fno"],
            "vrmse_0-4": [0.001, 0.002],
            "vrmse_31-99": [0.01, 0.2],
            "coverage_0-4": [0.04, 0.08],
            "coverage_31-99": [0.12, 0.30],
        }
    )
    styles = {
        "cln": {"label": "CLN", "color": "tab:blue"},
        "fno": {"label": "FNO", "color": "tab:red"},
    }

    pdc.write_rollout_window_summary_table(
        df,
        tmp_path,
        styles,
        dataset_order=["AD"],
        hue_order=["CLN", "FNO"],
    )

    markdown = (tmp_path / "rollout_window_summary_results.md").read_text()
    assert "VRMSE [0:4) ↓" in markdown
    assert "Coverage MAE [31:99) ↓" in markdown
    assert "| AD | CLN | **1.0e-03** | **1.0e-02** | **0.04** | **0.12** |" in markdown
    assert "| AD | FNO | 2.0e-03 | 2.0e-01 | 0.08 | 0.30 |" in markdown


def test_load_single_run_metrics_falls_back_to_rollout_coverage_curve(
    tmp_path: Path,
):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "coverage_level": [0.1, 0.5, 0.9],
            "observed_mean": [0.0, 0.3, 0.8],
        }
    ).to_csv(eval_dir / "rollout_coverage_window_31-99.csv", index=False)

    row = pdc.load_single_run_metrics(tmp_path / "run1")

    assert row["coverage_31-99"] == pytest.approx((0.1 + 0.2 + 0.1) / 3)


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

    assert isinstance(fig, Figure)
    axes = fig.axes
    assert axes[0].get_ylabel() == "Empirical coverage"
    assert axes[1].get_ylabel() == "Empirical coverage [0:4)"
    assert axes[1].get_xlabel() == r"Nominal coverage (1 - $\alpha$)"
    # Perfect-calibration reference: y = x from (0, 0) to (1, 1), drawn first
    xd = axes[0].lines[0].get_xdata()
    yd = axes[0].lines[0].get_ydata()
    assert np.allclose(np.asarray(xd, dtype=float), np.array([0.0, 1.0]))
    assert np.allclose(np.asarray(yd, dtype=float), np.array([0.0, 1.0]))
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

    assert isinstance(fig, Figure)
    assert fig.get_size_inches()[1] == 2.3 * 2 * 1.5
    assert fig.axes[1].get_xlabel() == ""
    assert r"Nominal coverage (1 - $\alpha$)" in [text.get_text() for text in fig.texts]
    plt.close(fig)


def test_lead_time_coverage_delta_is_observed_minus_nominal(tmp_path: Path):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        [[0.25, 0.75]],
        index=pd.Index(["coverage_0.5"], name="metric"),
        columns=pd.Index(["0", "1"]),
    ).to_csv(eval_dir / "rollout_metrics_per_timestep_channel_all.csv")
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

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_ylabel() == ""
    assert r"Empirical coverage minus nominal" in [
        text.get_text() for text in fig.texts
    ]
    assert np.asarray(ax.lines[0].get_ydata(), dtype=float).tolist() == [-0.25, 0.25]
    plt.close(fig)


def test_short_axis_labels_use_compact_shared_coverage_delta(tmp_path: Path):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        [[0.25, 0.75]],
        index=pd.Index(["coverage_0.5"], name="metric"),
        columns=pd.Index(["0", "1"]),
    ).to_csv(eval_dir / "rollout_metrics_per_timestep_channel_all.csv")
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

    assert isinstance(fig, Figure)
    assert r"$\Delta$ empirical coverage" in [text.get_text() for text in fig.texts]
    plt.close(fig)


def test_lead_time_error_labels_are_uppercase(tmp_path: Path):
    eval_dir = tmp_path / "run1" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.Index(["vrmse"], name="metric"),
        columns=pd.Index(["0", "1"]),
    ).to_csv(eval_dir / "rollout_metrics_per_timestep_channel_all.csv")
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

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_ylabel() == "VRMSE"
    plt.close(fig)


@pytest.mark.parametrize("show_error_bands", [True, False])
def test_trajectory_bands_can_be_hidden_without_changing_means(
    tmp_path: Path,
    show_error_bands: bool,
):
    stats_dir = tmp_path / "trajectory_statistics"
    stats_dir.mkdir()
    pd.DataFrame(
        {
            "dataset": ["advection_diffusion"] * 8,
            "trajectory_id": [f"test_{i}" for i in range(4)] * 2,
            "mse": np.arange(8, dtype=float),
            "vrmse": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
            "lead_time": np.repeat([0, 1], 4),
        }
    ).to_csv(stats_dir / "rollout_metrics_per_timestep_per_trajectory.csv", index=False)
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "run_path": ["run1"],
            "eval_subdir": ["eval"],
            "trajectory_statistics_dir": [str(stats_dir)],
        }
    )
    styles = {
        "model": {
            "color": "black",
            "label": "model",
            "linestyle": "-",
            "show_error_bands": show_error_bands,
        }
    }

    fig = pdc.plot_lead_time_panel(
        df,
        ["vrmse"],
        tmp_path,
        tmp_path,
        "lead_time.png",
        styles,
        save=False,
    )

    assert isinstance(fig, Figure)
    assert np.asarray(fig.axes[0].lines[0].get_ydata()).tolist() == [2.5, 3.5]
    assert len(fig.axes[0].collections) == int(show_error_bands)
    if show_error_bands:
        band = fig.axes[0].collections[0]
        assert isinstance(band, PolyCollection)
        assert np.asarray(band.get_linewidth()).tolist() == [0]
    plt.close(fig)

    pd.DataFrame(
        {
            "dataset": ["advection_diffusion"] * 4,
            "trajectory_id": [f"test_{i}" for i in range(4)],
            "coverage_0.20": [0.1, 0.2, 0.3, 0.4],
            "coverage_0.80": [0.5, 0.6, 0.7, 0.8],
        }
    ).to_csv(stats_dir / "single_step_metrics_per_trajectory.csv", index=False)
    fig = pdc.plot_coverage_calibration_panel(
        df,
        tmp_path,
        tmp_path,
        styles,
        window_rows=["all"],
        save=False,
    )
    assert isinstance(fig, Figure)
    # The first line is the diagonal reference; the second is the model curve.
    np.testing.assert_allclose(
        np.asarray(fig.axes[0].lines[1].get_ydata()), [0.25, 0.65]
    )
    assert len(fig.axes[0].collections) == int(show_error_bands)
    plt.close(fig)


@pytest.mark.parametrize("nominal", [0.1, 0.5, 0.9])
def test_coverage_difference_preserves_standard_error(tmp_path: Path, nominal: float):
    stats = tmp_path / "statistics"
    stats.mkdir()
    metric = f"coverage_{nominal}"
    pd.DataFrame(
        {
            "dataset": ["advection_diffusion"] * 4,
            "trajectory_id": ["a", "b", "a", "b"],
            "lead_time": [0, 0, 1, 1],
            metric: [0.2, 0.4, 0.6, 0.8],
        }
    ).to_csv(stats / "rollout_metrics_per_timestep_per_trajectory.csv", index=False)
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "run_path": ["run1"],
            "eval_subdir": ["eval"],
            "trajectory_statistics_dir": [str(stats)],
        }
    )
    fig = pdc.plot_lead_time_panel(
        df,
        [metric],
        tmp_path,
        tmp_path,
        "difference.png",
        {"model": {"color": "black", "label": "model"}},
        coverage_delta=True,
        save=False,
    )
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    np.testing.assert_allclose(
        np.asarray(ax.lines[0].get_ydata(), dtype=float), np.array([0.3, 0.7]) - nominal
    )
    vertices = np.asarray(ax.collections[0].get_paths()[0].vertices, dtype=float)
    for time, mean in enumerate([0.3, 0.7]):
        bounds = vertices[vertices[:, 0] == time, 1]
        np.testing.assert_allclose(
            [bounds.min(), bounds.max()], np.array([mean - 0.1, mean + 0.1]) - nominal
        )
    plt.close(fig)


@pytest.mark.parametrize("axis_sharing", [False, "row", True])
def test_shared_coverage_difference_axes_include_every_nominal_level(
    tmp_path: Path, axis_sharing: bool | Literal["row"]
):
    folder = tmp_path / "run1" / "eval"
    folder.mkdir(parents=True)
    pd.DataFrame(
        [[0.1, 0.2], [0.11, 0.12]],
        index=pd.Index(["coverage_0.9", "coverage_0.1"]),
        columns=pd.Index([0, 1]),
    ).to_csv(folder / "rollout_metrics_per_timestep_channel_all.csv")
    df = pd.DataFrame(
        {
            "dataset_label": ["AD"],
            "plot_group": ["model"],
            "run_path": ["run1"],
            "eval_subdir": ["eval"],
        }
    )
    fig, axes = plt.subplots(2, 1, sharey=axis_sharing, squeeze=False)
    pdc.plot_lead_time_panel(
        df,
        ["coverage_0.9", "coverage_0.1"],
        tmp_path,
        tmp_path,
        "difference.png",
        {"model": {"color": "black", "label": "model"}},
        coverage_delta=True,
        axes=axes,
        fig=fig,
        sharey=True,
        save=False,
    )
    for ax in fig.axes:
        assert ax.get_ylim()[0] < -0.8
        assert ax.get_ylim()[1] > 0.02
    plt.close(fig)


@pytest.mark.parametrize("max_error", [0.02, 0.8])
def test_coverage_scale_is_local_to_panel_and_excludes_other_metrics(
    tmp_path: Path, max_error: float
):
    folder = tmp_path / "run1" / "eval"
    folder.mkdir(parents=True)
    pd.DataFrame(
        [[0.9 - max_error, 0.9], [0.1, 0.11], [10.0, 20.0]],
        index=pd.Index(["coverage_0.9", "coverage_0.1", "vrmse"]),
        columns=pd.Index([0, 1]),
    ).to_csv(folder / "rollout_metrics_per_timestep_channel_all.csv")
    frame = pd.DataFrame(
        {"dataset_label": ["AD"], "plot_group": ["model"], "run_path": ["run1"]}
    )
    fig = pdc.plot_lead_time_panel(
        frame,
        ["coverage_0.9", "coverage_0.1", "vrmse"],
        tmp_path,
        tmp_path,
        "difference.png",
        {"model": {"color": "black", "label": "model"}},
        coverage_delta=True,
        save=False,
    )
    assert isinstance(fig, Figure)
    expected = 1.1 * max_error
    for ax in fig.axes[:2]:
        np.testing.assert_allclose(ax.get_ylim(), [-expected, expected])
        assert ax.get_yscale() == "linear"
    assert fig.axes[2].get_yscale() == "log"
    assert fig.axes[2].get_ylim()[1] > 20
    plt.close(fig)
