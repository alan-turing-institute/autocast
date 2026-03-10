"""Utility functions for AutoCast scripts."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import uuid
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import yaml


def default_run_name(prefix: str = "run") -> str:
    """Generate a short default run name when none is provided."""
    return f"{prefix}_{uuid.uuid4().hex[:7]}"


def resolve_work_dir(
    *,
    output_base: str | Path = "outputs",
    run_group: str | None = None,
    run_id: str | None = None,
    work_dir: str | Path | None = None,
    prefix: str = "run",
) -> tuple[Path, str]:
    """Resolve final work directory and run name.

    Priority:
    1. If ``work_dir`` is provided, use it directly.
    2. Otherwise build ``<output_base>/<run_group>/<run_id>``.
    3. If ``run_id`` is missing, generate a short default.

    Parameters
    ----------
    run_group
        Preferred top-level output folder grouping label.
    run_id
        Preferred run folder identifier.

    """
    if work_dir is not None:
        resolved = Path(work_dir).expanduser().resolve()
        return resolved, (run_id or resolved.name)

    resolved_group = run_group or datetime.now().strftime("%Y-%m-%d")
    resolved_id = run_id or default_run_name(prefix=prefix)
    resolved = (Path(output_base) / resolved_group / resolved_id).expanduser().resolve()
    return resolved, resolved_id


def get_default_config_path() -> str:
    """Resolve default Hydra config path.

    Resolution order:
    1. ``AUTOCAST_CONFIG_PATH`` environment variable (if set and exists).
    2. Repository ``src/autocast/configs/`` directory (detected via ``pyproject.toml``).
    3. Packaged ``autocast/configs`` resources (for wheel/sdist installs).

    This allows local development with repository configs while supporting
    installed-package layouts.

    Returns
    -------
    str
        Absolute path to the configs directory.

    Raises
    ------
    FileNotFoundError
        If no valid config directory can be resolved.
    """
    env_path = os.environ.get("AUTOCAST_CONFIG_PATH")
    if env_path:
        config_dir = Path(env_path).expanduser().resolve()
        if config_dir.exists():
            return str(config_dir)
        msg = f"AUTOCAST_CONFIG_PATH was set but does not exist: {config_dir}"
        raise FileNotFoundError(msg)

    current = Path(__file__).resolve().parent
    while current != current.parent:  # Stop at filesystem root
        if (current / "pyproject.toml").exists():
            config_dir = current / "src" / "autocast" / "configs"
            if not config_dir.exists():
                msg = f"Project root found at {current}, but configs directory missing"
                raise FileNotFoundError(msg)
            return str(config_dir)
        current = current.parent

    try:
        packaged_configs = resources.files("autocast") / "configs"
        if packaged_configs.is_dir():
            return str(Path(str(packaged_configs)).resolve())
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    msg = (
        "Could not resolve configs directory. Set AUTOCAST_CONFIG_PATH or install "
        "package data including autocast/configs."
    )
    raise FileNotFoundError(msg)


class RunCollator:
    """Collate results from multiple runs in an outputs directory.

    Supports glob-like pattern matching for flexible config parameter extraction.

    Parameters
    ----------
    outputs_dir : str | Path, default="outputs"
        Path to the outputs directory containing runs.
    config_params : dict[str, str] | None, default=None
        Dictionary mapping output column names to config path patterns.
        If None, uses default parameters. Supports glob wildcards:
        - Exact paths: "model.encoder._target_"
        - Wildcards: "model.*.hidden_dim" (matches any single key)
        - End wildcards: "model.processor.hidden_*" (matches any suffix)

    Examples
    --------
    >>> # Use with wildcard patterns
    >>> collator = RunCollator(
    ...     config_params={
    ...         "encoder": "model.encoder._target_",
    ...         "hidden_size": "model.processor.hidden_*",  # Flexible matching
    ...     }
    ... )
    >>> df = collator.collate()
    """

    # Default parameters to extract from config files
    # Supports glob patterns (*, ?) for flexible matching across varying configs
    DEFAULT_CONFIG_PARAMS: ClassVar[dict[str, str]] = {
        "encoder": "model.encoder._target_",
        "processor": "model.processor._target_",
        "decoder": "model.decoder._target_",
        # Matches hidden_dim, hidden_channels, etc.
        "hidden_size": "model.processor.hidden_*",
        "n_members": "model.n_members",
        "batch_size": "datamodule.batch_size",
        "use_normalization": "datamodule.use_normalization",
        "learning_rate": "optimizer.learning_rate",
        "loss_function": "model.loss_func._target_",
        "train_in_latent": "model.train_in_latent_space",
    }

    def __init__(
        self,
        outputs_dir: str | Path = "outputs",
        config_params: dict[str, str] | None = None,
    ):
        """Initialize the RunCollator."""
        self.log = logging.getLogger(__name__)

        # Resolve outputs directory
        self.outputs_path = Path(outputs_dir).expanduser().resolve()
        if not self.outputs_path.exists():
            msg = f"Outputs directory not found: {self.outputs_path}"
            raise FileNotFoundError(msg)

        # Set config parameters
        self.config_params = config_params or self.DEFAULT_CONFIG_PARAMS

    def _discover_runs(self) -> list[Path]:
        """Discover all valid run directories.

        A valid run must have:
        - resolved_config.yaml
        - eval/evaluation_metrics.csv

        Returns
        -------
        list[Path]
            List of run directory paths, sorted by modification time.
        """
        runs = []
        for config_file in self.outputs_path.rglob("resolved_config.yaml"):
            run_dir = config_file.parent
            eval_metrics = run_dir / "eval" / "evaluation_metrics.csv"
            if eval_metrics.exists():
                runs.append(run_dir)
        return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)

    def _extract_run_metadata(self, run_dir: Path) -> dict[str, str]:
        """Extract run metadata (name, date, path, category).

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        dict[str, str]
            Dictionary with run_path, run_name, date, and category.
        """
        metadata = {}

        # Relative path from outputs
        metadata["run_path"] = str(run_dir.relative_to(self.outputs_path))

        # Run name (directory name)
        metadata["run_name"] = run_dir.name

        # Try to extract date from parent directory
        parent_name = run_dir.parent.name
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", parent_name)
        if date_match:
            metadata["date"] = date_match.group(1)
            metadata["category"] = "date-based"
        elif parent_name == "processor":
            # Extract from run name (YYYYMMDD_HHMMSS)
            timestamp_match = re.match(r"(\d{8})_(\d{6})", run_dir.name)
            if timestamp_match:
                yyyymmdd = timestamp_match.group(1)
                metadata["date"] = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
            else:
                metadata["date"] = "unknown"
            metadata["category"] = "processor"
        elif parent_name == "autoencoder":
            # Extract from run name (YYYY-MM-DD_HH-MM-SS)
            date_match = re.match(r"(\d{4}-\d{2}-\d{2})", run_dir.name)
            if date_match:
                metadata["date"] = date_match.group(1)
            else:
                metadata["date"] = "unknown"
            metadata["category"] = "autoencoder"
        else:
            metadata["date"] = "unknown"
            metadata["category"] = "other"

        return metadata

    def _find_matching_paths(self, config: dict, pattern: str) -> list[tuple[str, Any]]:
        """Find all config paths matching the glob pattern.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        pattern : str
            Dot-separated pattern with optional wildcards.
            Examples: "model.encoder._target_", "model.*.hidden_dim"

        Returns
        -------
        list[tuple[str, Any]]
            List of (path, value) tuples for all matching paths.
        """
        parts = pattern.split(".")
        results = []

        def _traverse(
            obj: Any, remaining_parts: list[str], current_path: str = ""
        ) -> None:
            """Recursively traverse and match pattern parts."""
            if not remaining_parts:
                # Reached end of pattern - this is a match
                results.append((current_path.lstrip("."), obj))
                return

            if not isinstance(obj, dict):
                # Can't traverse further
                return

            current_pattern = remaining_parts[0]
            rest = remaining_parts[1:]

            # Check if pattern contains wildcards
            if "*" in current_pattern or "?" in current_pattern:
                # Wildcard - try matching against all keys
                for key in obj:
                    if fnmatch.fnmatch(key, current_pattern):
                        _traverse(obj[key], rest, f"{current_path}.{key}")
            # Exact match
            elif current_pattern in obj:
                _traverse(
                    obj[current_pattern], rest, f"{current_path}.{current_pattern}"
                )

        _traverse(config, parts)
        return results

    def _get_nested_value(self, config: dict, path: str, default: Any = "N/A") -> Any:
        """Get a value from nested dictionary using glob patterns.

        Supports both exact paths and wildcard patterns:
        - "model.encoder._target_" (exact match)
        - "model.*.hidden_dim" (wildcard - matches any key)
        - "model.processor.hidden_*" (wildcard at end)

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        path : str
            Dot-separated path with optional wildcards
            (e.g., "model.encoder._target_" or "model.*.hidden_dim").
        default : Any, default="N/A"
            Default value if path not found.

        Returns
        -------
        Any
            Value at the first matching path, or default if not found.
        """
        matches = self._find_matching_paths(config, path)
        if matches:
            return matches[0][1]  # Return first match value
        return default

    def _simplify_target_name(self, target: str) -> str:
        """Simplify a _target_ value by extracting the last component.

        Parameters
        ----------
        target : str
            Full target path (e.g., "autocast.models.encoder.Encoder").

        Returns
        -------
        str
            Simplified name (e.g., "Encoder"), or "Unknown" if empty.
        """
        if not target or target == "N/A":
            return "Unknown"
        return target.split(".")[-1]

    def _extract_config_params(self, config: dict) -> dict[str, Any]:
        """Extract specified parameters from config using configured paths.

        Parameters
        ----------
        config : dict
            Loaded configuration dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with extracted parameter values.
        """
        info = {}

        for param_name, config_path in self.config_params.items():
            value = self._get_nested_value(config, config_path)

            # Simplify _target_ values
            if config_path.endswith("._target_") and value != "N/A":
                value = self._simplify_target_name(value)

            info[param_name] = value

        return info

    def _extract_dataset_from_config(self, config: dict) -> str | None:
        """Extract dataset name from config (datamodule or dataset).

        Mirrors infer_dataset_from_workdir logic for use when config is already loaded.
        """
        datamodule_cfg = config.get("datamodule")
        if isinstance(datamodule_cfg, str):
            return datamodule_cfg
        if isinstance(datamodule_cfg, dict):
            dataset_name = datamodule_cfg.get("dataset")
            if isinstance(dataset_name, str) and dataset_name:
                return dataset_name
            data_path = datamodule_cfg.get("data_path")
            if data_path is not None:
                try:
                    return Path(str(data_path)).name
                except (TypeError, ValueError):
                    pass
        top_level = config.get("dataset")
        if isinstance(top_level, str) and top_level:
            return top_level
        return None

    def _parse_metrics(self, run_dir: Path) -> dict[str, Any]:  # noqa: PLR0912
        """Parse both evaluation_metrics.csv and rollout_metrics.csv.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        dict[str, Any]
            Dictionary with overall metrics and windowed metrics.
        """
        metrics: dict[str, Any] = {}

        # Base metric columns: support both scalar and multi-level coverage
        base_cols = ["mse", "mae", "rmse", "vrmse"]

        # Parse overall metrics
        eval_csv = run_dir / "eval" / "evaluation_metrics.csv"
        if eval_csv.exists():
            try:
                df_eval = pd.read_csv(eval_csv)
                # Get row where window='all' and batch_idx='all'
                overall_row = df_eval[
                    (df_eval["window"] == "all") & (df_eval["batch_idx"] == "all")
                ]
                if not overall_row.empty:
                    for col in base_cols:
                        if col in overall_row.columns:
                            value = overall_row.iloc[0][col]
                            metrics[f"overall_{col}"] = value
                    # Coverage: support "coverage" or coverage_0.5, coverage_0.9, etc.
                    cov_cols_found = [
                        c
                        for c in overall_row.columns
                        if c == "coverage" or c.startswith("coverage_")
                    ]
                    for col in cov_cols_found:
                        value = overall_row.iloc[0][col]
                        suffix = col.replace("coverage", "", 1) or ""
                        metrics[f"overall_coverage{suffix}"] = value
            except Exception as e:
                self.log.warning(
                    "Failed to parse evaluation metrics from %s: %s", eval_csv, e
                )

        # Parse windowed metrics
        rollout_csv = run_dir / "eval" / "rollout_metrics.csv"
        if rollout_csv.exists():
            try:
                df_rollout = pd.read_csv(rollout_csv)
                # Get rows where batch_idx='all'
                windowed = df_rollout[df_rollout["batch_idx"] == "all"]
                for _, row in windowed.iterrows():
                    window = str(row["window"])
                    for col in base_cols:
                        if col in row.index:
                            metrics[f"{col}_{window}"] = float(row[col])
                    cov_cols_found = [
                        c
                        for c in row.index
                        if c == "coverage"
                        or (isinstance(c, str) and c.startswith("coverage_"))
                    ]
                    for col in cov_cols_found:
                        try:
                            val = row[col]
                            if not isinstance(col, str):
                                continue
                            suffix = col.replace("coverage", "", 1) or ""
                            metrics[f"coverage{suffix}_{window}"] = float(val)
                        except (TypeError, ValueError):
                            pass
            except Exception as e:
                self.log.warning(
                    "Failed to parse rollout metrics from %s: %s", rollout_csv, e
                )

        return metrics

    def _parse_metadata_csvs(self, run_dir: Path) -> pd.DataFrame:
        """Parse evaluation/rollout metadata CSVs for a single run.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        pd.DataFrame
            Combined metadata DataFrame for the run. Empty if no metadata CSVs
            are present or parseable.
        """
        metadata_frames: list[pd.DataFrame] = []

        metadata_sources = {
            "evaluation": run_dir / "eval" / "evaluation_metadata.csv",
            "rollout": run_dir / "eval" / "rollout_metadata.csv",
            "benchmark": run_dir / "eval" / "benchmark_metrics.csv",
        }

        for source_name, csv_path in metadata_sources.items():
            if not csv_path.exists():
                continue
            try:
                df_source = pd.read_csv(csv_path)
                if df_source.empty:
                    continue
                df_source = df_source.copy()
                df_source["metadata_source"] = source_name
                metadata_frames.append(df_source)
            except Exception as e:
                self.log.warning("Failed to parse metadata CSV %s: %s", csv_path, e)

        if not metadata_frames:
            return pd.DataFrame()

        df_metadata = pd.concat(metadata_frames, ignore_index=True)
        run_metadata = self._extract_run_metadata(run_dir)
        for key, value in run_metadata.items():
            df_metadata[key] = value

        return df_metadata

    def _process_single_run(self, run_dir: Path) -> dict[str, Any] | None:
        """Process a single run directory, handling errors gracefully.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        dict[str, Any] | None
            Dictionary of run data, or None if processing fails.
        """
        try:
            # Extract metadata
            run_data = self._extract_run_metadata(run_dir)

            # Load and parse config
            config_file = run_dir / "resolved_config.yaml"
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                config_info = self._extract_config_params(config)
                run_data.update(config_info)
                # Extract dataset from datamodule config
                dataset = self._extract_dataset_from_config(config)
                if dataset:
                    run_data["dataset"] = dataset
            except Exception as e:
                self.log.warning("Failed to parse config for %s: %s", run_dir.name, e)
                # Continue with missing config info

            # Parse metrics
            try:
                metrics = self._parse_metrics(run_dir)
                run_data.update(metrics)
            except Exception as e:
                self.log.warning("Failed to parse metrics for %s: %s", run_dir.name, e)
                # Continue with missing metrics

            return run_data

        except Exception as e:
            self.log.error("Failed to process run %s: %s", run_dir.name, e)
            return None

    def _save_results(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save DataFrame to CSV following existing patterns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save.
        output_path : Path
            Path where the CSV file will be saved.
        """
        if df.empty:
            self.log.warning("No runs to save; DataFrame is empty.")
            return

        # Sort by date (descending) and run name
        if "date" in df.columns:
            df = df.sort_values(["date", "run_name"], ascending=[False, True])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        self.log.info("Saved collated results to %s", output_path)

    def collate(
        self,
        output_csv: str | Path = "collated_results.csv",
        save_csv: bool = True,
        include_metadata_dataframes: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Collate results from multiple runs in the outputs directory.

        Parameters
        ----------
        output_csv : str | Path, default="collated_results.csv"
            Path where the CSV file will be saved.
        save_csv : bool, default=True
            Whether to save the DataFrame to CSV.
        include_metadata_dataframes : bool, default=False
            Whether to also return per-run metadata DataFrames loaded from
            ``eval/evaluation_metadata.csv`` and ``eval/rollout_metadata.csv``.

        Returns
        -------
        pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]
            If ``include_metadata_dataframes`` is False, returns the collated
            metrics DataFrame. If True, returns a tuple of:
            1) collated metrics DataFrame
            2) dict mapping run_path -> metadata DataFrame for that run
        """
        # Discover runs
        runs = self._discover_runs()
        self.log.info("Found %d valid runs", len(runs))

        # Process each run
        results = []
        metadata_by_run: dict[str, pd.DataFrame] = {}
        for run_dir in runs:
            run_data = self._process_single_run(run_dir)
            if run_data is not None:
                results.append(run_data)

                if include_metadata_dataframes:
                    run_path = str(run_dir.relative_to(self.outputs_path))
                    df_metadata = self._parse_metadata_csvs(run_dir)
                    if not df_metadata.empty:
                        metadata_by_run[run_path] = df_metadata

        # Create DataFrame
        df = pd.DataFrame(results)
        self.log.info("Successfully processed %d runs", len(df))

        # Save to CSV
        if save_csv:
            output_path = Path(output_csv).expanduser().resolve()
            self._save_results(df, output_path)

        if include_metadata_dataframes:
            return df, metadata_by_run

        return df


def collate_run_results(
    outputs_dir: str | Path = "outputs",
    output_csv: str | Path = "collated_results.csv",
    save_csv: bool = True,
    config_params: dict[str, str] | None = None,
    include_metadata_dataframes: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Collate results from multiple runs in the outputs directory.

    This is a convenience function that creates a RunCollator instance
    and calls its collate method.

    Parameters
    ----------
    outputs_dir : str | Path, default="outputs"
        Path to the outputs directory containing runs.
    output_csv : str | Path, default="collated_results.csv"
        Path where the CSV file will be saved.
    save_csv : bool, default=True
        Whether to save the DataFrame to CSV.
    config_params : dict[str, str] | None, default=None
        Dictionary mapping output column names to config paths (dot notation).
        If None, uses default parameters.
    include_metadata_dataframes : bool, default=False
        Whether to also return per-run metadata DataFrames loaded from
        ``eval/evaluation_metadata.csv`` and ``eval/rollout_metadata.csv``.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]
        If ``include_metadata_dataframes`` is False, returns the collated
        metrics DataFrame. If True, returns ``(metrics_df, metadata_by_run)``.
    """
    collator = RunCollator(outputs_dir=outputs_dir, config_params=config_params)
    return collator.collate(
        output_csv=output_csv,
        save_csv=save_csv,
        include_metadata_dataframes=include_metadata_dataframes,
    )


def flatten_metadata_by_run(
    metadata_by_run: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Flatten per-run metadata frames into one wide row per run.

    This helper is intended for notebooks/analysis. It consumes the dict returned
    by ``RunCollator.collate(include_metadata_dataframes=True)`` and produces a
    wide DataFrame with columns such as:
    - training time: ``train_total_s``, ``train_mean_epoch_s``
    - params: ``params_model_total``, ``params_model_trainable``, etc.
    - benchmark: ``model_latency_ms_per_sample``, ``model_throughput_samples_per_sec``,
      ``model_gflops_per_sample``, and analogous ``rollout_*`` fields

    Notes
    -----
    - If multiple benchmark rows exist for the same (benchmark_type, metric),
      values are averaged.
    - Only rows with ``window == 'meta'`` and ``batch_idx == 'all'`` are used for
      evaluation/rollout metadata.
    """
    rows: list[dict[str, Any]] = []
    for run_path, df_meta in metadata_by_run.items():
        if df_meta is None or df_meta.empty:
            rows.append({"run_path": run_path})
            continue

        row: dict[str, Any] = {"run_path": run_path}

        # Carry through identifiers when present.
        for col in ["run_name", "date"]:
            if col in df_meta.columns:
                vals = df_meta[col].dropna().unique()
                if len(vals) > 0:
                    row[col] = vals[0]

        # evaluation / rollout metadata: long format (category, metric, value)
        if "metadata_source" in df_meta.columns and "value" in df_meta.columns:
            mask_base = True
            if "window" in df_meta.columns:
                mask_base = mask_base & (df_meta["window"] == "meta")
            if "batch_idx" in df_meta.columns:
                mask_base = mask_base & (df_meta["batch_idx"] == "all")

            df_er = df_meta[
                mask_base
                & df_meta["metadata_source"].isin(["evaluation", "rollout"])
                & df_meta["category"].notna()
                & df_meta["metric"].notna()
            ].copy()
            if not df_er.empty:
                df_er["key"] = (
                    df_er["metadata_source"].astype(str)
                    + "_"
                    + df_er["category"].astype(str)
                    + "_"
                    + df_er["metric"].astype(str)
                )
                wide = (
                    df_er.groupby("key", as_index=True)["value"]
                    .mean(numeric_only=True)
                    .to_dict()
                )
                row.update(wide)

        # benchmark metadata: long-ish format (benchmark_type, metric, value)
        if (
            "metadata_source" in df_meta.columns
            and "benchmark_type" in df_meta.columns
            and "metric" in df_meta.columns
            and "value" in df_meta.columns
        ):
            df_bench = df_meta[
                (df_meta["metadata_source"] == "benchmark")
                & df_meta["benchmark_type"].notna()
                & df_meta["metric"].notna()
                & df_meta["value"].notna()
            ].copy()
        else:
            df_bench = pd.DataFrame()
        if not df_bench.empty:
            df_bench["key"] = (
                df_bench["benchmark_type"].astype(str)
                + "_"
                + df_bench["metric"].astype(str)
            )
            wide = (
                df_bench.groupby("key", as_index=True)["value"]
                .mean(numeric_only=True)
                .to_dict()
            )
            row.update(wide)

        rows.append(row)

    df_wide = pd.DataFrame(rows)

    # Friendly aliases for commonly used metrics (if present).
    rename_map = {
        "evaluation_runtime_train_total_s": "train_total_s",
        "evaluation_runtime_train_mean_epoch_s": "train_mean_epoch_s",
        "evaluation_runtime_train_min_epoch_s": "train_min_epoch_s",
        "evaluation_runtime_train_max_epoch_s": "train_max_epoch_s",
        "evaluation_params_model_total": "params_model_total",
        "evaluation_params_model_trainable": "params_model_trainable",
        "evaluation_params_encoder_total": "params_encoder_total",
        "evaluation_params_decoder_total": "params_decoder_total",
        "evaluation_params_processor_total": "params_processor_total",
    }
    df_wide = df_wide.rename(
        columns={k: v for k, v in rename_map.items() if k in df_wide.columns}
    )

    return df_wide
