"""Unit tests for evaluation batch-limit resolution."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf

from autocast.metrics.ensemble import CRPS, AlphaFairCRPS, SpreadSkillRatio
from autocast.scripts.eval.encoder_processor_decoder import (
    DEFAULT_EVAL_METRICS,
    DEFAULT_EVAL_MODE,
    EVAL_PATH_AMBIENT_EPD,
    EVAL_PATH_ENCODE_ONCE,
    EVAL_PATH_LATENT_CACHED_LATENT_ONLY,
    EVAL_PATH_LATENT_CACHED_WITH_DECODER,
    _build_eval_predict_fn,
    _build_per_timestep_metric_factory,
    _decode_tensor,
    _maybe_swap_to_ambient_datamodule,
    _normalize_eval_mode,
    _normalize_per_batch_rows,
    _reindex_per_batch_rows_by_rank,
    _render_rollouts,
    _require_decoder_unless_latent_metrics_opt_in,
    _resolve_auto_eval_mode,
    _resolve_eval_path,
    _resolve_rollout_batch_limit,
    _resolve_rollout_channel_names,
    _resolve_rollout_timestep_limit,
    _should_skip_metric,
    _split_metric_and_metadata_rows,
    _training_runtime_rows,
    _validate_latent_space_metrics_flag,
    _validate_resolved_eval_path,
)
from autocast.types import Batch, EncodedBatch


def test_resolve_rollout_batch_limit_falls_back_to_test_limit_when_null():
    eval_cfg = OmegaConf.create(
        {
            "max_test_batches": 2,
            "max_rollout_batches": None,
        }
    )

    assert _resolve_rollout_batch_limit(eval_cfg) == 2


def test_resolve_rollout_batch_limit_prefers_explicit_rollout_limit():
    eval_cfg = OmegaConf.create(
        {
            "max_test_batches": 2,
            "max_rollout_batches": 5,
        }
    )

    assert _resolve_rollout_batch_limit(eval_cfg) == 5


def test_build_eval_predict_fn_uses_predict_for_wrapped_processor_model():
    batch = SimpleNamespace(
        encoded_output_fields=torch.randn(2, 4, 8, 8, 3),
    )
    expected = torch.randn(2, 4, 8, 8, 3, 10)

    class WrappedProcessorModel:
        def _predict(self, arg):
            assert arg is batch
            return expected

        def __call__(self, _arg):
            msg = "predict_fn should not call wrapped model directly"
            raise AssertionError(msg)

    predict_fn = _build_eval_predict_fn(
        WrappedProcessorModel(),
        is_processor_model=True,
        decode_fn=None,
    )

    preds, trues = predict_fn(batch)

    assert preds is expected
    assert trues is batch.encoded_output_fields


def test_build_eval_predict_fn_ensemble_expansion():
    """When n_members > 1, predict_fn expands the batch and adds ensemble dim."""

    inputs = torch.randn(2, 1, 8, 8, 3)
    targets = torch.randn(2, 4, 8, 8, 3)
    batch = EncodedBatch(
        encoded_inputs=inputs,
        encoded_output_fields=targets,
        global_cond=None,
        encoded_info={},
    )

    class FakeProcessorModel:
        def _predict(self, b):
            # Just return a tensor of the right batch shape
            return torch.randn(b.encoded_inputs.shape[0], 4, 8, 8, 3)

    predict_fn = _build_eval_predict_fn(
        FakeProcessorModel(),
        is_processor_model=True,
        decode_fn=None,
        n_members=5,
    )

    preds, trues = predict_fn(batch)

    # preds should have ensemble dim: (B, T, H, W, C, M)
    assert preds.shape == (2, 4, 8, 8, 3, 5)
    # trues should remain unchanged
    assert trues.shape == (2, 4, 8, 8, 3)


def test_decode_tensor_preserves_ensemble_axis():
    x = torch.randn(2, 4, 8, 8, 3, 5)

    def decode_fn(tensor):
        return tensor[..., :2]

    decoded = _decode_tensor(x, decode_fn, n_members=5)

    assert decoded.shape == (2, 4, 8, 8, 2, 5)


def test_resolve_rollout_timestep_limit_multiplies_by_stride():
    assert (
        _resolve_rollout_timestep_limit(max_rollout_steps=50, rollout_stride=4) == 200
    )


def test_resolve_rollout_timestep_limit_returns_none_for_invalid_inputs():
    assert (
        _resolve_rollout_timestep_limit(max_rollout_steps=None, rollout_stride=4)
        is None
    )
    assert (
        _resolve_rollout_timestep_limit(max_rollout_steps=0, rollout_stride=4) is None
    )
    assert (
        _resolve_rollout_timestep_limit(max_rollout_steps=10, rollout_stride=0) is None
    )


def test_split_metric_and_metadata_rows_separates_meta_rows():
    rows = [
        {"window": "all", "batch_idx": "all", "mse": 0.1},
        {
            "window": "meta",
            "batch_idx": "all",
            "category": "runtime_eval",
            "metric": "total_s",
            "value": 1.2,
        },
        {"window": "0-1", "batch_idx": 0, "rmse": 0.2},
    ]

    metric_rows, metadata_rows = _split_metric_and_metadata_rows(rows)

    assert metric_rows == [
        {"window": "all", "batch_idx": "all", "mse": 0.1},
        {"window": "0-1", "batch_idx": 0, "rmse": 0.2},
    ]
    assert metadata_rows == [
        {
            "window": "meta",
            "batch_idx": "all",
            "category": "runtime_eval",
            "metric": "total_s",
            "value": 1.2,
        }
    ]


def test_split_metric_and_metadata_rows_skips_malformed_rows():
    rows = [
        {"window": "all", "batch_idx": "all", "mse": 0.1},
        "bad-row",
        {
            "window": "meta",
            "batch_idx": "all",
            "category": "runtime_eval",
            "metric": "total_s",
            "value": 1.2,
        },
    ]

    metric_rows, metadata_rows = _split_metric_and_metadata_rows(cast(Any, rows))

    assert metric_rows == [{"window": "all", "batch_idx": "all", "mse": 0.1}]
    assert metadata_rows == [
        {
            "window": "meta",
            "batch_idx": "all",
            "category": "runtime_eval",
            "metric": "total_s",
            "value": 1.2,
        }
    ]


def test_normalize_per_batch_rows_flattens_nested_mappings_only():
    gathered = [
        [
            {
                "window": "all",
                "batch_idx": torch.tensor(0, dtype=torch.int32),
                "mse": torch.tensor(0.1),
            }
        ],
        (
            [{"window": "0-1", "batch_idx": 1, "rmse": 0.2}],
            "ignore-me",
            None,
        ),
    ]

    normalized = _normalize_per_batch_rows(gathered)

    assert normalized == [
        {"window": "all", "batch_idx": 0, "mse": pytest.approx(0.1)},
        {"window": "0-1", "batch_idx": 1, "rmse": 0.2},
    ]


def test_reindex_per_batch_rows_by_rank_interleaves_global_batch_index():
    rows_by_rank = [
        [
            {"window": "all", "batch_idx": 0, "mse": 0.1},
            {"window": "0-1", "batch_idx": 0, "mse": 0.11},
            {"window": "all", "batch_idx": 1, "mse": 0.2},
            {"window": "0-1", "batch_idx": 1, "mse": 0.21},
        ],
        [
            {"window": "all", "batch_idx": 0, "mse": 0.3},
            {"window": "0-1", "batch_idx": 0, "mse": 0.31},
            {"window": "all", "batch_idx": 1, "mse": 0.4},
            {"window": "0-1", "batch_idx": 1, "mse": 0.41},
        ],
    ]

    reindexed = _reindex_per_batch_rows_by_rank(rows_by_rank)

    assert [row["batch_idx"] for row in reindexed] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [row["window"] for row in reindexed] == [
        "all",
        "0-1",
        "all",
        "0-1",
        "all",
        "0-1",
        "all",
        "0-1",
    ]


def test_reindex_per_batch_rows_by_rank_preserves_non_numeric_batch_idx_rows():
    rows_by_rank = [
        [{"window": "all", "batch_idx": 0, "mse": 0.1}],
        [{"window": "meta", "batch_idx": "all", "value": 123.0}],
    ]

    reindexed = _reindex_per_batch_rows_by_rank(rows_by_rank)

    assert reindexed[0] == {"window": "all", "batch_idx": 0, "mse": 0.1}
    assert reindexed[1] == {"window": "meta", "batch_idx": "all", "value": 123.0}


def test_build_per_timestep_metric_factory_sets_reduce_all_false_for_crps():
    metric = _build_per_timestep_metric_factory(CRPS)()
    assert getattr(metric, "reduce_all", None) is False


def test_build_per_timestep_metric_factory_sets_reduce_all_false_for_afcrps():
    metric = _build_per_timestep_metric_factory(AlphaFairCRPS)()
    assert getattr(metric, "reduce_all", None) is False


def test_build_per_timestep_metric_factory_sets_reduce_all_false_for_ssr():
    metric = _build_per_timestep_metric_factory(SpreadSkillRatio)()
    assert getattr(metric, "reduce_all", None) is False


def test_default_eval_metrics_include_spread_and_skill_for_lola_comparison():
    assert "spread" in DEFAULT_EVAL_METRICS
    assert "skill" in DEFAULT_EVAL_METRICS
    assert "ssr" in DEFAULT_EVAL_METRICS


def test_should_skip_metric_variogram_only():
    assert _should_skip_metric("variogram") is True
    assert _should_skip_metric("crps") is False
    assert _should_skip_metric("ssr") is False


def test_resolve_rollout_channel_names_from_norm_already_subset():
    dataset = SimpleNamespace(
        norm=SimpleNamespace(core_field_names=["p", "u"]),
        channel_idxs=(2, 0),
    )

    assert _resolve_rollout_channel_names(dataset) == ["p", "u"]


def test_resolve_rollout_channel_names_from_stats_applies_idxs():
    dataset = SimpleNamespace(
        norm=None,
        normalization_stats={"core_field_names": ["u", "v", "p"]},
        channel_idxs=(2, 0),
    )

    assert _resolve_rollout_channel_names(dataset) == ["p", "u"]


def test_resolve_rollout_channel_names_returns_none_without_norm_names():
    dataset = SimpleNamespace(
        norm=None,
        metadata=SimpleNamespace(field_names={0: ["velocity_x", "velocity_y"]}),
        channel_idxs=None,
    )

    assert _resolve_rollout_channel_names(dataset) is None


def test_resolve_rollout_channel_names_returns_none_on_invalid_output_indices():
    dataset = SimpleNamespace(
        norm=None,
        normalization_stats={"core_field_names": ["u", "v"]},
        channel_idxs=(0, 3),
    )

    assert _resolve_rollout_channel_names(dataset) is None


# --- _training_runtime_rows with actual epoch times ---


def _make_timed_payload(
    epoch: int,
    global_step: int,
    total_s: float,
    epoch_times: list[float],
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "callbacks": {
            "TrainingTimerCallback": {
                "training_runtime_total_s": total_s,
                "training_runtime_elapsed_s": total_s,
                "mean_epoch_s": sum(epoch_times) / len(epoch_times),
                "min_epoch_s": min(epoch_times),
                "max_epoch_s": max(epoch_times),
                "epoch_times_s": epoch_times,
            }
        },
    }


def test_training_runtime_rows_emit_min_mean_max_when_epoch_times_available():
    payload = _make_timed_payload(
        epoch=2, global_step=300, total_s=30.0, epoch_times=[8.0, 10.0, 12.0]
    )
    rows = _training_runtime_rows(payload)
    by_metric = {r["metric"]: r["value"] for r in rows}

    assert by_metric["total_s"] == pytest.approx(30.0)
    assert by_metric["mean_epoch_s"] == pytest.approx(10.0)
    assert by_metric["min_epoch_s"] == pytest.approx(8.0)
    assert by_metric["max_epoch_s"] == pytest.approx(12.0)


def test_training_runtime_rows_fall_back_to_elapsed_runtime_when_total_missing():
    payload = {
        "callbacks": {
            "TrainingTimerCallback": {
                "training_runtime_total_s": None,
                "training_runtime_elapsed_s": 12.5,
            }
        }
    }
    rows = _training_runtime_rows(payload)
    by_metric = {r["metric"]: r["value"] for r in rows}
    assert by_metric["total_s"] == pytest.approx(12.5)


def test_training_runtime_rows_fall_back_to_average_without_epoch_times():
    payload = {"epoch": 3, "global_step": 400, "training_runtime_total_s": 40.0}
    rows = _training_runtime_rows(payload)

    assert rows == []


def test_render_rollouts_resolves_indices_within_batched_samples(tmp_path, monkeypatch):
    class DummyModel:
        def rollout(self, *_args, **_kwargs):
            preds = torch.randn(4, 3, 2, 2, 1)
            trues = torch.randn(4, 3, 2, 2, 1)
            return preds, trues

    captured_paths: list[str] = []

    def _fake_plot_spatiotemporal_video(**kwargs):
        captured_paths.append(kwargs["save_path"])

    monkeypatch.setattr(
        "autocast.scripts.eval.encoder_processor_decoder.plot_spatiotemporal_video",
        _fake_plot_spatiotemporal_video,
    )

    out_paths = _render_rollouts(
        model=cast(Any, DummyModel()),
        dataloader=[object()],
        batch_indices=[0, 1, 2, 3],
        video_dir=tmp_path,
        sample_index=0,
        fmt="mp4",
        fps=5,
        stride=1,
        max_rollout_steps=2,
        free_running_only=True,
        n_members=None,
    )

    assert len(out_paths) == 4
    assert len(captured_paths) == 4
    for idx in range(4):
        assert any(f"batch_{idx}_sample_{idx}.mp4" in p for p in captured_paths)


# ---------------------------------------------------------------------------
# eval.mode + ambient/latent path resolution
# ---------------------------------------------------------------------------


def test_default_eval_mode_is_auto():
    assert DEFAULT_EVAL_MODE == "auto"


def test_normalize_eval_mode_accepts_known_values_and_none():
    assert _normalize_eval_mode(None) == "auto"
    assert _normalize_eval_mode("Auto") == "auto"
    assert _normalize_eval_mode("Encode_Once") == "encode_once"
    assert _normalize_eval_mode("Ambient") == "ambient"
    assert _normalize_eval_mode("LATENT") == "latent"


def test_normalize_eval_mode_rejects_unknown():
    with pytest.raises(ValueError, match=r"Unknown eval\.mode"):
        _normalize_eval_mode("something-else")


def _example_batch(kind):
    if kind == "batch":
        return Batch(
            input_fields=torch.zeros(1, 1, 2, 2, 1),
            output_fields=torch.zeros(1, 1, 2, 2, 1),
            constant_scalars=None,
            constant_fields=None,
        )
    return EncodedBatch(
        encoded_inputs=torch.zeros(1, 1, 2, 2, 1),
        encoded_output_fields=torch.zeros(1, 1, 2, 2, 1),
        global_cond=None,
        encoded_info={},
    )


@pytest.mark.parametrize(
    ("processor_only", "batch_type", "ae_ckpt", "expected"),
    [
        # Full EPD (or stateless AE baked into processor) -> ambient.
        (False, "batch", False, "ambient"),
        (False, "encoded", False, "ambient"),
        (False, "batch", True, "ambient"),
        # Processor-only on raw Batch with AE -> encode_once.
        (True, "batch", True, "encode_once"),
        # Processor-only on cached latents with AE -> encode_once (a swap will
        # happen downstream); without AE -> latent.
        (True, "encoded", True, "encode_once"),
        (True, "encoded", False, "latent"),
        # Processor-only on raw Batch without AE: falls back to latent.
        (True, "batch", False, "latent"),
    ],
)
def test_resolve_auto_eval_mode_picks_faithful_default(
    processor_only, batch_type, ae_ckpt, expected
):
    resolved = _resolve_auto_eval_mode(
        processor_only=processor_only,
        example_batch=_example_batch(batch_type),
        has_autoencoder_checkpoint=ae_ckpt,
    )

    assert resolved == expected


_RESOLVE_EVAL_PATH_CASES = [
    # Full EPD: always resolves to ambient_epd regardless of mode.
    ("ambient", False, "batch", False, False, EVAL_PATH_AMBIENT_EPD),
    ("encode_once", False, "batch", False, False, EVAL_PATH_AMBIENT_EPD),
    ("ambient", False, "encoded", False, False, EVAL_PATH_AMBIENT_EPD),
    # Processor-only + raw Batch + AE ckpt: mode selects the path.
    ("ambient", True, "batch", True, False, EVAL_PATH_AMBIENT_EPD),
    ("encode_once", True, "batch", True, False, EVAL_PATH_ENCODE_ONCE),
    # Processor-only + cached latents: latent paths, independent of mode.
    ("latent", True, "encoded", False, True, EVAL_PATH_LATENT_CACHED_WITH_DECODER),
    ("latent", True, "encoded", False, False, EVAL_PATH_LATENT_CACHED_LATENT_ONLY),
    # encode_once on cached latents still maps to the latent paths here;
    # _validate_resolved_eval_path is what rejects the combination.
    (
        "encode_once",
        True,
        "encoded",
        False,
        True,
        EVAL_PATH_LATENT_CACHED_WITH_DECODER,
    ),
    (
        "encode_once",
        True,
        "encoded",
        False,
        False,
        EVAL_PATH_LATENT_CACHED_LATENT_ONLY,
    ),
]


@pytest.mark.parametrize(
    (
        "eval_mode",
        "processor_only",
        "batch_type",
        "ae_ckpt",
        "decoder_loaded",
        "expected",
    ),
    _RESOLVE_EVAL_PATH_CASES,
)
def test_resolve_eval_path_matches_run_evaluation_branches(
    eval_mode, processor_only, batch_type, ae_ckpt, decoder_loaded, expected
):
    resolved = _resolve_eval_path(
        eval_mode=eval_mode,
        processor_only=processor_only,
        example_batch=_example_batch(batch_type),
        has_autoencoder_checkpoint=ae_ckpt,
        decode_fn_loaded=decoder_loaded,
    )

    assert resolved == expected


def test_validate_resolved_eval_path_ambient_rejects_latent_path():
    with pytest.raises(ValueError, match=r"eval\.mode=ambient"):
        _validate_resolved_eval_path(
            eval_mode="ambient",
            resolved_path=EVAL_PATH_LATENT_CACHED_WITH_DECODER,
        )


def test_validate_resolved_eval_path_latent_rejects_ambient_path():
    with pytest.raises(ValueError, match=r"eval\.mode=latent"):
        _validate_resolved_eval_path(
            eval_mode="latent",
            resolved_path=EVAL_PATH_AMBIENT_EPD,
        )


def test_validate_resolved_eval_path_latent_rejects_encode_once_path():
    with pytest.raises(ValueError, match=r"eval\.mode=latent"):
        _validate_resolved_eval_path(
            eval_mode="latent",
            resolved_path=EVAL_PATH_ENCODE_ONCE,
        )


def test_validate_resolved_eval_path_encode_once_rejects_latent_paths():
    for path in (
        EVAL_PATH_LATENT_CACHED_WITH_DECODER,
        EVAL_PATH_LATENT_CACHED_LATENT_ONLY,
    ):
        with pytest.raises(ValueError, match=r"eval\.mode=encode_once"):
            _validate_resolved_eval_path(eval_mode="encode_once", resolved_path=path)


def test_validate_resolved_eval_path_happy_paths():
    _validate_resolved_eval_path(
        eval_mode="ambient",
        resolved_path=EVAL_PATH_AMBIENT_EPD,
    )
    _validate_resolved_eval_path(
        eval_mode="encode_once",
        resolved_path=EVAL_PATH_AMBIENT_EPD,
    )
    _validate_resolved_eval_path(
        eval_mode="encode_once",
        resolved_path=EVAL_PATH_ENCODE_ONCE,
    )
    _validate_resolved_eval_path(
        eval_mode="latent",
        resolved_path=EVAL_PATH_LATENT_CACHED_WITH_DECODER,
    )
    _validate_resolved_eval_path(
        eval_mode="latent",
        resolved_path=EVAL_PATH_LATENT_CACHED_LATENT_ONLY,
    )


@pytest.mark.parametrize("mode", ["auto", "ambient", "encode_once"])
def test_validate_latent_space_metrics_flag_rejects_non_latent_modes(mode):
    with pytest.raises(ValueError, match=r"eval\.latent_space_metrics"):
        _validate_latent_space_metrics_flag(
            requested_eval_mode=mode,
            latent_space_metrics=True,
        )
    # flag=False is always a no-op, regardless of mode.
    _validate_latent_space_metrics_flag(
        requested_eval_mode=mode,
        latent_space_metrics=False,
    )


def test_validate_latent_space_metrics_flag_allows_explicit_latent():
    _validate_latent_space_metrics_flag(
        requested_eval_mode="latent",
        latent_space_metrics=True,
    )


def test_require_decoder_fails_fast_without_opt_in():
    with pytest.raises(RuntimeError, match=r"latent_space_metrics=true"):
        _require_decoder_unless_latent_metrics_opt_in(
            resolved_path=EVAL_PATH_LATENT_CACHED_LATENT_ONLY,
            latent_space_metrics=False,
        )


def test_require_decoder_allows_opt_in_and_warns(caplog):
    with caplog.at_level("WARNING"):
        _require_decoder_unless_latent_metrics_opt_in(
            resolved_path=EVAL_PATH_LATENT_CACHED_LATENT_ONLY,
            latent_space_metrics=True,
        )
    # Decoded paths are no-ops whether or not the flag is set.
    for path in (
        EVAL_PATH_AMBIENT_EPD,
        EVAL_PATH_ENCODE_ONCE,
        EVAL_PATH_LATENT_CACHED_WITH_DECODER,
    ):
        _require_decoder_unless_latent_metrics_opt_in(
            resolved_path=path, latent_space_metrics=False
        )
        _require_decoder_unless_latent_metrics_opt_in(
            resolved_path=path, latent_space_metrics=True
        )

    assert any(
        "latent_space_metrics=true" in record.message for record in caplog.records
    ), "expected a prominent warning about latent-only metrics"


def test_maybe_swap_to_ambient_datamodule_is_noop_for_raw_batch(tmp_path):
    cfg = OmegaConf.create(
        {"datamodule": {"_target_": "raw.DataModule", "data_path": str(tmp_path)}}
    )
    raw_batch = Batch(
        input_fields=torch.zeros(1, 1, 2, 2, 1),
        output_fields=torch.zeros(1, 1, 2, 2, 1),
        constant_scalars=None,
        constant_fields=None,
    )

    result = _maybe_swap_to_ambient_datamodule(
        cfg, eval_mode="ambient", example_batch=raw_batch
    )

    assert result is cfg
    assert cfg.datamodule._target_ == "raw.DataModule"


def test_maybe_swap_to_ambient_datamodule_is_noop_for_non_ambient(tmp_path):
    cfg = OmegaConf.create(
        {
            "datamodule": {
                "_target_": "cached.LatentDataModule",
                "data_path": str(tmp_path),
            }
        }
    )
    encoded = EncodedBatch(
        encoded_inputs=torch.zeros(1, 1, 2, 2, 1),
        encoded_output_fields=torch.zeros(1, 1, 2, 2, 1),
        global_cond=None,
        encoded_info={},
    )

    result = _maybe_swap_to_ambient_datamodule(
        cfg, eval_mode="latent", example_batch=encoded
    )

    assert result is cfg
    assert cfg.datamodule._target_ == "cached.LatentDataModule"


def test_maybe_swap_to_ambient_datamodule_swaps_for_encode_once(tmp_path):
    # encode_once also needs raw inputs (encoder runs once); same swap as ambient.
    ae_cfg_path = tmp_path / "autoencoder_config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "datamodule": {
                    "_target_": "raw.TheWellDataModule",
                    "data_path": "/path/to/raw",
                    "use_normalization": True,
                }
            }
        ),
        ae_cfg_path,
    )
    cfg = OmegaConf.create(
        {
            "datamodule": {
                "_target_": "cached.LatentDataModule",
                "data_path": str(tmp_path),
            }
        }
    )
    encoded = EncodedBatch(
        encoded_inputs=torch.zeros(1, 1, 2, 2, 1),
        encoded_output_fields=torch.zeros(1, 1, 2, 2, 1),
        global_cond=None,
        encoded_info={},
    )

    _maybe_swap_to_ambient_datamodule(
        cfg, eval_mode="encode_once", example_batch=encoded
    )

    assert cfg.datamodule._target_ == "raw.TheWellDataModule"
    assert cfg.datamodule.data_path == "/path/to/raw"


def test_maybe_swap_to_ambient_datamodule_loads_from_cache_dir(tmp_path):
    ae_cfg_path = tmp_path / "autoencoder_config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "datamodule": {
                    "_target_": "raw.TheWellDataModule",
                    "data_path": "/path/to/raw",
                    "use_normalization": True,
                }
            }
        ),
        ae_cfg_path,
    )
    cfg = OmegaConf.create(
        {
            "datamodule": {
                "_target_": "cached.LatentDataModule",
                "data_path": str(tmp_path),
            }
        }
    )
    encoded = EncodedBatch(
        encoded_inputs=torch.zeros(1, 1, 2, 2, 1),
        encoded_output_fields=torch.zeros(1, 1, 2, 2, 1),
        global_cond=None,
        encoded_info={},
    )

    result = _maybe_swap_to_ambient_datamodule(
        cfg, eval_mode="ambient", example_batch=encoded
    )

    assert result is cfg
    assert cfg.datamodule._target_ == "raw.TheWellDataModule"
    assert cfg.datamodule.data_path == "/path/to/raw"
    assert cfg.datamodule.use_normalization is True


def test_maybe_swap_to_ambient_datamodule_errors_without_ae_config(tmp_path):
    cfg = OmegaConf.create(
        {
            "datamodule": {
                "_target_": "cached.LatentDataModule",
                "data_path": str(tmp_path),
            }
        }
    )
    encoded = EncodedBatch(
        encoded_inputs=torch.zeros(1, 1, 2, 2, 1),
        encoded_output_fields=torch.zeros(1, 1, 2, 2, 1),
        global_cond=None,
        encoded_info={},
    )

    with pytest.raises(FileNotFoundError, match=r"autoencoder_config\.yaml"):
        _maybe_swap_to_ambient_datamodule(
            cfg, eval_mode="ambient", example_batch=encoded
        )


def test_maybe_swap_to_ambient_datamodule_errors_without_data_path():
    cfg = OmegaConf.create({"datamodule": {"_target_": "cached.LatentDataModule"}})
    encoded = EncodedBatch(
        encoded_inputs=torch.zeros(1, 1, 2, 2, 1),
        encoded_output_fields=torch.zeros(1, 1, 2, 2, 1),
        global_cond=None,
        encoded_info={},
    )

    with pytest.raises(ValueError, match="no data_path"):
        _maybe_swap_to_ambient_datamodule(
            cfg, eval_mode="ambient", example_batch=encoded
        )
