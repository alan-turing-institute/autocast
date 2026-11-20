"""Initialize configured references using the parent model's coordinate context."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import torch
from omegaconf import DictConfig, OmegaConf
from the_well.data.normalization import ZScoreNormalization

from autocast.data.dataset import SpatioTemporalDataset
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.processor import ProcessorModel
from autocast.processors.residual_reference import ProcessorReference
from autocast.scripts.execution import extract_state_dict

ModelT = TypeVar("ModelT", EncoderProcessorDecoder, ProcessorModel)


def _local_file(value: str) -> Path:
    if "://" in value:
        msg = "Reference checkpoint/config paths must be local files."
        raise ValueError(msg)
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Reference file not found: {path}")
    return path


def _validate_normalization(data: DictConfig, dataset: Any) -> None:
    # Path equality does not establish the statistics used when training. Only
    # inline statistics in the saved config preserve those values and field order.
    if type(dataset) is not SpatioTemporalDataset:
        msg = "Reference normalization validation requires SpatioTemporalDataset."
        raise ValueError(msg)
    enabled = data.get("use_normalization")
    if enabled is False and dataset.norm is None and not dataset.use_normalization:
        return
    stats = data.get("normalization_stats")
    if (
        enabled is not True
        or not isinstance(stats, DictConfig)
        or type(dataset.norm) is not ZScoreNormalization
        or not dataset.use_normalization
        or data.get("normalization_type") is not None
    ):
        msg = (
            "Reference normalization requires explicit use_normalization=false or "
            "inline normalization_stats with the default ZScoreNormalization. "
            "A statistics path alone does not prove compatibility."
        )
        raise ValueError(msg)
    fields = list(stats["core_field_names"])
    if data.get("channel_idxs") is not None:
        fields = [fields[i] for i in data.channel_idxs]
    expected = ZScoreNormalization(
        OmegaConf.to_container(stats["stats"], resolve=True),  # type: ignore[arg-type]
        fields,
        list(stats.get("constant_field_names", [])),
    )
    actual = dataset.norm
    if (
        actual.core_field_names != expected.core_field_names
        or actual.core_constant_field_names != expected.core_constant_field_names
        or any(
            not torch.equal(
                getattr(actual, key)[mode].cpu(), getattr(expected, key)[mode]
            )
            for key in ("flattened_means", "flattened_stds")
            for mode in ("variable", "constant")
        )
    ):
        msg = "Reference normalization statistics or field order do not match."
        raise ValueError(msg)


def _validate_config(
    saved: DictConfig,
    current: DictConfig,
    processor_config: DictConfig,
    dataset: Any,
) -> None:
    source_model = saved.get("model")
    source_data = saved.get("datamodule")
    if not isinstance(source_model, DictConfig) or not isinstance(
        source_data, DictConfig
    ):
        msg = "Reference resolved config is missing model/datamodule metadata."
        raise ValueError(msg)
    if (
        source_data.get("_target_")
        != "autocast.data.datamodule.SpatioTemporalDataModule"
        or source_data != current.get("datamodule")
        or saved.get("simulator") != current.get("simulator")
    ):
        msg = (
            "Reference requires identical SpatioTemporalDataModule coordinate settings."
        )
        raise ValueError(msg)
    _validate_normalization(source_data, dataset)
    for key in ("encoder", "decoder"):
        if not isinstance(source_model.get(key), DictConfig) or (
            source_model[key] != current.model.get(key)
        ):
            raise ValueError(
                f"Reference {key} config does not match parent coordinates."
            )
    if source_model.get("processor") != processor_config:
        msg = "Reference processor config does not match the saved resolved config."
        raise ValueError(msg)
    if source_model.get("residual_prediction", False) or any(
        cfg.get("input_noise_injector") is not None
        for cfg in (source_model, current.model)
    ):
        msg = "Reference checkpoints require full-state outputs and no input noise."
        raise ValueError(msg)


def _validate_codec(
    model: EncoderProcessorDecoder, saved: DictConfig, state: Mapping[str, torch.Tensor]
) -> None:
    codec = model.encoder_decoder.state_dict()
    saved_codec = {
        key.removeprefix("encoder_decoder."): value
        for key, value in state.items()
        if key.startswith("encoder_decoder.")
    }
    if codec.keys() != saved_codec.keys() or any(
        not isinstance(saved_codec[key], torch.Tensor)
        or value.shape != saved_codec[key].shape
        or value.dtype != saved_codec[key].dtype
        or not torch.equal(value.cpu(), saved_codec[key])
        for key, value in codec.items()
    ):
        msg = "Reference checkpoint codec state does not match parent coordinates."
        raise ValueError(msg)
    if codec and not (
        (model.train_in_latent_space or model.freeze_encoder_decoder)
        and (
            saved.model.get("train_in_latent_space", False)
            or saved.model.get("freeze_encoder_decoder", False)
            or saved.datamodule.get("freeze_autoencoder", False)
        )
    ):
        msg = "Reference checkpoints with learned coordinates require frozen codecs."
        raise ValueError(msg)


def load_processor_references(
    model: ModelT,
    config: DictConfig,
    dataset: Any,
) -> ModelT:
    """Initialize reference weights once, before any parent checkpoint restoration.

    Supports paired EPD state_dict checkpoints and resolved configs with matching
    data settings, fixed codecs and inline (or disabled) normalization. Cached
    latent and metadata-free checkpoints cannot establish coordinate provenance.
    Only the exact ``processor.`` state prefix is loaded, strictly. Parent codec
    and normalizer objects are checked but never replaced or loaded here.
    """
    # Initialize children first so an enclosing reference checkpoint, like a
    # subsequently restored parent checkpoint, remains authoritative.
    for name, reference in reversed(list(model.processor.named_modules())):
        if not isinstance(reference, ProcessorReference) or reference.checkpoint_loaded:
            continue
        if not isinstance(model, EncoderProcessorDecoder):
            msg = (
                "Configured reference checkpoints require EPD setup; cached latent "
                "metadata does not establish encoder/normalization provenance."
            )
            raise ValueError(msg)
        assert reference.config_path is not None
        assert reference.checkpoint_path is not None
        saved = OmegaConf.load(_local_file(reference.config_path))
        if not isinstance(saved, DictConfig):
            msg = "Reference resolved config must be a mapping."
            raise ValueError(msg)
        nested_config = OmegaConf.select(config.model.processor, f"{name}.processor")
        _validate_config(saved, config, nested_config, dataset)
        try:
            payload = torch.load(
                _local_file(reference.checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            msg = "Could not load reference checkpoint safely as a CPU state_dict."
            raise ValueError(msg) from exc
        if not isinstance(payload, Mapping) or "state_dict" not in payload:
            msg = "Reference checkpoint must contain an EPD state_dict."
            raise ValueError(msg)
        state = extract_state_dict(payload)
        if not all(isinstance(key, str) for key in state):
            msg = "Reference checkpoint state_dict keys must be strings."
            raise ValueError(msg)
        _validate_codec(model, saved, state)
        weights = {
            key.removeprefix("processor."): value
            for key, value in state.items()
            if key.startswith("processor.")
        }
        expected = reference.processor.state_dict()
        if weights.keys() != expected.keys() or any(
            not isinstance(weights[key], torch.Tensor)
            or value.shape != weights[key].shape
            or value.dtype != weights[key].dtype
            for key, value in expected.items()
        ):
            msg = (
                "Reference checkpoint processor state_dict keys/shapes/dtypes "
                "do not match."
            )
            raise ValueError(msg)
        reference.processor.load_state_dict(weights, strict=True)
        reference.checkpoint_loaded = True
    return model
