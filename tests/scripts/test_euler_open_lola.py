"""Regressions for the Euler cache and paired raw-frame protocol."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.testing import assert_close
from torch.utils.data import DataLoader, Dataset, Subset

from autocast.data.encoded_dataset import MiniWellDataModule
from autocast.external.lola.field_transform import (
    named_field_indices,
    restore_fields,
    standardize_fields,
)
from autocast.nn.spatiotemporal_vit import SpatiotemporalViTBackbone
from autocast.scripts.eval.encoder_processor_decoder import (
    _apply_test_window_subset,
    _maybe_swap_to_ambient_datamodule,
)
from autocast.scripts.setup import _build_processor
from autocast.types.batch import Batch, EncodedBatch
from slurm_scripts.comparison.the_well.euler_open import cache, workflow
from slurm_scripts.comparison.the_well.rayleigh_benard.paired_test_windows import (
    validate_selection,
)


def test_euler_lola_rollout_draw_is_split_by_boundary():
    cfg = workflow.experiment_config("production")
    selections = [
        workflow.lola_rollout_selection(cfg, source) for source in ("open", "periodic")
    ]
    assert [len(selection["indices"]) for selection in selections] == [25, 39]
    for selection in selections:
        assert selection["population_size"] == sum(selection["files"].values()) == 500
        assert selection["indices"] == sorted(set(selection["indices"]))
        assert all(0 <= index < 500 for index in selection["indices"])


def test_euler_stride_one_paired_windows_allow_the_last_four_raw_frames():
    cfg = workflow.experiment_config("production")
    spec = cfg.euler_open_lola
    selection = workflow.paired_selection(spec, data_stride=1)
    starts = [window["context_frame"] for window in selection["windows"]]

    assert selection.get("stride", 1) == 1
    assert len(starts) == 512
    assert max(starts) == 96
    assert spec.cache.window_stride == 4


def test_euler_window_selection_matches_well_and_miniwell_offsets(tmp_path):
    cfg = workflow.experiment_config("production")
    spec = cfg.euler_open_lola
    selection = workflow.paired_selection(spec)
    per_trajectory = 85
    per_file = 50 * per_trajectory
    well = SimpleNamespace(
        metadata=SimpleNamespace(
            dataset_name=spec.dataset, field_names={0: list(spec.fields)}
        ),
        files_paths=list(selection["files"]),
        n_trajectories_per_file=[50] * 10,
        n_steps_input=1,
        n_steps_output=4,
        min_dt_stride=4,
        max_dt_stride=4,
        n_steps_per_trajectory=[101] * 10,
        n_windows_per_trajectory=[per_trajectory] * 10,
        file_index_offsets=[-1, *(i * per_file for i in range(1, 11))],
        transform=None,
    )
    indices = validate_selection(SimpleNamespace(well_dataset=well), selection)
    assert len(indices) == len(set(indices)) == 512
    assert [index % per_trajectory for index in indices] == [
        row["context_frame"] for row in selection["windows"]
    ]
    assert cfg.datamodule.stride == 4
    assert spec.cache.frame_stride == 1
    assert cfg.model.processor.context_mask.rho == 0.66
    assert len(cache.planned_shards(spec)) == 30

    class WindowDataset(Dataset):
        well_dataset = well

        def __len__(self):
            return 10 * per_file

        def __getitem__(self, index):
            return index

    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(selection, sort_keys=False))
    loader = _apply_test_window_subset(DataLoader(WindowDataset()), str(path), tmp_path)
    assert isinstance(loader.dataset, Subset)
    assert list(loader.dataset.indices) == indices
    assert (
        yaml.safe_load((tmp_path / "test_window_subset.yaml").read_text()) == selection
    )


def test_legacy_euler_evaluator_rejects_joint_checkpoints(
    tmp_path, monkeypatch, capsys
):
    cfg = workflow.experiment_config("production", joint=True)
    config = tmp_path / "joint.yaml"
    OmegaConf.save(cfg, config)
    manifest = tmp_path / "paired.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflow",
            "evaluate",
            str(config),
            "processor.ckpt",
            str(tmp_path),
            str(manifest),
        ],
    )
    with pytest.raises(SystemExit) as error:
        workflow.main()
    assert error.value.code == 2
    assert "maintained evaluator" in capsys.readouterr().err
    assert not manifest.exists()


@pytest.mark.parametrize("joint", [False, True])
def test_euler_capped_validation_covers_all_gamma_shards(tmp_path, joint):
    cfg = workflow.experiment_config("production", joint=joint)
    sources = ("open", "periodic") if joint else ("open",)
    roots = [str(tmp_path / source) for source in sources]
    paths = {"data_paths": roots} if joint else {"data_path": roots[0]}
    for source, root in zip(sources, roots, strict=True):
        spec = workflow.source_spec(cfg, source)
        for split in ("train", "valid", "test"):
            split_dir = Path(root) / split
            split_dir.mkdir(parents=True)
            for name in cache.shard_names(spec):
                with h5py.File(split_dir / name, "w") as handle:
                    handle.create_dataset("state", shape=(50, 101, 1, 1, 1), dtype="f4")

    dm = instantiate(cfg.datamodule, num_workers=0, **paths)
    dataset = dm.val_dataloader().dataset
    assert isinstance(dataset, Subset)
    expected_shards = set(range(10 * len(sources)))
    per_shard = 50 * 85
    for batch_size in (1, 8):  # CRPS and FM on 32 ranks.
        window_count = 32 * batch_size * cfg.trainer.limit_val_batches
        windows = dataset.indices[:window_count]
        assert len(set(windows)) == window_count
        assert {index // per_shard for index in windows} == expected_shards

    other = instantiate(cfg.datamodule, num_workers=0, **paths)
    other_dataset = other.val_dataloader().dataset
    assert isinstance(other_dataset, Subset)
    assert other_dataset.indices == dataset.indices

    if joint:
        missing = instantiate(
            cfg.datamodule,
            data_paths=[roots[0], str(tmp_path / "missing")],
            num_workers=0,
        )
        with pytest.raises(FileNotFoundError, match="No MiniWell shards"):
            missing.setup("fit")


def test_euler_evaluator_resolves_open_data_and_codec_transform(tmp_path):
    cfg = workflow.experiment_config("production")
    spec = cfg.euler_open_lola
    codec = tmp_path / spec.dataset / spec.codec.name
    cache_path = codec / "cache" / spec.dataset
    cache_path.mkdir(parents=True)
    (codec / "state.pth").touch()
    OmegaConf.save(
        {
            "ae": {"lat_channels": 64},
            "dataset": {
                "name": "euler_all",
                "physics": list(spec.codec.training_datasets),
                "stats": OmegaConf.to_container(spec.codec.stats),
                "transform": dict.fromkeys(spec.log1p_indices, "log1p"),
            },
        },
        codec / "config.yaml",
    )
    cfg.datamodule.data_path = str(cache_path)
    cfg.eval.lola_decoder_noisy = False
    batch = EncodedBatch(torch.empty(0), torch.empty(0), None, {})
    result = _maybe_swap_to_ambient_datamodule(
        cfg, eval_mode="encode_once", example_batch=batch
    )
    assert result.datamodule.well_base_path == str(tmp_path)
    assert result.datamodule.well_dataset_name == spec.dataset
    assert result.datamodule.min_dt_stride == result.datamodule.max_dt_stride == 4
    assert not result.datamodule.use_normalization
    assert list(result.model.encoder.log1p_indices) == [0, 1, 2]
    assert list(result.model.decoder.log1p_indices) == [0, 1, 2]
    assert not result.model.decoder.noisy


def test_euler_log_fields_round_trip_in_named_codec_order():
    fields = ["energy", "density", "pressure", "momentum_x", "momentum_y"]
    order = named_field_indices(
        [["density", "pressure", "energy"], ["momentum_y", "momentum_x"]], fields
    )
    assert order == [2, 0, 1, 4, 3]
    raw = torch.tensor([1.0, 3.0, 4.0, -2.0, 0.5])[order]
    mean = torch.tensor([1.2, 0.6, 0.58, 0.0, 0.0])
    std = torch.tensor([0.51, 0.27, 0.29, 0.53, 0.52])
    normalized = standardize_fields(raw, mean, std, [0, 1, 2])
    expected = (torch.cat((raw[:3].log1p(), raw[3:])) - mean) / std
    assert_close(normalized, expected)
    assert_close(restore_fields(normalized, mean, std, [0, 1, 2]), raw)


def test_euler_ambient_applies_the_same_transform_without_compression(REPO_ROOT):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                "local_experiment=the_well/euler_open/crps_spatiotemporal_ambient_matched"
            ],
        )
    assert cfg.datamodule.min_dt_stride == cfg.datamodule.max_dt_stride == 4
    assert cfg.eval.mode == "ambient"
    encoder = instantiate(cfg.model.encoder, in_channels=5)
    decoder = instantiate(cfg.model.decoder, in_channels=5)
    raw = torch.tensor([2.0, 1.0, 0.25, -2.0, 4.0]).expand(1, 1, 8, 8, 5)
    batch = Batch(
        input_fields=raw,
        output_fields=raw,
        constant_scalars=torch.tensor([[1.13]]),
        constant_fields=None,
        boundary_conditions=torch.ones(1, 2, 2),
    )
    encoded, label = encoder.encode_with_cond(batch)
    assert encoded.shape == raw.shape
    assert_close(encoded, standardize_fields(raw, encoder.mean, encoder.std, [0, 1, 2]))
    assert_close(decoder.decode(encoded), raw)
    assert_close(label, torch.tensor([[1.13, 1, 1, 1, 1]]))
    assert sum(p.numel() for p in encoder.parameters()) == 0
    assert sum(p.numel() for p in decoder.parameters()) == 0


@pytest.mark.parametrize(
    ("suffix", "batch_size", "shards"), [("", 64, 30), ("_smoke", 1, 3)]
)
def test_euler_flow_matching_uses_the_shared_cache_and_rb_sampler(
    REPO_ROOT, suffix, batch_size, shards
):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="processor",
            overrides=[
                f"local_experiment=the_well/euler_open/fm_spatiotemporal_latent{suffix}"
            ],
        )
    crps = workflow.experiment_config("smoke" if suffix else "production")
    assert cfg.datamodule.data_path == crps.datamodule.data_path
    assert cfg.datamodule.stride == crps.datamodule.stride == 4
    assert cfg.datamodule.batch_size == batch_size
    assert len(cache.planned_shards(cfg.euler_open_lola)) == shards
    assert cfg.model.get("n_members", 1) == 1
    assert cfg.model.processor.context_mask == crps.model.processor.context_mask
    assert not cfg.model.processor.loss_on_observed
    assert cfg.eval.n_members == 16
    with torch.device("meta"):
        processor = _build_processor(
            cfg.model,
            {
                "in_channels": 64,
                "out_channels": 64,
                "n_channels_out": 64,
                "n_steps_input": 1,
                "n_steps_output": 4,
                "spatial_resolution": (16, 16),
            },
            global_cond_channels=5,
        )
    assert processor.integrator == "ab3"
    assert processor.flow_ode_steps == 16
    assert sum(p.numel() for p in processor.parameters()) == 222_888_256


@pytest.mark.parametrize(
    ("dataset", "family", "channels", "labels", "grid", "parameters"),
    [
        ("rayleigh_benard", "latent", 64, 6, (16, 4), 223_019_328),
        ("rayleigh_benard", "ambient_matched", 4, 6, (512, 128), 225_969_408),
        ("euler_open", "latent", 64, 5, (16, 16), 223_019_072),
        ("euler_open", "ambient_matched", 5, 5, (512, 512), 226_755_840),
    ],
)
def test_rb_and_euler_comparison_parameter_counts(
    REPO_ROOT, dataset, family, channels, labels, grid, parameters
):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name=(
                "processor" if family == "latent" else "encoder_processor_decoder"
            ),
            overrides=[
                f"local_experiment=the_well/{dataset}/crps_spatiotemporal_{family}"
            ],
        )
    with torch.device("meta"):
        processor = _build_processor(
            cfg.model,
            {
                "in_channels": channels,
                "out_channels": channels,
                "n_channels_out": channels,
                "n_steps_input": 1,
                "n_steps_output": 4,
                "spatial_resolution": grid,
            },
            global_cond_channels=labels,
        )
    assert sum(p.numel() for p in processor.parameters()) == parameters
    backbone = processor.backbone
    assert isinstance(backbone, SpatiotemporalViTBackbone)
    assert backbone.n_steps_input == backbone.n_steps_output == 5


@pytest.mark.parametrize(
    ("source_name", "boundary_code"), [("open", 1), ("periodic", 2)]
)
def test_interrupted_shard_is_invisible_and_complete_shard_is_reused(
    tmp_path, monkeypatch, source_name, boundary_code
):
    spec = workflow.source_spec(workflow.experiment_config("smoke"), source_name)
    spec.autocast_revision = "4" * 40
    spec.well_base_path = str(tmp_path / "datasets")
    spec.cache.directory = str(tmp_path / "cache")
    spec.physical_shape = [2, 2, 5]
    spec.latent_shape = [5, 2, 2]
    raw = torch.arange(spec.frames).float().view(-1, 1, 1, 1).expand(-1, 2, 2, 5)
    sample = {
        "input_fields": raw,
        "constant_scalars": torch.tensor([1.13]),
        "boundary_conditions": torch.full((2, 2), boundary_code, dtype=torch.float32),
    }
    spec.cache.encode_chunk_size = 16
    for split, name in cache.planned_shards(spec):
        source = Path(spec.well_base_path) / spec.dataset / "data" / split / name
        source.parent.mkdir(parents=True)
        source.touch()

    class IdentityCodec(nn.Module):
        failure = True
        calls = 0

        def encode(self, x):
            self.calls += 1
            if self.failure and self.calls == 2:
                msg = "interrupted encode"
                raise RuntimeError(msg)
            return x

    class FakeWell:
        metadata = SimpleNamespace(field_names={0: list(spec.fields)})

        def __getitem__(self, _):
            return sample

    codec = IdentityCodec()
    torch.save(codec.state_dict(), tmp_path / "state.pth")
    monkeypatch.setattr(cache, "_load_well_shard", lambda *_: FakeWell())
    monkeypatch.setattr(
        cache, "load_codec_config", lambda _: (OmegaConf.create({"ae": {}}), tmp_path)
    )
    monkeypatch.setattr(cache, "get_autoencoder", lambda **_: codec)

    with pytest.raises(RuntimeError, match="interrupted encode"):
        cache.cache_shard(spec, 0, torch.device("cpu"))
    assert not list((Path(spec.cache.directory) / "train").iterdir())
    codec.failure = False
    for index, _ in enumerate(cache.planned_shards(spec)):
        cache.cache_shard(spec, index, torch.device("cpu"))
    calls = codec.calls
    cache.cache_shard(spec, 0, torch.device("cpu"))
    assert codec.calls == calls
    cache.verify_cache(spec)
    spec.autocast_revision = "another-reader-revision"
    with pytest.raises(ValueError, match="Cache provenance differs"):
        cache.verify_cache(spec)
    cache.verify_cache(spec, accept_recorded_revision=True)
    _, name = cache.planned_shards(spec)[0]
    with h5py.File(Path(spec.cache.directory) / "train" / name) as handle:
        state = handle["state"]
        assert isinstance(state, h5py.Dataset)
        assert state.shape == (1, 101, 5, 2, 2)
        provenance = json.loads(str(handle.attrs["provenance"]))
        assert provenance["source_trajectory_range"] == [0, 1]
        assert provenance["source_shard"].endswith(f"train/{name}")
    data = MiniWellDataModule(
        data_path=spec.cache.directory, n_steps_input=1, n_steps_output=4, stride=4
    )
    data.setup("test")
    assert data.test_dataset is not None
    window = data.test_dataset[0]
    assert_close(window.global_cond, torch.tensor([1.13, *([boundary_code] * 4)]))
    physical = restore_fields(
        window.encoded_output_fields,
        torch.tensor(spec.codec.stats.mean),
        torch.tensor(spec.codec.stats.std),
        spec.log1p_indices,
    )
    assert_close(physical, raw[[4, 8, 12, 16]])
    (Path(spec.cache.directory) / "test" / "unexpected.h5").touch()
    with pytest.raises(ValueError, match="unexpected shards"):
        cache.verify_cache(spec)
