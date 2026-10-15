"""Pool/config and tiny CPU-loop tests; no production fits or jobs."""

import copy
import importlib
import json
import math
import sys
from itertools import islice
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from autocast.data.dataset import SpatioTemporalDataset

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/swe64s"))
    return importlib.import_module("prepare_ae_pretraining")


def _write_payload(path, count, offset):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = torch.arange(count * 9 * 2 * 2 * 3, dtype=torch.float32)
    fields = fields.reshape(count, 9, 2, 2, 3) / 1000 + offset
    torch.save(
        {
            "data": fields,
            "constant_scalars": torch.full((count, 1), 0.1),
            "constant_fields": None,
        },
        path,
    )


@pytest.fixture
def inputs(tmp_path, pool):
    baseline, source = tmp_path / "baseline", tmp_path / "source"
    baseline.mkdir()
    source.mkdir()
    simulator = {
        "nx": 2,
        "ny": 2,
        "T": 2.0,
        "dt_save": 0.25,
        "skip_nt": 0,
        "parameters_range": {"amp": [0.1, 0.1]},
    }
    baseline_config = {
        "simulator": simulator,
        "seed": 0,
        "dataset": {"n_train": 2, "n_valid": 1, "n_test": 1},
    }
    source_config = {
        "simulator": simulator,
        "seed": pool.POOL_SEED,
        "dataset": {"n_train": 1000, "n_valid": 0, "n_test": 0},
    }
    for path, cfg in ((baseline, baseline_config), (source, source_config)):
        OmegaConf.save(OmegaConf.create(cfg), path / "resolved_config.yaml")
    for split, count, offset in (("train", 2, 1), ("valid", 1, 3), ("test", 1, 4)):
        _write_payload(baseline / split / "data.pt", count, offset)
    _write_payload(source / "train/data.pt", 1000, 10)
    stats = {
        "core_field_names": ["h", "u", "v"],
        "constant_field_names": [],
        "stats": {
            name: dict.fromkeys(("h", "u", "v"), value)
            for name, value in (
                ("mean", 0.0),
                ("std", 1.0),
                ("mean_delta", 0.0),
                ("std_delta", 1.0),
            )
        },
    }
    OmegaConf.save(OmegaConf.create(stats), baseline / "stats.yml")
    contract = {
        "simulator": simulator,
        "splits": {"train": 2, "valid": 1, "test": 1},
        "frames": 9,
        "resolution": [2, 2],
        "field_names": ["h", "u", "v"],
        "saved_times": {"start": 0.0, "stop": 2.0, "interval": 0.25},
        "sha256": {
            name: pool._sha256(baseline / name)
            for name in (
                "train/data.pt",
                "valid/data.pt",
                "test/data.pt",
                "stats.yml",
                "resolved_config.yaml",
            )
        },
    }
    return baseline, source, tmp_path / "view", contract


def test_generation_plan_is_read_only_and_worker_independent(tmp_path, pool):
    first = pool.generation_plan(tmp_path / "shards", tmp_path / "autosim", workers=1)
    second = pool.generation_plan(tmp_path / "shards", tmp_path / "autosim", workers=4)
    assert first["shards"] == second["shards"]
    assert sum(part["count"] for part in first["shards"]) == 1000
    assert not list(tmp_path.iterdir())
    assert not first["submitted"]
    for left, right in zip(first["shards"], first["shards"][1:], strict=False):
        assert left["reserved_seed_range"][1] < right["reserved_seed_range"][0]
    assert first["shards"][0]["seed"] > 20261014 + 3 * 4001 - 1
    assert len({part["output"] for part in first["shards"]}) == 10
    assert all("OMP_NUM_THREADS=1" in part["command"] for part in first["shards"])
    with pytest.raises(ValueError, match="Worker count"):
        pool.generation_plan(tmp_path, tmp_path, workers=0)


def test_pretraining_config_matches_fresh_ae_and_original_update_clock(pool):
    preflight = importlib.import_module("check_ae_diversity")
    cfg = preflight.check_config(fresh=True, pool_size=pool.POOL_COUNT)
    reference = preflight.check_config(fresh=True)
    assert cfg.trainer.devices == 4
    assert cfg.trainer.num_nodes == 1
    assert cfg.trainer.strategy == "ddp"
    for key in ("model", "optimizer"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(
            reference[key], resolve=True
        )
    expected_trainer = OmegaConf.to_container(reference.trainer, resolve=True)
    assert isinstance(expected_trainer, dict)
    expected_trainer.update(
        limit_train_batches=1004, max_steps=514048, max_time="00:23:30:00"
    )
    expected_trainer["callbacks"][0]["stop_step"] = 514048
    assert OmegaConf.to_container(cfg.trainer, resolve=True) == expected_trainer
    expected_data = OmegaConf.to_container(reference.datamodule, resolve=True)
    assert isinstance(expected_data, dict)
    expected_data["data_path"] = "/preflight/ae1000"
    expected_data["normalization_path"] = "/preflight/ae1000/stats.yml"
    assert OmegaConf.to_container(cfg.datamodule, resolve=True) == expected_data
    assert cfg.experiment_name == "swe64s_autoencoder_pretraining_1000"
    assert cfg.seed == 42
    assert cfg.trainer.limit_train_batches == 1004
    assert cfg.trainer.max_epochs == 512
    assert cfg.trainer.max_steps == 512 * 1004
    assert cfg.trainer.max_time == "00:23:30:00"
    assert cfg.trainer.val_check_interval == 1004
    assert cfg.trainer.check_val_every_n_epoch is None
    assert cfg.datamodule.start_frame == 0
    assert cfg.datamodule.stride == 1
    assert cfg.datamodule.autoencoder_mode
    assert not cfg.resume_from_checkpoint
    assert not cfg.resume_weights_only
    assert not cfg.reset_resume_time_budget
    callback = instantiate(cfg.trainer.callbacks[0])
    assert callback.expected_resume_step == 0
    assert callback.stop_step == cfg.trainer.max_steps
    assert callback.schedule == {
        "original_steps_per_epoch": 1004,
        "cosine_epochs": 512,
        "base_lr": 1e-5,
    }
    with pytest.raises(ValueError, match="fresh 1000"):
        preflight.check_config(pool_size=1000)
    with pytest.raises(ValueError, match="must not be supplied"):
        preflight.check_config(fresh=True, pool_size=1000, checkpoint=Path("old.ckpt"))


def test_preflight_reports_the_full_sampled_epoch_budget(pool, monkeypatch, capsys):
    preflight = importlib.import_module("check_ae_diversity")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_ae_diversity.py", "--fresh", "--pool-size", "1000", "--configs-only"],
    )
    preflight.main()
    record = json.loads(capsys.readouterr().out)
    assert record["resume_step"] == 0
    assert record["stop_step"] == record["additional_updates"] == 514048
    assert record["pool_size"] == pool.POOL_COUNT
    assert record["config"]["trainer"]["limit_train_batches"] == 1004
    assert record["config"]["trainer"]["max_time"] == "00:23:30:00"
    assert not record["submitted"]


def test_sampled_ddp_epochs_are_disjoint_reproducible_and_reshuffled(pool):
    preflight = importlib.import_module("check_ae_diversity")
    cfg = preflight.check_config(fresh=True, pool_size=pool.POOL_COUNT)
    states = TensorDataset(torch.arange(pool.POOL_COUNT * 321))
    per_rank = cfg.trainer.limit_train_batches * cfg.datamodule.batch_size
    selections = []
    for epoch in (0, 1, 0):
        ranks = []
        for rank in range(4):
            sampler = DistributedSampler(
                states, num_replicas=4, rank=rank, shuffle=True, seed=cfg.seed
            )
            sampler.set_epoch(epoch)
            selected = list(islice(sampler, per_rank))
            assert len(selected) == len(set(selected)) == per_rank
            ranks.append(set(selected))
        combined = set.union(*ranks)
        assert sum(map(len, ranks)) == len(combined) == 64256
        assert len(combined) / len(states) == pytest.approx(0.2, abs=0.0002)
        assert {index // 321 for index in combined} == set(range(1000))
        selections.append(combined)
    assert selections[0] != selections[1]
    assert selections[0] == selections[2]


class _SampledAE(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.seen = {}
        self.validation_steps = []
        self.rates = []

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        self.seen.setdefault(self.current_epoch, []).extend(batch[0].tolist())
        return (self.weight - batch[0] / 200).square().mean()

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self.validation_steps.append(self.global_step)
        self.log("val_loss", self.weight.square())

    def on_before_optimizer_step(self, optimizer):
        self.rates.append(optimizer.param_groups[0]["lr"])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.SGD(self.parameters(), lr=1e-5)


@pytest.mark.usefixtures("pool")
def test_lightning_truncates_and_reshuffles_epochs_without_fixed_subset(tmp_path):
    ae = importlib.import_module("ae_diversity")
    dataset = TensorDataset(torch.arange(200, dtype=torch.float32))
    sampler = DistributedSampler(dataset, num_replicas=4, rank=0, shuffle=True, seed=42)
    train = DataLoader(dataset, batch_size=2, sampler=sampler)
    # Simulate one DDP rank on CPU; the test above checks all four rank sets.
    callback = ae.HistoricalAELearningRate(
        original_steps_per_epoch=5,
        cosine_epochs=3,
        expected_resume_step=0,
        stop_step=15,
        expected_world_size=1,
        expected_batch_size=2,
    )
    model = _SampledAE()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        max_steps=15,
        limit_train_batches=5,
        val_check_interval=5,
        check_val_every_n_epoch=None,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=DataLoader(dataset))
    assert trainer.global_step == 15
    assert model.validation_steps == [5, 10, 15]
    assert len(model.seen) == 3
    for epoch, seen in model.seen.items():
        sampler.set_epoch(epoch)
        assert seen == list(islice(sampler, 10))
    assert model.seen[0] != model.seen[1] != model.seen[2]
    assert model.rates == [
        ae.historical_lr(step, original_steps_per_epoch=5, cosine_epochs=3)
        for step in range(15)
    ]
    assert trainer.optimizers[0].param_groups[0]["lr"] == 0
    assert ae.historical_lr(514048) == 0
    assert ae.historical_lr(514047) > 0
    assert ae.historical_lr(1004 * 256) == pytest.approx(5e-6)
    assert math.isfinite(model.weight.item())


def test_full_pool_reuses_training_data_without_thinning_or_copying(inputs, pool):
    baseline, source, output, contract = inputs
    checked = pool._check_sources(baseline, [source], output, contract)
    assert not output.exists()
    record = pool._assemble(checked, contract)
    assert (output / "train").is_symlink()
    assert (output / "train").resolve() == source / "train"
    assert record["shape"] == [1000, 9, 2, 2, 3]
    assert record["all_frames_retained"]
    assert (output / "valid").resolve() == baseline / "valid"
    assert (output / "test").resolve() == baseline / "test"
    assert (output / "stats.yml").resolve() == baseline / "stats.yml"
    assert pool.validate_pretraining_data(output, contract)["shape"] == record["shape"]
    dataset = SpatioTemporalDataset(
        data_path=str(output / "train/data.pt"),
        autoencoder_mode=True,
        use_normalization=True,
        normalization_path=str(output / "stats.yml"),
    )
    assert len(dataset) == 1000 * 9
    for index in (0, 8, 9, len(dataset) - 1):
        assert torch.equal(dataset[index].input_fields, dataset[index].output_fields)
    for name, expected in contract["sha256"].items():
        assert pool._sha256(baseline / name) == expected


def test_shards_merge_in_seed_order_and_preserve_every_frame(inputs, pool):
    baseline, source, output, contract = inputs
    cfg = OmegaConf.load(source / "resolved_config.yaml")
    shards = []
    for index in range(10):
        shard = source.parent / f"shard-{index:03d}"
        shard.mkdir()
        shard_cfg = copy.deepcopy(cfg)
        shard_cfg.seed = pool.POOL_SEED + index * pool.SHARD_SEED_STRIDE
        shard_cfg.dataset.n_train = 100
        OmegaConf.save(shard_cfg, shard / "resolved_config.yaml")
        _write_payload(shard / "train/data.pt", 100, 10 + index * 100)
        shards.append(shard)
    checked = pool._check_sources(baseline, list(reversed(shards)), output, contract)
    record = pool._assemble(checked, contract)
    merged = pool._load_payload(output / "train/data.pt")
    assert record["shape"] == [1000, 9, 2, 2, 3]
    for index, shard in enumerate(shards):
        original = pool._load_payload(shard / "train/data.pt")
        assert torch.equal(
            merged["data"][index * 100 : (index + 1) * 100], original["data"]
        )
    assert not (output / "train/.field-buffer.bin").exists()
    pool.validate_pretraining_data(output, contract)


@pytest.mark.parametrize(
    "change", ["physics", "seed", "count", "duplicate", "nonfinite", "amp", "shape"]
)
def test_changed_or_invalid_source_is_rejected_before_writing(inputs, pool, change):
    baseline, source, output, contract = inputs
    cfg = OmegaConf.load(source / "resolved_config.yaml")
    payload = copy.deepcopy(pool._load_payload(source / "train/data.pt"))
    if change == "physics":
        cfg.simulator.T = 3
    elif change == "seed":
        cfg.seed = 20261014
    elif change == "count":
        cfg.dataset.n_valid = 1
    elif change == "duplicate":
        payload["data"][0] = pool._load_payload(baseline / "valid/data.pt")["data"][0]
    elif change == "nonfinite":
        payload["data"][0, 0, 0, 0, 0] = float("nan")
    elif change == "amp":
        payload["constant_scalars"][0] = 0.2
    elif change == "shape":
        payload["data"] = payload["data"][:, :-1]
    OmegaConf.save(cfg, source / "resolved_config.yaml")
    torch.save(payload, source / "train/data.pt")
    message = {
        "physics": "simulator settings",
        "seed": "seed or shard",
        "count": "split counts",
        "duplicate": "Duplicate retained",
        "nonfinite": "Nonfinite",
        "amp": "Amp must remain",
        "shape": "data shape",
    }[change]
    with pytest.raises(ValueError, match=message):
        pool._check_sources(baseline, [source], output, contract)
    assert not output.exists()


def test_existing_or_nested_destinations_are_rejected(inputs, pool):
    baseline, source, output, contract = inputs
    output.symlink_to(output.parent / "absent")
    with pytest.raises(FileExistsError):
        pool._check_sources(baseline, [source], output, contract)
    with pytest.raises(ValueError, match="disjoint"):
        pool._check_sources(baseline, [source], source / "new", contract)


@pytest.mark.parametrize("change", ["frames", "seed", "training", "normalization"])
def test_verification_rejects_changed_manifest_or_training_data(inputs, pool, change):
    baseline, source, output, contract = inputs
    pool._assemble(pool._check_sources(baseline, [source], output, contract), contract)
    path = output / pool.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    if change == "frames":
        manifest["all_frames_retained"] = False
    elif change == "seed":
        manifest["sources"][0]["seed"] += 1
    elif change == "training":
        train = source / "train/data.pt"
        train.write_bytes(train.read_bytes() + b"changed")
    else:
        (output / "stats.yml").unlink()
        (output / "stats.yml").write_text((baseline / "stats.yml").read_text())
    path.write_text(json.dumps(manifest))
    with pytest.raises(
        ValueError, match=r"manifest|provenance|hash mismatch|unchanged"
    ):
        pool.validate_pretraining_data(output, contract)
