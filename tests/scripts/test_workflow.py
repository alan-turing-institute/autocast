"""Tests for the workflow CLI package."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

from autocast.scripts.workflow import cli as workflow_cli
from autocast.scripts.workflow.cli import build_parser
from autocast.scripts.workflow.commands import (
    build_effective_eval_overrides,
    build_train_overrides,
    eval_command,
    infer_dataset_from_workdir,
    infer_hydra_config_from_workdir,
    infer_resume_checkpoint,
)
from autocast.scripts.workflow.helpers import run_module_command
from autocast.scripts.workflow.naming import (
    auto_run_name,
    dataset_name_token,
    sanitize_name_part,
)
from autocast.scripts.workflow.overrides import (
    contains_override,
    expand_sweep_overrides,
    extract_override_value,
    hydra_string_list_literal,
    normalized_override,
    set_override,
    split_top_level_csv,
    strip_hydra_sweep_controls,
)
from autocast.scripts.workflow.slurm import _parse_override_scalar, _should_use_srun


@pytest.fixture
def parser() -> argparse.ArgumentParser:
    return build_parser()


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------


def test_normalized_override_plain():
    assert normalized_override("key=val") == "key=val"


def test_normalized_override_plus_prefix():
    assert normalized_override("+key=val") == "key=val"


def test_extract_override_value_found():
    assert extract_override_value(["a=1", "b=2"], "b") == "2"


def test_extract_override_value_not_found():
    assert extract_override_value(["a=1"], "z") is None


def test_extract_override_value_last_wins():
    assert extract_override_value(["k=1", "k=2"], "k") == "2"


def test_extract_override_value_plus_prefix():
    assert extract_override_value(["+k=42"], "k") == "42"


def test_contains_override_present():
    assert contains_override(["a=1", "b.c=2"], "b.c=")


def test_contains_override_absent():
    assert not contains_override(["a=1"], "b=")


def test_set_override_new_key():
    result = set_override(["a=1"], "b", "2")
    assert result == ["a=1", "b=2"]


def test_set_override_replace():
    result = set_override(["a=1", "b=old"], "b", "new")
    assert result == ["a=1", "b=new"]


def test_strip_hydra_sweep_controls_removes_mode_and_sweep():
    overrides = ["key=val", "hydra.mode=MULTIRUN", "hydra.sweep.dir=/tmp"]
    assert strip_hydra_sweep_controls(overrides) == ["key=val"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a,b,c", ["a", "b", "c"]),
        ("[1,2],3", ["[1,2]", "3"]),
        ("fn(a,b),c", ["fn(a,b)", "c"]),
        ('"a,b",c', ['"a,b"', "c"]),
        ("solo", ["solo"]),
        ("{a:1,b:2},c", ["{a:1,b:2}", "c"]),
    ],
)
def test_split_top_level_csv(raw: str, expected: list[str]):
    assert split_top_level_csv(raw) == expected


def test_expand_sweep_overrides_no_sweep():
    result = expand_sweep_overrides(["a=1", "b=2"])
    assert result == [["a=1", "b=2"]]


def test_expand_sweep_overrides_single_sweep():
    result = expand_sweep_overrides(["a=1,2", "b=3"])
    assert result == [["a=1", "b=3"], ["a=2", "b=3"]]


def test_expand_sweep_overrides_product():
    result = expand_sweep_overrides(["a=1,2", "b=x,y"])
    assert len(result) == 4
    assert ["a=1", "b=x"] in result
    assert ["a=2", "b=y"] in result


def test_expand_sweep_overrides_limit_exceeded():
    overrides = [f"k{i}=0,1,2,3,4,5,6,7,8,9" for i in range(3)]
    with pytest.raises(ValueError, match="Refusing"):
        expand_sweep_overrides(overrides)


def test_expand_sweep_overrides_no_equals():
    result = expand_sweep_overrides(["~datamodule"])
    assert result == [["~datamodule"]]


def test_hydra_string_list_literal_basic():
    assert hydra_string_list_literal(["a", "b"]) == '["a","b"]'


def test_hydra_string_list_literal_escaping():
    result = hydra_string_list_literal(['a"b'])
    assert r"\"" in result


def test_hydra_string_list_literal_empty():
    assert hydra_string_list_literal([]) == "[]"


def test_parse_override_scalar_parses_bool_true_false():
    assert _parse_override_scalar("true") is True
    assert _parse_override_scalar("false") is False


def test_should_use_srun_auto_for_multi_task_or_gpu():
    assert _should_use_srun({"tasks_per_node": 2, "gpus_per_node": 1}) is True
    assert _should_use_srun({"tasks_per_node": 1, "gpus_per_node": 2}) is True


def test_should_use_srun_auto_false_for_single_task_single_gpu():
    assert _should_use_srun({"tasks_per_node": 1, "gpus_per_node": 1}) is False


def test_should_use_srun_respects_explicit_override():
    assert (
        _should_use_srun({"tasks_per_node": 1, "gpus_per_node": 1, "use_srun": True})
        is True
    )
    assert (
        _should_use_srun({"tasks_per_node": 2, "gpus_per_node": 2, "use_srun": False})
        is False
    )


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------


def test_sanitize_name_part_clean():
    assert sanitize_name_part("hello") == "hello"


def test_sanitize_name_part_special_chars():
    assert sanitize_name_part("a b/c") == "a-b-c"


def test_sanitize_name_part_strips_quotes():
    assert sanitize_name_part('"quoted"') == "quoted"


def test_sanitize_name_part_dots_and_dashes_preserved():
    assert sanitize_name_part("v1.0-beta") == "v1.0-beta"


def test_dataset_name_token_known():
    assert dataset_name_token("advection_diffusion_multichannel_64_64", []) == "adm64"


def test_dataset_name_token_unknown_passthrough():
    assert dataset_name_token("my_custom_data", []) == "my_custom_data"


def test_dataset_name_token_datamodule_override_takes_precedence():
    overrides = ["datamodule=reaction_diffusion"]
    assert dataset_name_token("something_else", overrides) == "rd32"


def test_auto_run_name_ae():
    with (
        patch("autocast.scripts.workflow.naming._git_hash", return_value="abc1234"),
        patch("autocast.scripts.workflow.naming._short_uuid", return_value="xyz7890"),
    ):
        name = auto_run_name("ae", "advection_diffusion_multichannel_64_64", [])
    assert name.startswith("ae_adm64_")
    assert "abc1234" in name
    assert "xyz7890" in name


def test_auto_run_name_epd():
    with (
        patch("autocast.scripts.workflow.naming._git_hash", return_value="abc1234"),
        patch("autocast.scripts.workflow.naming._short_uuid", return_value="xyz7890"),
    ):
        name = auto_run_name("epd", "reaction_diffusion", [])
    assert name.startswith("epd_rd32_")


def test_auto_run_name_diff_prefix():
    with (
        patch("autocast.scripts.workflow.naming._git_hash", return_value="abc1234"),
        patch("autocast.scripts.workflow.naming._short_uuid", return_value="xyz7890"),
    ):
        name = auto_run_name(
            "epd",
            "reaction_diffusion",
            ["processor@model.processor=flow_matching_vit"],
        )
    assert name.startswith("diff_")


def test_auto_run_name_crps_prefix():
    with (
        patch("autocast.scripts.workflow.naming._git_hash", return_value="abc1234"),
        patch("autocast.scripts.workflow.naming._short_uuid", return_value="xyz7890"),
    ):
        name = auto_run_name(
            "epd",
            "reaction_diffusion",
            ["model.loss_func._target_=autocast.losses.ensemble.CRPSLoss"],
        )
    assert name.startswith("crps_")


def test_auto_run_name_hidden_dim_included():
    with (
        patch("autocast.scripts.workflow.naming._git_hash", return_value="abc1234"),
        patch("autocast.scripts.workflow.naming._short_uuid", return_value="xyz7890"),
    ):
        name = auto_run_name(
            "epd",
            "reaction_diffusion",
            ["processor@model.processor=fno", "model.processor.hidden_channels=256"],
        )
    assert "256" in name


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def test_build_effective_eval_overrides_filters_train_only():
    train = ["trainer.max_epochs=5", "model.x=1", "optimizer.lr=1e-3"]
    result = build_effective_eval_overrides(train, [])
    assert "model.x=1" in result
    assert "trainer.max_epochs=5" not in result
    assert "optimizer.lr=1e-3" not in result


def test_build_effective_eval_overrides_appended():
    result = build_effective_eval_overrides(["model.x=1"], ["eval.y=2"])
    assert result == ["model.x=1", "eval.y=2"]


def test_build_effective_eval_overrides_order_preserved():
    train = ["model.a=1", "model.b=2"]
    result = build_effective_eval_overrides(train, [])
    assert result == ["model.a=1", "model.b=2"]


def test_infer_dataset_from_workdir_from_datamodule_data_path(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text(
        "datamodule:\n  data_path: /tmp/datasets/reaction_diffusion\n",
        encoding="utf-8",
    )
    assert infer_dataset_from_workdir(tmp_path) == "reaction_diffusion"


def test_infer_dataset_from_workdir_from_datamodule_string(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text(
        'datamodule: "advection_diffusion_multichannel_64_64"\n',
        encoding="utf-8",
    )
    assert (
        infer_dataset_from_workdir(tmp_path) == "advection_diffusion_multichannel_64_64"
    )


def test_infer_resume_checkpoint_kind_specific(tmp_path):
    ckpt = tmp_path / "encoder_processor_decoder.ckpt"
    ckpt.touch()
    assert infer_resume_checkpoint("epd", tmp_path) == ckpt.resolve()


def test_infer_resume_checkpoint_returns_none_when_missing(tmp_path):
    assert infer_resume_checkpoint("epd", tmp_path) is None


def test_infer_hydra_config_from_workdir_prefers_resolved_config(tmp_path):
    (tmp_path / "resolved_autoencoder_config.yaml").write_text(
        "x: 1\n", encoding="utf-8"
    )
    (tmp_path / "resolved_config.yaml").write_text("x: 2\n", encoding="utf-8")

    result = infer_hydra_config_from_workdir(tmp_path)
    assert result == (str(tmp_path.resolve()), "resolved_config")


def test_infer_hydra_config_from_workdir_uses_run_subdir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("x: 1\n", encoding="utf-8")

    result = infer_hydra_config_from_workdir(tmp_path)
    assert result == (str(run_dir.resolve()), "resolved_config")


def test_eval_command_auto_infers_hydra_config(monkeypatch, tmp_path):
    (tmp_path / "resolved_config.yaml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "encoder_processor_decoder.ckpt").touch()
    captured: dict[str, object] = {}

    def _fake_run_module(module, overrides, dry_run=False, mode="local"):
        captured["module"] = module
        captured["overrides"] = overrides
        captured["dry_run"] = dry_run
        captured["mode"] = mode

    monkeypatch.setattr(
        "autocast.scripts.workflow.commands.run_module", _fake_run_module
    )

    eval_command(
        mode="local",
        dataset="reaction_diffusion",
        work_dir=str(tmp_path),
        overrides=["datamodule.batch_size=8"],
        dry_run=True,
    )

    overrides = captured["overrides"]
    assert isinstance(overrides, list)
    assert "--config-name" in overrides
    assert "--config-path" in overrides
    assert "resolved_config" in overrides
    assert str(tmp_path.resolve()) in overrides
    # Defaults-group overrides must be absent when using a resolved config
    assert "eval=encoder_processor_decoder" not in overrides
    assert not any(o.startswith("datamodule=") for o in overrides)
    # Dot-path overrides (e.g. datamodule.data_path) should still be present
    assert any(o.startswith("datamodule.data_path=") for o in overrides)


def test_eval_command_includes_defaults_without_resolved_config(monkeypatch, tmp_path):
    # No resolved_config.yaml in workdir
    (tmp_path / "encoder_processor_decoder.ckpt").touch()
    captured: dict[str, object] = {}

    def _fake_run_module(module, overrides, dry_run=False, mode="local"):  # noqa: ARG001 unused but included for clarity
        captured["overrides"] = overrides

    monkeypatch.setattr(
        "autocast.scripts.workflow.commands.run_module", _fake_run_module
    )

    eval_command(
        mode="local",
        dataset="reaction_diffusion",
        work_dir=str(tmp_path),
        overrides=[],
        dry_run=True,
    )

    overrides = captured["overrides"]
    assert isinstance(overrides, list)
    assert "eval=encoder_processor_decoder" in overrides
    assert any(o.startswith("datamodule=") for o in overrides)


def test_eval_command_keeps_explicit_hydra_config(monkeypatch, tmp_path):
    (tmp_path / "resolved_config.yaml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "encoder_processor_decoder.ckpt").touch()
    captured: dict[str, object] = {}

    def _fake_run_module(module, overrides, dry_run=False, mode="local"):  # noqa: ARG001 unused but included for clarity
        captured["overrides"] = overrides

    monkeypatch.setattr(
        "autocast.scripts.workflow.commands.run_module", _fake_run_module
    )

    eval_command(
        mode="local",
        dataset="reaction_diffusion",
        work_dir=str(tmp_path),
        overrides=["--config-name", "custom", "--config-path", "custom/path"],
        dry_run=True,
    )

    overrides = captured["overrides"]
    assert isinstance(overrides, list)
    assert overrides.count("--config-name") == 1
    assert overrides.count("--config-path") == 1
    assert "custom" in overrides
    assert "custom/path" in overrides


def test_eval_command_explicit_resolved_config_skips_defaults(monkeypatch, tmp_path):
    (tmp_path / "encoder_processor_decoder.ckpt").touch()
    captured: dict[str, object] = {}

    def _fake_run_module(module, overrides, dry_run=False, mode="local"):  # noqa: ARG001 unused but included for clarity
        captured["overrides"] = overrides

    monkeypatch.setattr(
        "autocast.scripts.workflow.commands.run_module", _fake_run_module
    )

    eval_command(
        mode="local",
        dataset="reaction_diffusion",
        work_dir=str(tmp_path),
        overrides=[
            "--config-name",
            "resolved_config",
            "--config-path",
            str(tmp_path),
        ],
        dry_run=True,
    )

    overrides = captured["overrides"]
    assert isinstance(overrides, list)
    assert "eval=encoder_processor_decoder" not in overrides
    assert not any(o.startswith("datamodule=") for o in overrides)
    assert any(o.startswith("datamodule.data_path=") for o in overrides)


def test_build_train_overrides_normalizes_relative_resume_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative_ckpt = "outputs/2026-02-22/run/autoencoder.ckpt"
    expected_ckpt = (tmp_path / relative_ckpt).resolve()

    _work_dir, _run_id, command_overrides = build_train_overrides(
        kind="ae",
        mode="local",
        dataset="reaction_diffusion",
        output_base="outputs",
        run_group="2026-02-22",
        run_id="ae_test",
        work_dir=str(tmp_path / "my_workdir"),
        resume_from=relative_ckpt,
        overrides=[],
    )

    assert f"+resume_from_checkpoint={expected_ckpt}" in command_overrides


def test_build_train_overrides_default_wandb_name_uses_run_id():
    _work_dir, resolved_run_id, command_overrides = build_train_overrides(
        kind="epd",
        mode="local",
        dataset="reaction_diffusion",
        output_base="outputs",
        run_group="2026-02-23",
        run_id="explicit_run",
        work_dir=None,
        resume_from=None,
        overrides=[],
    )

    assert resolved_run_id == "explicit_run"
    assert "logging.wandb.name=explicit_run" in command_overrides


def test_build_train_overrides_explicit_logging_override_wins_default():
    _work_dir, _resolved_run_id, command_overrides = build_train_overrides(
        kind="epd",
        mode="local",
        dataset="reaction_diffusion",
        output_base="outputs",
        run_group="2026-02-23",
        run_id="explicit_run",
        work_dir=None,
        resume_from=None,
        overrides=["logging.wandb.name=from_override"],
    )

    assert "logging.wandb.name=from_override" in command_overrides
    assert "logging.wandb.name=explicit_run" not in command_overrides


def test_build_train_overrides_explicit_logging_override_only_once():
    _work_dir, _resolved_run_id, command_overrides = build_train_overrides(
        kind="epd",
        mode="local",
        dataset="reaction_diffusion",
        output_base="outputs",
        run_group="2026-02-23",
        run_id="explicit_run",
        work_dir=None,
        resume_from=None,
        overrides=["logging.wandb.name=from_override"],
    )

    wandb_entries = [
        o for o in command_overrides if o.startswith("logging.wandb.name=")
    ]
    assert wandb_entries == ["logging.wandb.name=from_override"]


def test_run_module_command_places_config_flags_before_overrides():
    command = run_module_command(
        "autocast.scripts.eval.encoder_processor_decoder",
        [
            "eval.batch_indices=[0,1]",
            "datamodule.batch_size=16",
            "--config-name",
            "resolved_config",
            "--config-path",
            "/tmp/workdir",
        ],
    )

    assert "--config-name" in command
    assert "resolved_config" in command
    assert "--config-path" in command
    assert "/tmp/workdir" in command
    assert command.index("--config-name") < command.index("eval.batch_indices=[0,1]")
    assert command.index("--config-path") < command.index("eval.batch_indices=[0,1]")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_build_parser_ae_basic(parser: argparse.ArgumentParser):
    args = parser.parse_args(["ae"])
    assert args.command == "ae"
    assert args.mode == "local"


def test_build_parser_epd_slurm(parser: argparse.ArgumentParser):
    args = parser.parse_args(["epd", "--mode", "slurm"])
    assert args.mode == "slurm"


def test_build_parser_processor_subcommand(parser: argparse.ArgumentParser):
    args = parser.parse_args(["processor"])
    assert args.command == "processor"


def test_build_parser_eval_basic(parser: argparse.ArgumentParser):
    args = parser.parse_args(["eval", "--workdir", "/tmp/w"])
    assert args.command == "eval"
    assert args.workdir == "/tmp/w"


def test_build_parser_train_eval_with_eval_overrides(
    parser: argparse.ArgumentParser,
):
    args = parser.parse_args(
        [
            "train-eval",
            "trainer.z=3",
            "--eval-overrides",
            "eval.x=1",
            "eval.y=2",
        ]
    )
    assert args.eval_overrides == ["eval.x=1", "eval.y=2"]
    assert "trainer.z=3" in args.overrides


def test_build_parser_override_and_positional_combined(
    parser: argparse.ArgumentParser,
):
    args = parser.parse_args(
        [
            "ae",
            "k1=v1",
            "k2=v2",
        ]
    )
    assert args.overrides == ["k1=v1", "k2=v2"]


def test_build_parser_config_name_passthrough(parser: argparse.ArgumentParser):
    args = parser.parse_args(
        [
            "ae",
            "--config-name",
            "custom_autoencoder",
        ]
    )
    assert args.config_name == "custom_autoencoder"


def test_build_parser_config_path_passthrough(parser: argparse.ArgumentParser):
    args = parser.parse_args(
        [
            "ae",
            "--config-path",
            "src/autocast/configs/variants",
        ]
    )
    assert args.config_path == "src/autocast/configs/variants"


def test_build_parser_train_eval_has_positional_override(
    parser: argparse.ArgumentParser,
):
    args = parser.parse_args(
        [
            "train-eval",
            "my_key=my_val",
        ]
    )
    assert "my_key=my_val" in args.overrides


def test_build_parser_dry_run(parser: argparse.ArgumentParser):
    args = parser.parse_args(["ae", "--dry-run"])
    assert args.dry_run is True


def test_build_parser_resume_from(parser: argparse.ArgumentParser):
    args = parser.parse_args(["epd", "--resume-from", "/ckpt"])
    assert args.resume_from == "/ckpt"


def test_build_parser_run_group_alias(parser: argparse.ArgumentParser):
    args = parser.parse_args(["epd", "--run-group", "my_group"])
    assert args.run_group == "my_group"


def test_build_parser_run_id_alias(parser: argparse.ArgumentParser):
    args = parser.parse_args(["epd", "--run-id", "my_run"])
    assert args.run_id == "my_run"


def test_build_parser_run_label_backward_compatible(parser: argparse.ArgumentParser):
    args = parser.parse_args(["epd", "--run-label", "my_label"])
    assert args.run_group == "my_label"


def test_build_parser_run_name_backward_compatible(parser: argparse.ArgumentParser):
    args = parser.parse_args(["epd", "--run-name", "my_name"])
    assert args.run_id == "my_name"


def test_main_train_eval_dispatches_combined_overrides(monkeypatch):
    captured = {}

    def _fake_train_eval_single_job_command(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "autocast.scripts.workflow.cli.train_eval_single_job_command",
        _fake_train_eval_single_job_command,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autocast",
            "train-eval",
            "datamodule=demo_dataset",
            "--mode",
            "slurm",
            "trainer.max_epochs=1",
            "optimizer.learning_rate=0.001",
            "--eval-overrides",
            "eval.batch_indices=[0,1]",
            "eval.n_members=10",
            "--dry-run",
        ],
    )

    workflow_cli.main()

    assert captured["dataset"] == "demo_dataset"
    assert captured["mode"] == "slurm"
    assert captured["dry_run"] is True
    assert "optimizer.learning_rate=0.001" in captured["train_overrides"]
    assert "trainer.max_epochs=1" in captured["train_overrides"]
    assert "datamodule=demo_dataset" in captured["train_overrides"]
    assert captured["eval_overrides"] == [
        "eval.batch_indices=[0,1]",
        "eval.n_members=10",
    ]


def test_main_ae_dispatches_hydra_config_passthrough(monkeypatch):
    captured = {}

    def _fake_train_command(**kwargs):
        captured.update(kwargs)
        return None, "dummy"

    monkeypatch.setattr(
        "autocast.scripts.workflow.cli.train_command",
        _fake_train_command,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autocast",
            "ae",
            "datamodule=demo_dataset",
            "--config-name",
            "my_ae_top_level",
            "--config-path",
            "src/autocast/configs",
            "trainer.max_epochs=1",
        ],
    )

    workflow_cli.main()

    assert captured["overrides"][:4] == [
        "--config-name",
        "my_ae_top_level",
        "--config-path",
        "src/autocast/configs",
    ]
    assert "trainer.max_epochs=1" in captured["overrides"]
    assert "datamodule=demo_dataset" in captured["overrides"]


def test_main_ae_allows_bare_overrides_before_options(monkeypatch):
    captured = {}

    def _fake_train_command(**kwargs):
        captured.update(kwargs)
        return None, "dummy"

    monkeypatch.setattr(
        "autocast.scripts.workflow.cli.train_command",
        _fake_train_command,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autocast",
            "ae",
            "datamodule=demo_dataset",
            "--mode",
            "slurm",
            "trainer.max_epochs=1",
            "--dry-run",
        ],
    )

    workflow_cli.main()

    assert captured["mode"] == "slurm"
    assert captured["dataset"] == "demo_dataset"
    assert "trainer.max_epochs=1" in captured["overrides"]


def test_main_unknown_dashed_flag_still_errors(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autocast",
            "ae",
            "--not-a-real-flag",
        ],
    )

    with pytest.raises(SystemExit):
        workflow_cli.main()


def test_main_eval_dispatches_inferred_dataset_from_workdir(monkeypatch, tmp_path):
    (tmp_path / "resolved_config.yaml").write_text(
        "datamodule:\n  data_path: /tmp/datasets/reaction_diffusion\n",
        encoding="utf-8",
    )

    captured = {}

    def _fake_eval_command(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "autocast.scripts.workflow.cli.eval_command",
        _fake_eval_command,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["autocast", "eval", "--workdir", str(tmp_path), "--dry-run"],
    )

    workflow_cli.main()

    assert captured["dataset"] == "reaction_diffusion"
    assert captured["work_dir"] == str(tmp_path)


def test_resolve_dataset_from_datamodule_override():
    dataset = workflow_cli._resolve_dataset(
        work_dir=None,
        overrides=["datamodule=reaction_diffusion"],
    )

    assert dataset == "reaction_diffusion"


def test_resolve_dataset_from_datamodule_data_path_override():
    dataset = workflow_cli._resolve_dataset(
        work_dir=None,
        overrides=["datamodule.data_path=/tmp/datasets/advection_diffusion"],
    )

    assert dataset == "advection_diffusion"


def test_main_train_dispatches_dataset_from_datamodule_override(monkeypatch):
    captured = {}

    def _fake_train_command(**kwargs):
        captured.update(kwargs)
        return None, "dummy"

    monkeypatch.setattr(
        "autocast.scripts.workflow.cli.train_command",
        _fake_train_command,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autocast",
            "ae",
            "datamodule=reaction_diffusion",
            "--dry-run",
        ],
    )

    workflow_cli.main()

    assert captured["dataset"] == "reaction_diffusion"


def test_main_train_dispatches_inferred_dataset_and_resume(monkeypatch, tmp_path):
    (tmp_path / "resolved_config.yaml").write_text(
        "datamodule:\n"
        "  data_path: /tmp/datasets/advection_diffusion_multichannel_64_64\n",
        encoding="utf-8",
    )
    ckpt = tmp_path / "encoder_processor_decoder.ckpt"
    ckpt.touch()

    captured = {}

    def _fake_train_command(**kwargs):
        captured.update(kwargs)
        return tmp_path, "dummy"

    monkeypatch.setattr(
        "autocast.scripts.workflow.cli.train_command",
        _fake_train_command,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["autocast", "epd", "--workdir", str(tmp_path), "--dry-run"],
    )

    workflow_cli.main()

    assert captured["dataset"] == "advection_diffusion_multichannel_64_64"
    assert captured["resume_from"] == str(ckpt.resolve())
    assert captured["kind"] == "epd"


def test_main_ae_dispatches_inferred_dataset_and_resume(monkeypatch, tmp_path):
    (tmp_path / "resolved_config.yaml").write_text(
        "datamodule:\n"
        "  data_path: /tmp/datasets/advection_diffusion_multichannel_64_64\n",
        encoding="utf-8",
    )
    ckpt = tmp_path / "autoencoder.ckpt"
    ckpt.touch()

    captured = {}

    def _fake_train_command(**kwargs):
        captured.update(kwargs)
        return tmp_path, "dummy"

    monkeypatch.setattr(
        "autocast.scripts.workflow.cli.train_command",
        _fake_train_command,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["autocast", "ae", "--workdir", str(tmp_path), "--dry-run"],
    )

    workflow_cli.main()

    assert captured["dataset"] == "advection_diffusion_multichannel_64_64"
    assert captured["resume_from"] == str(ckpt.resolve())
    assert captured["kind"] == "ae"
