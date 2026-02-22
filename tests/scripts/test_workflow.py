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
    resolve_eval_checkpoint,
)
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


def test_resolve_eval_checkpoint_explicit(tmp_path):
    ckpt = tmp_path / "my.ckpt"
    ckpt.touch()
    result = resolve_eval_checkpoint(tmp_path, str(ckpt))
    assert result == ckpt.resolve()


def test_resolve_eval_checkpoint_auto_epd(tmp_path):
    ckpt = tmp_path / "encoder_processor_decoder.ckpt"
    ckpt.touch()
    result = resolve_eval_checkpoint(tmp_path, None)
    assert result == ckpt


def test_resolve_eval_checkpoint_auto_ae(tmp_path):
    ckpt = tmp_path / "autoencoder.ckpt"
    ckpt.touch()
    result = resolve_eval_checkpoint(tmp_path, None)
    assert result == ckpt


def test_resolve_eval_checkpoint_auto_model(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.touch()
    result = resolve_eval_checkpoint(tmp_path, None)
    assert result == ckpt


def test_resolve_eval_checkpoint_fallback_when_nothing_exists(tmp_path):
    result = resolve_eval_checkpoint(tmp_path, None)
    assert result == tmp_path / "encoder_processor_decoder.ckpt"


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


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_build_parser_ae_basic(parser: argparse.ArgumentParser):
    args = parser.parse_args(["ae", "--dataset", "mydata"])
    assert args.command == "ae"
    assert args.dataset == "mydata"
    assert args.mode == "local"


def test_build_parser_epd_slurm(parser: argparse.ArgumentParser):
    args = parser.parse_args(["epd", "--dataset", "d", "--mode", "slurm"])
    assert args.mode == "slurm"


def test_build_parser_processor_subcommand(parser: argparse.ArgumentParser):
    args = parser.parse_args(["processor", "--dataset", "d"])
    assert args.command == "processor"


def test_build_parser_eval_basic(parser: argparse.ArgumentParser):
    args = parser.parse_args(["eval", "--dataset", "d", "--workdir", "/tmp/w"])
    assert args.command == "eval"
    assert args.workdir == "/tmp/w"


def test_build_parser_train_eval_with_eval_overrides(
    parser: argparse.ArgumentParser,
):
    args = parser.parse_args(
        [
            "train-eval",
            "--dataset",
            "d",
            "trainer.z=3",
            "--eval-overrides",
            "eval.x=1",
            "eval.y=2",
        ]
    )
    assert args.eval_overrides == ["eval.x=1", "eval.y=2"]
    assert "trainer.z=3" in args.overrides


def test_build_parser_detach_flag(parser: argparse.ArgumentParser):
    args = parser.parse_args(["ae", "--dataset", "d", "--detach"])
    assert args.detach is True


def test_build_parser_override_flag(parser: argparse.ArgumentParser):
    args = parser.parse_args(
        [
            "ae",
            "--dataset",
            "d",
            "--override",
            "k1=v1",
            "--override",
            "k2=v2",
        ]
    )
    assert args.override == ["k1=v1", "k2=v2"]


def test_build_parser_override_and_positional_combined(
    parser: argparse.ArgumentParser,
):
    args = parser.parse_args(
        [
            "ae",
            "--dataset",
            "d",
            "--override",
            "k1=v1",
            "k2=v2",
        ]
    )
    assert args.override == ["k1=v1"]
    assert args.overrides == ["k2=v2"]


def test_build_parser_train_eval_has_override(
    parser: argparse.ArgumentParser,
):
    args = parser.parse_args(
        [
            "train-eval",
            "--dataset",
            "d",
            "--override",
            "my_key=my_val",
        ]
    )
    assert "my_key=my_val" in args.override


def test_build_parser_dry_run(parser: argparse.ArgumentParser):
    args = parser.parse_args(["ae", "--dataset", "d", "--dry-run"])
    assert args.dry_run is True


def test_build_parser_resume_from(parser: argparse.ArgumentParser):
    args = parser.parse_args(["epd", "--dataset", "d", "--resume-from", "/ckpt"])
    assert args.resume_from == "/ckpt"


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
            "--dataset",
            "demo_dataset",
            "--mode",
            "slurm",
            "trainer.max_epochs=1",
            "--override",
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
    assert captured["train_overrides"] == [
        "optimizer.learning_rate=0.001",
        "trainer.max_epochs=1",
    ]
    assert captured["eval_overrides"] == [
        "eval.batch_indices=[0,1]",
        "eval.n_members=10",
    ]
