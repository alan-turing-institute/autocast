"""Prepare, submit, and validate the multi-dataset seed-43 rerun campaign."""

# ruff: noqa: EM101, PLR0912

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from autocast.scripts.workflow.naming import auto_run_name

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
WORKER = SCRIPT_DIR / "worker.sh"
SPLITS = ("train", "valid", "test")
STAGES = ("data", "cache", "crps", "fm", "eval_crps", "eval_fm")
CURRENT_SIMULATOR_TARGETS = {
    "ad": "autosim.simulations.spatiotemporal.AdvectionDiffusion",
    "gpe": "autosim.simulations.spatiotemporal.GrossPitaevskiiEquation2D",
    "gs": "autosim.simulations.spatiotemporal.GrayScott",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
    temporary.replace(path)


def _run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    printable = [str(part) for part in command]
    print(f"+ {shlex.join(printable)}", flush=True)
    return subprocess.run(
        printable,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )


def _git(*args: str, capture: bool = True) -> str:
    result = _run(["git", "-C", REPO_ROOT, *args], capture=capture)
    return result.stdout.strip() if capture else ""


def _source_commit() -> str:
    return _git("rev-parse", "HEAD")


def _repository_commit(repository: Path) -> str:
    result = _run(["git", "-C", repository, "rev-parse", "HEAD"], capture=True)
    return result.stdout.strip()


def _repository_status(repository: Path) -> str:
    result = _run(["git", "-C", repository, "status", "--porcelain"], capture=True)
    return result.stdout.strip()


def _autosim_repo(manifest: dict[str, Any]) -> Path:
    return Path(str(manifest["campaign"]["autosim_repo"])).expanduser().resolve()


def _require_source_ready(manifest: dict[str, Any], expected: str | None = None) -> str:
    current = _source_commit()
    if expected is not None and current != expected:
        raise RuntimeError(
            "The checkout changed after campaign preparation: "
            f"expected {expected}, found {current}"
        )
    required = str(manifest["campaign"]["required_source_commit"])
    ancestor = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", required, "HEAD"],
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(f"Checkout does not contain required fix {required}")
    if _git("status", "--porcelain"):
        raise RuntimeError("Refusing to run from an uncommitted AutoCast checkout")
    return current


def _require_autosim_ready(
    manifest: dict[str, Any], expected: str | None = None
) -> str:
    repository = _autosim_repo(manifest)
    current = _repository_commit(repository)
    if expected is not None and current != expected:
        raise RuntimeError(
            "The AutoSim checkout changed after campaign preparation: "
            f"expected {expected}, found {current}"
        )
    if _repository_status(repository):
        raise RuntimeError("Refusing to run from an uncommitted AutoSim checkout")
    return current


def _absolute_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_resources_and_evaluation(manifest: dict[str, Any]) -> None:
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise TypeError("campaign.yaml must contain a resources mapping")
    for stage in STAGES:
        if stage not in resources:
            raise ValueError(f"Missing resources.{stage}")
        stage_resources = resources[stage]
        if stage_resources["gpus_per_node"] == 1 and stage_resources["mem"] != "115G":
            raise ValueError(f"One-GPU stage {stage} must request mem=115G")

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise TypeError("campaign.yaml must contain an evaluation mapping")
    trajectory = evaluation.get("trajectory_statistics")
    if evaluation.get("aggregate_statistics") is not True:
        raise ValueError("evaluation.aggregate_statistics must be true")
    if not isinstance(trajectory, dict) or trajectory.get("enabled") is not True:
        raise ValueError("evaluation.trajectory_statistics.enabled must be true")
    if trajectory.get("include_per_timestep") is not True:
        raise ValueError(
            "evaluation.trajectory_statistics.include_per_timestep must be true"
        )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    campaign = manifest.get("campaign")
    datasets = manifest.get("datasets")
    if not isinstance(campaign, dict) or not isinstance(datasets, dict):
        raise TypeError("campaign.yaml must contain campaign and datasets mappings")
    _validate_resources_and_evaluation(manifest)
    if set(datasets) != {"ad", "gpe", "gs"}:
        raise ValueError("The campaign must define exactly AD, GPE, and GS")
    run_group = str(campaign["run_group"])
    datetime.strptime(run_group, "%Y-%m-%d")
    required_commit = str(campaign["required_source_commit"])
    if re.fullmatch(r"[0-9a-f]{40}", required_commit) is None:
        raise ValueError("campaign.required_source_commit must be a full git hash")
    autosim_repo = _autosim_repo(manifest)
    if not (autosim_repo / ".git").exists():
        raise FileNotFoundError(f"Missing AutoSim repository: {autosim_repo}")
    for key, spec in datasets.items():
        if not isinstance(spec, dict):
            raise TypeError(f"datasets.{key} must be a mapping")
        schema = spec.get("schema")
        generator = spec.get("generator")
        if not isinstance(schema, dict) or not isinstance(generator, dict):
            raise TypeError(f"datasets.{key} needs schema and generator mappings")
        config_path = (
            autosim_repo / "src/autosim/configs" / f"{generator['config_name']}.yaml"
        )
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        simulator_overrides = [
            item
            for item in generator["overrides"]
            if str(item).startswith("simulator=")
        ]
        if len(simulator_overrides) != 1:
            raise ValueError(f"datasets.{key} needs one simulator group override")
        simulator_group = str(simulator_overrides[0]).split("=", maxsplit=1)[1]
        simulator_path = (
            autosim_repo / "src/autosim/configs/simulator" / f"{simulator_group}.yaml"
        )
        if not simulator_path.is_file():
            raise FileNotFoundError(simulator_path)
        if set(schema["split_sizes"]) != set(SPLITS):
            raise ValueError(f"datasets.{key}.schema.split_sizes is incomplete")
        for dependency in (
            spec["published_dataset_dir"],
            spec["published_ae_run"],
            spec["reference_crps_run"],
            spec["reference_fm_run"],
        ):
            if not _absolute_repo_path(str(dependency)).exists():
                raise FileNotFoundError(dependency)
        ae_run = _absolute_repo_path(str(spec["published_ae_run"]))
        for dependency in ("autoencoder.ckpt", "resolved_autoencoder_config.yaml"):
            if not (ae_run / dependency).is_file():
                raise FileNotFoundError(ae_run / dependency)
        local_configs = (
            spec["cache_experiment"],
            spec["crps_experiment"],
            spec["fm_experiment"],
        )
        for local_config in local_configs:
            path = REPO_ROOT / "local_hydra/local_experiment" / f"{local_config}.yaml"
            if not path.is_file():
                raise FileNotFoundError(path)


def _short_uuid() -> str:
    return uuid.uuid4().hex[:7]


def _custom_run_name(prefix: str, token: str, git_hash: str) -> str:
    return f"{prefix}_{token}_{git_hash}_{_short_uuid()}"


def _load_yaml_list(path: Path) -> list[str]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Expected a YAML string list in {path}")
    return value


def _single_override(overrides: list[str], key: str) -> str:
    prefix = f"{key}="
    matches = [
        value.removeprefix(prefix) for value in overrides if value.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {key} override, found {len(matches)}")
    return matches[0]


def _build_state(
    manifest: dict[str, Any], source_commit: str, autosim_source_commit: str
) -> dict[str, Any]:
    campaign = manifest["campaign"]
    output_root = REPO_ROOT / str(campaign["output_base"]) / str(campaign["run_group"])
    short_hash = source_commit[:7]
    campaign_name = _custom_run_name(
        "campaign_main_comparison_seed43", "runs", short_hash
    )
    campaign_dir = output_root / campaign_name
    runs: dict[str, Any] = {}
    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        for key, spec in manifest["datasets"].items():
            canonical = str(spec["canonical_dataset"])
            crps_experiment = str(spec["crps_experiment"])
            fm_experiment = str(spec["fm_experiment"])
            crps_id = auto_run_name(
                "epd", canonical, [f"local_experiment={crps_experiment}"]
            )
            fm_id = auto_run_name(
                "processor", canonical, [f"local_experiment={fm_experiment}"]
            )
            cache_id = _custom_run_name(
                "cache_published_ae", str(spec["token"]), short_hash
            )
            eval_crps_id = _custom_run_name("eval_crps", str(spec["token"]), short_hash)
            eval_fm_id = _custom_run_name("eval_fm", str(spec["token"]), short_hash)
            runs[key] = {
                "dataset_dir": str(Path(spec["dataset_dir"])),
                "crps_id": crps_id,
                "crps_dir": str(output_root / crps_id),
                "cache_id": cache_id,
                "cache_dir": str(output_root / cache_id),
                "fm_id": fm_id,
                "fm_dir": str(output_root / fm_id),
                "eval_crps_subdir": eval_crps_id,
                "eval_fm_subdir": eval_fm_id,
                "jobs": {},
            }
    finally:
        os.chdir(previous_cwd)
    return {
        "campaign_id": campaign["id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "autosim_source_commit": autosim_source_commit,
        "manifest": str(Path(_manifest_path()).resolve()),
        "campaign_dir": str(campaign_dir),
        "runs": runs,
    }


_ACTIVE_MANIFEST: Path | None = None


def _manifest_path() -> Path:
    if _ACTIVE_MANIFEST is None:
        raise RuntimeError("Manifest path has not been initialized")
    return _ACTIVE_MANIFEST


def _format_plan(manifest: dict[str, Any], state: dict[str, Any] | None = None) -> str:
    campaign = manifest["campaign"]
    lines = [
        f"Campaign: {campaign['id']}",
        f"Run group: outputs/{campaign['run_group']}",
        "Data seed / training seed: "
        f"{campaign['dataset_seed']} / {campaign['training_seed']}",
        "Execution is disabled by default; submission requires --yes-submit.",
        "",
    ]
    if state is not None:
        lines.extend(
            [
                f"AutoCast source: {state['source_commit']}",
                f"AutoSim source:  {state['autosim_source_commit']}",
                "",
            ]
        )
    for key, spec in manifest["datasets"].items():
        lines.extend(
            [
                f"{key.upper()}:",
                f"  data:  {spec['dataset_dir']}",
                f"  AE:    {spec['published_ae_run']} (fixed published AE)",
                f"  CRPS:  {spec['crps_epochs']} epochs, CNS rerun callbacks",
                f"  FM:    {spec['fm_epochs']} epochs, published FM callbacks",
            ]
        )
        if state is not None:
            run = state["runs"][key]
            lines.extend(
                [
                    f"  cache: {run['cache_dir']}",
                    f"  CRPS:  {run['crps_dir']}",
                    f"  FM:    {run['fm_dir']}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "Manual gates:",
            "  data -> inspect/validate -> cache + CRPS -> inspect cache",
            "  -> FM -> inspect checkpoints -> eval_crps + eval_fm",
        ]
    )
    return "\n".join(lines)


def _prepare(manifest: dict[str, Any]) -> Path:
    source_commit = _require_source_ready(manifest)
    autosim_source_commit = _require_autosim_ready(manifest)
    state = _build_state(manifest, source_commit, autosim_source_commit)
    campaign_dir = Path(state["campaign_dir"])
    state_path = campaign_dir / "state.yaml"
    if campaign_dir.exists() or state_path.exists():
        raise FileExistsError(f"Refusing to reuse campaign state path: {state_path}")
    for run in state["runs"].values():
        for key in ("crps_dir", "cache_dir", "fm_dir"):
            if Path(run[key]).exists():
                raise FileExistsError(
                    f"Refusing to reserve existing run path: {run[key]}"
                )
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "slurm_logs").mkdir()
    _write_yaml(state_path, state)
    print(_format_plan(manifest, state))
    print(f"Prepared state: {state_path}")
    return state_path


def _load_state(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    state = _load_yaml(path)
    if state.get("campaign_id") != manifest["campaign"]["id"]:
        raise ValueError("State file belongs to a different campaign")
    if Path(str(state.get("manifest"))).resolve() != _manifest_path().resolve():
        raise ValueError("State file points to a different manifest")
    for key in ("source_commit", "autosim_source_commit"):
        if re.fullmatch(r"[0-9a-f]{40}", str(state.get(key))) is None:
            raise ValueError(f"State file has an invalid {key}")
    return state


def _dataset_validation_marker(spec: dict[str, Any]) -> Path:
    return Path(spec["dataset_dir"]) / "validation_complete.yaml"


def _cache_validation_marker(run: dict[str, Any]) -> Path:
    return Path(run["cache_dir"]) / "validation_complete.yaml"


def _require_dataset_validation(spec: dict[str, Any], state: dict[str, Any]) -> None:
    marker_path = _dataset_validation_marker(spec)
    if not marker_path.is_file():
        raise FileNotFoundError(f"Dataset has not passed validation: {marker_path}")
    marker = _load_yaml(marker_path)
    state_keys = {
        "autocast_source_commit": "source_commit",
        "autosim_source_commit": "autosim_source_commit",
    }
    for key, state_key in state_keys.items():
        if marker.get(key) != state[state_key]:
            raise RuntimeError(
                f"Dataset validation marker has the wrong {key}: {marker_path}"
            )


def _select_crps_checkpoint(run_dir: Path) -> Path:
    matches = sorted(
        run_dir.glob("autocast/*/checkpoints/best-multiwinkler-overall-*.ckpt")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one overall-best multi-Winkler checkpoint in {run_dir}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _preflight_stage(
    manifest: dict[str, Any], state: dict[str, Any], dataset: str, stage: str
) -> None:
    spec = manifest["datasets"][dataset]
    run = state["runs"][dataset]
    if stage in run.get("jobs", {}):
        raise RuntimeError(f"{dataset}/{stage} already has job {run['jobs'][stage]}")
    if stage == "data":
        if Path(spec["dataset_dir"]).exists():
            raise FileExistsError(spec["dataset_dir"])
        return
    _require_dataset_validation(spec, state)
    if stage == "cache":
        if Path(run["cache_dir"]).exists():
            raise FileExistsError(run["cache_dir"])
    elif stage == "crps":
        if Path(run["crps_dir"]).exists():
            raise FileExistsError(run["crps_dir"])
    elif stage == "fm":
        if not _cache_validation_marker(run).is_file():
            raise FileNotFoundError(_cache_validation_marker(run))
        if Path(run["fm_dir"]).exists():
            raise FileExistsError(run["fm_dir"])
    elif stage == "eval_crps":
        run_dir = Path(run["crps_dir"])
        _select_crps_checkpoint(run_dir)
        output = run_dir / run["eval_crps_subdir"]
        if output.exists():
            raise FileExistsError(output)
    elif stage == "eval_fm":
        run_dir = Path(run["fm_dir"])
        for required in (run_dir / "resolved_config.yaml", run_dir / "processor.ckpt"):
            if not required.is_file():
                raise FileNotFoundError(required)
        output = run_dir / run["eval_fm_subdir"]
        if output.exists():
            raise FileExistsError(output)


def _sbatch_command(
    manifest: dict[str, Any], state_path: Path, dataset: str, stage: str
) -> list[str]:
    resources = manifest["resources"][stage]
    campaign_dir = Path(_load_yaml(state_path)["campaign_dir"])
    logs = campaign_dir / "slurm_logs"
    job_name = f"rerun43_{dataset}_{stage}"
    return [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--nodes={resources['nodes']}",
        f"--ntasks-per-node={resources['tasks_per_node']}",
        f"--gpus-per-node={resources['gpus_per_node']}",
        f"--cpus-per-task={resources['cpus_per_task']}",
        f"--time={resources['time']}",
        f"--mem={resources['mem']}",
        f"--output={logs}/{dataset}-{stage}-%j.out",
        f"--error={logs}/{dataset}-{stage}-%j.err",
        str(WORKER),
        str(_manifest_path().resolve()),
        str(state_path.resolve()),
        dataset,
        stage,
    ]


def _submit(
    manifest: dict[str, Any], state_path: Path, datasets: list[str], stage: str
) -> None:
    state = _load_state(state_path, manifest)
    _require_source_ready(manifest, str(state["source_commit"]))
    if stage == "data":
        _require_autosim_ready(manifest, str(state["autosim_source_commit"]))
    for dataset in datasets:
        _preflight_stage(manifest, state, dataset, stage)
    for dataset in datasets:
        result = _run(
            _sbatch_command(manifest, state_path, dataset, stage), capture=True
        )
        job_id = result.stdout.strip().split(";", maxsplit=1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Could not parse sbatch job ID from: {result.stdout!r}")
        state["runs"][dataset].setdefault("jobs", {})[stage] = job_id
        _write_yaml(state_path, state)
        print(f"Submitted {dataset}/{stage}: {job_id}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_command(
    manifest: dict[str, Any], dataset: str
) -> list[str | os.PathLike[str]]:
    campaign = manifest["campaign"]
    spec = manifest["datasets"][dataset]
    generator = spec["generator"]
    schema = spec["schema"]
    dataset_dir = Path(spec["dataset_dir"])
    return [
        "uv",
        "run",
        "--project",
        _autosim_repo(manifest),
        "--frozen",
        "--no-sync",
        "autosim",
        f"--config-name={generator['config_name']}",
        *generator["overrides"],
        f"dataset.output_dir={dataset_dir}",
        f"dataset.n_train={schema['split_sizes']['train']}",
        f"dataset.n_valid={schema['split_sizes']['valid']}",
        f"dataset.n_test={schema['split_sizes']['test']}",
        f"seed={campaign['dataset_seed']}",
        "overwrite=false",
    ]


def _run_data(manifest: dict[str, Any], state: dict[str, Any], dataset: str) -> None:
    spec = manifest["datasets"][dataset]
    campaign = manifest["campaign"]
    dataset_dir = Path(spec["dataset_dir"])
    if dataset_dir.exists():
        raise FileExistsError(dataset_dir)
    _run(_generation_command(manifest, dataset), cwd=REPO_ROOT)
    provenance = {
        "campaign": campaign["id"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_seed": campaign["dataset_seed"],
        "training_seed": campaign["training_seed"],
        "autocast_source_commit": state["source_commit"],
        "autosim_source_commit": state["autosim_source_commit"],
        "published_reference_dataset": spec["published_dataset_dir"],
    }
    _write_yaml(dataset_dir / "provenance.yaml", provenance)
    validate_dataset(manifest, dataset)
    artifacts = [
        *(dataset_dir / split / "data.pt" for split in SPLITS),
        dataset_dir / "stats.yml",
        dataset_dir / "resolved_config.yaml",
        dataset_dir / "provenance.yaml",
    ]
    checksums = {
        str(path.relative_to(dataset_dir)): _sha256(path) for path in artifacts
    }
    _write_yaml(dataset_dir / "artifact_sha256.yaml", checksums)
    _write_yaml(
        _dataset_validation_marker(spec),
        {
            "validated_at_utc": datetime.now(UTC).isoformat(),
            "autocast_source_commit": state["source_commit"],
            "autosim_source_commit": state["autosim_source_commit"],
        },
    )


def _hydra_list(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _training_overrides(
    manifest: dict[str, Any], state: dict[str, Any], dataset: str, stage: str
) -> tuple[str, list[str]]:
    campaign = manifest["campaign"]
    spec = manifest["datasets"][dataset]
    run = state["runs"][dataset]
    refs = campaign["reference_source_commits"]
    common = [
        f"seed={campaign['training_seed']}",
        "logging.wandb.enabled=true",
        f"+provenance.source_commit={state['source_commit']}",
        f"+provenance.dataset_seed={campaign['dataset_seed']}",
    ]
    if stage == "crps":
        epochs = int(spec["crps_epochs"])
        overrides = [
            f"logging.wandb.name={run['crps_id']}",
            f"local_experiment={spec['crps_experiment']}",
            "trainer=crps_main_comparison_rerun",
            f"datamodule.data_path={spec['dataset_dir']}",
            f"datamodule.normalization_path={spec['dataset_dir']}/stats.yml",
            *common,
            f"+provenance.reference_source_commit={refs['crps']}",
            f"+provenance.reference_run={Path(spec['reference_crps_run']).name}",
            f"optimizer.cosine_epochs={epochs}",
            f"+trainer.max_epochs={epochs}",
            "trainer.max_time=00:23:59:00",
            f"hydra.run.dir={run['crps_dir']}",
        ]
        return "autocast.scripts.train.encoder_processor_decoder", overrides
    epochs = int(spec["fm_epochs"])
    quarter_epochs = epochs // 4
    overrides = [
        f"logging.wandb.name={run['fm_id']}",
        f"local_experiment={spec['fm_experiment']}",
        "trainer=fm_main_comparison",
        f"datamodule.data_path={run['cache_dir']}",
        *common,
        f"+provenance.reference_source_commit={refs['flow_matching']}",
        f"+provenance.reference_run={Path(spec['reference_fm_run']).name}",
        f"+provenance.cache_source_run={run['cache_id']}",
        f"optimizer.cosine_epochs={epochs}",
        f"+trainer.max_epochs={epochs}",
        "trainer.max_time=00:23:59:00",
        f"trainer.callbacks.0.every_n_epochs={quarter_epochs}",
        "trainer.callbacks.0.save_top_k=-1",
        'trainer.callbacks.0.filename="quarter-{epoch:04d}"',
        f"hydra.run.dir={run['fm_dir']}",
    ]
    return "autocast.scripts.train.processor", overrides


def _run_training(
    manifest: dict[str, Any], state: dict[str, Any], dataset: str, stage: str
) -> None:
    module, overrides = _training_overrides(manifest, state, dataset, stage)
    run_dir = Path(state["runs"][dataset][f"{stage}_dir"])
    if run_dir.exists():
        raise FileExistsError(run_dir)
    _run(
        ["srun", "--kill-on-bad-exit=1", sys.executable, "-m", module, *overrides],
        cwd=REPO_ROOT,
    )


def _autocast_executable() -> Path:
    executable = Path(sys.executable).parent / "autocast"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return executable


def _run_cache(manifest: dict[str, Any], state: dict[str, Any], dataset: str) -> None:
    campaign = manifest["campaign"]
    spec = manifest["datasets"][dataset]
    run = state["runs"][dataset]
    cache_dir = Path(run["cache_dir"])
    ae_run = _absolute_repo_path(str(spec["published_ae_run"]))
    if cache_dir.exists():
        raise FileExistsError(cache_dir)
    # Start from the cache experiment rather than the resolved AE training
    # config: caching requires full_trajectory_mode, not autoencoder_mode.
    command = [
        "srun",
        "--kill-on-bad-exit=1",
        _autocast_executable(),
        "cache-latents",
        "--mode",
        "local",
        "--workdir",
        cache_dir,
        "--output-dir",
        cache_dir,
        f"autoencoder_checkpoint={ae_run / 'autoencoder.ckpt'}",
        f"local_experiment={spec['cache_experiment']}",
        f"datamodule.data_path={spec['dataset_dir']}",
        f"datamodule.normalization_path={spec['published_dataset_dir']}/stats.yml",
        f"+provenance.source_commit={state['source_commit']}",
        f"+provenance.reference_ae_source_commit={campaign['reference_source_commits']['autoencoder']}",
        f"+provenance.cache_source_run={ae_run.name}",
        f"+provenance.dataset_seed={campaign['dataset_seed']}",
        f"+provenance.normalization_source_dataset={Path(spec['published_dataset_dir']).name}",
    ]
    _run(command, cwd=REPO_ROOT)
    validate_cache(manifest, state, dataset)
    _write_yaml(
        _cache_validation_marker(run),
        {
            "validated_at_utc": datetime.now(UTC).isoformat(),
            "autocast_source_commit": state["source_commit"],
            "published_ae_run": str(ae_run),
        },
    )


def _eval_overrides(
    manifest: dict[str, Any], state: dict[str, Any], dataset: str, stage: str
) -> tuple[Path, str, list[str]]:
    spec = manifest["datasets"][dataset]
    run = state["runs"][dataset]
    evaluation = manifest["evaluation"]
    trajectory_statistics = evaluation["trajectory_statistics"]
    is_crps = stage == "eval_crps"
    run_dir = Path(run["crps_dir"] if is_crps else run["fm_dir"])
    output_subdir = str(run["eval_crps_subdir"] if is_crps else run["eval_fm_subdir"])
    checkpoint = (
        _select_crps_checkpoint(run_dir) if is_crps else run_dir / "processor.ckpt"
    )
    published_ae_checkpoint = (
        _absolute_repo_path(str(spec["published_ae_run"])) / "autoencoder.ckpt"
    )
    autoencoder = (
        "autoencoder_checkpoint=null"
        if is_crps
        else f"+autoencoder_checkpoint={published_ae_checkpoint}"
    )
    mode = "ambient" if is_crps else "encode_once"
    batch_size = spec["eval_batch_size"]["crps" if is_crps else "fm"]
    output_dir = run_dir / output_subdir
    trajectory_dir = output_dir / "trajectory_statistics"
    overrides = [
        f"eval.checkpoint={checkpoint}",
        autoencoder,
        f"eval.mode={mode}",
        "eval.accelerator=cuda",
        "eval.devices=1",
        f"eval.batch_size={batch_size}",
        f"eval.n_members={evaluation['n_members']}",
        "eval.max_test_batches=null",
        "eval.max_rollout_batches=null",
        "eval.max_rollout_steps=25",
        f"eval.metric_windows={_hydra_list(evaluation['metric_windows'])}",
        f"eval.metric_windows_rollout={_hydra_list(evaluation['rollout_windows'])}",
        f"eval.metrics={_hydra_list(evaluation['metrics'])}",
        "eval.compute_rollout_coverage=true",
        f"eval.compute_rollout_metrics={str(evaluation['aggregate_statistics']).lower()}",
        f"eval.batch_indices={_hydra_list(evaluation['visual_batch_indices'])}",
        "eval.save_rollout_snapshots=true",
        f"eval.rollout_snapshot_timesteps={_hydra_list(evaluation['snapshot_timesteps'])}",
        "eval.rollout_snapshot_format=png",
        "eval.benchmark.enabled=true",
        "eval.benchmark_rollout.enabled=true",
        f"eval.csv_path={output_dir / 'evaluation_metrics.csv'}",
        f"eval.video_dir={output_dir / 'videos'}",
        "eval.trajectory_statistics.enabled="
        f"{str(trajectory_statistics['enabled']).lower()}",
        f"eval.trajectory_statistics.output_dir={trajectory_dir}",
        "eval.trajectory_statistics.overwrite_existing=false",
        "eval.trajectory_statistics.sampling_seed=42",
        "eval.trajectory_statistics.include_per_timestep="
        f"{str(trajectory_statistics['include_per_timestep']).lower()}",
        "logging.wandb.enabled=false",
    ]
    return run_dir, output_subdir, overrides


def _run_eval(
    manifest: dict[str, Any], state: dict[str, Any], dataset: str, stage: str
) -> None:
    run_dir, output_subdir, overrides = _eval_overrides(manifest, state, dataset, stage)
    output_dir = run_dir / output_subdir
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _run(
        [
            "srun",
            "--kill-on-bad-exit=1",
            _autocast_executable(),
            "eval",
            "--mode",
            "local",
            "--workdir",
            run_dir,
            "--output-subdir",
            output_subdir,
            *overrides,
        ],
        cwd=REPO_ROOT,
    )
    validate_eval(manifest, state, dataset, stage)


def _run_stage(
    manifest: dict[str, Any], state_path: Path, dataset: str, stage: str
) -> None:
    state = _load_state(state_path, manifest)
    _require_source_ready(manifest, str(state["source_commit"]))
    if stage != "data":
        _require_dataset_validation(manifest["datasets"][dataset], state)
    if stage == "data":
        _require_autosim_ready(manifest, str(state["autosim_source_commit"]))
        _run_data(manifest, state, dataset)
    elif stage == "cache":
        _run_cache(manifest, state, dataset)
    elif stage in {"crps", "fm"}:
        _run_training(manifest, state, dataset, stage)
    else:
        _run_eval(manifest, state, dataset, stage)


def _canonical_visualization(config: dict[str, Any]) -> None:
    visualization = config.get("visualize")
    if not isinstance(visualization, dict):
        return
    batch_indices = visualization.pop("batch_indices", None)
    max_examples = visualization.pop("max_examples", None)
    if isinstance(batch_indices, list):
        visualization["example_count"] = len(batch_indices)
    elif max_examples is not None:
        visualization["example_count"] = int(max_examples)


def _canonical_simulator(config: dict[str, Any], dataset: str) -> None:
    simulator = config.get("simulator")
    if not isinstance(simulator, dict):
        raise TypeError("Generated config has no simulator mapping")
    target = str(simulator.get("_target_"))
    if dataset == "ad":
        if target.endswith(".AdvectionDiffusionMultichannel"):
            if simulator.get("output_indices") != [0]:
                raise ValueError("Published AD simulator is not vorticity-only")
            simulator.pop("output_indices")
        elif not target.endswith(".AdvectionDiffusion"):
            raise ValueError(f"Unexpected AD simulator target: {target}")
        simulator["_target_"] = "advection_diffusion_vorticity"
    else:
        simulator["_target_"] = target.rsplit(".", maxsplit=1)[-1]


def _normalized_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    copied = json.loads(json.dumps(config))
    copied["seed"] = "<new-data-seed>"
    copied.setdefault("dataset", {})["output_dir"] = "<dataset-output>"
    if dataset == "ad":
        copied["dataset"]["ensure_exact_n"] = True
    _canonical_visualization(copied)
    _canonical_simulator(copied, dataset)
    normalization = copied.get("normalization")
    if normalization == {"shared_core_field_groups": []}:
        copied.pop("normalization")
    return copied


def _preflight(manifest: dict[str, Any]) -> None:
    """Audit cluster-local campaign references without creating any outputs."""
    source_commit = _require_source_ready(manifest)
    autosim_commit = _require_autosim_ready(manifest)
    shared_callbacks = _load_yaml(
        REPO_ROOT / "src/autocast/configs/trainer/crps_main_comparison_rerun.yaml"
    )["callbacks"]
    cns_callbacks = _load_yaml(
        REPO_ROOT
        / "local_hydra/local_experiment/reruns/main_comparison_cns_seed43"
        / "crps_vit_azula_large.yaml"
    )["trainer"]["callbacks"]
    if shared_callbacks != cns_callbacks:
        raise ValueError("Shared CRPS callbacks differ from the successful CNS rerun")

    autosim_config_dir = _autosim_repo(manifest) / "src/autosim/configs"
    for dataset, spec in manifest["datasets"].items():
        published_dir = Path(spec["published_dataset_dir"])
        published_overrides = _load_yaml_list(published_dir / ".hydra/overrides.yaml")
        expected_generator = [
            item.replace("simulator=", "simulator=spatiotemporal/", 1)
            if item.startswith("simulator=")
            else item
            for item in published_overrides
            if not item.startswith("seed=")
        ]
        if spec["generator"]["overrides"] != expected_generator:
            raise ValueError(f"{dataset} generator overrides differ from published")

        current_overrides = [
            *spec["generator"]["overrides"],
            f"seed={manifest['campaign']['dataset_seed']}",
            "dataset.output_dir=/tmp/autosim-config-check",
        ]
        with initialize_config_dir(
            version_base=None, config_dir=str(autosim_config_dir)
        ):
            current_cfg = compose(
                config_name=spec["generator"]["config_name"],
                overrides=current_overrides,
            )
        current_value = OmegaConf.to_container(current_cfg, resolve=True)
        if not isinstance(current_value, dict):
            raise TypeError(f"{dataset} AutoSim config did not compose to a mapping")
        current = cast(dict[str, Any], current_value)
        if (
            current.get("simulator", {}).get("_target_")
            != CURRENT_SIMULATOR_TARGETS[dataset]
        ):
            raise ValueError(f"{dataset} did not compose the current simulator target")
        published = _load_yaml(published_dir / "resolved_config.yaml")
        if _normalized_config(current, dataset) != _normalized_config(
            published, dataset
        ):
            raise ValueError(
                f"{dataset} current AutoSim config differs from published science"
            )

        crps_overrides = _load_yaml_list(
            _absolute_repo_path(str(spec["reference_crps_run"]))
            / ".hydra/overrides.yaml"
        )
        fm_overrides = _load_yaml_list(
            _absolute_repo_path(str(spec["reference_fm_run"])) / ".hydra/overrides.yaml"
        )
        if (
            int(_single_override(crps_overrides, "optimizer.cosine_epochs"))
            != spec["crps_epochs"]
        ):
            raise ValueError(f"{dataset} CRPS epoch budget differs from published")
        if (
            int(_single_override(fm_overrides, "optimizer.cosine_epochs"))
            != spec["fm_epochs"]
        ):
            raise ValueError(f"{dataset} FM epoch budget differs from published")

    gpe = manifest["datasets"]["gpe"]
    gpe_ae = _load_yaml(
        _absolute_repo_path(str(gpe["published_ae_run"]))
        / "resolved_autoencoder_config.yaml"
    )
    if gpe_ae["datamodule"].get("channel_idxs") != gpe["schema"].get("channel_idxs"):
        raise ValueError("GPE channel selection differs from the published AE")
    print(f"Preflight passed for AutoCast {source_commit}")
    print(f"Preflight passed for AutoSim  {autosim_commit}")
    print("Aggregate and per-trajectory evaluation outputs are enabled")


def _validate_stats(path: Path, expected_fields: list[str]) -> None:
    stats = _load_yaml(path)
    if stats.get("core_field_names") != expected_fields:
        raise ValueError(f"Unexpected core fields in {path}")
    buckets = stats.get("stats")
    if not isinstance(buckets, dict):
        raise TypeError(f"Missing stats mapping in {path}")
    for name in ("mean", "std", "mean_delta", "std_delta"):
        values = buckets.get(name)
        if not isinstance(values, dict) or set(values) != set(expected_fields):
            raise ValueError(f"Unexpected stats.{name} in {path}")
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"Non-finite stats.{name} in {path}")
    if any(float(value) <= 0 for value in buckets["std"].values()):
        raise ValueError(f"Non-positive standard deviation in {path}")


def validate_dataset(manifest: dict[str, Any], dataset: str) -> None:
    """Validate a generated dataset against its manifest and published source."""
    spec = manifest["datasets"][dataset]
    schema = spec["schema"]
    dataset_dir = Path(spec["dataset_dir"])
    published_dir = Path(spec["published_dataset_dir"])
    generated_cfg = _load_yaml(dataset_dir / "resolved_config.yaml")
    published_cfg = _load_yaml(published_dir / "resolved_config.yaml")
    generated_target = generated_cfg.get("simulator", {}).get("_target_")
    if generated_target != CURRENT_SIMULATOR_TARGETS[dataset]:
        raise ValueError(
            f"{dataset} did not use the current AutoSim simulator target: "
            f"{generated_target}"
        )
    if _normalized_config(generated_cfg, dataset) != _normalized_config(
        published_cfg, dataset
    ):
        raise ValueError(
            f"{dataset} generated config differs from the published procedure "
            "outside seed and output path"
        )
    if generated_cfg.get("seed") != manifest["campaign"]["dataset_seed"]:
        raise ValueError(f"{dataset} did not record seed 43")
    _validate_stats(dataset_dir / "stats.yml", schema["core_field_names"])
    for split in SPLITS:
        path = dataset_dir / split / "data.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a mapping in {path}")
        data = payload.get("data")
        constants = payload.get("constant_scalars")
        expected_shape = (
            schema["split_sizes"][split],
            schema["frames"],
            schema["height"],
            schema["width"],
            schema["channels"],
        )
        if not isinstance(data, torch.Tensor) or tuple(data.shape) != expected_shape:
            raise ValueError(
                f"Unexpected data shape in {path}: {getattr(data, 'shape', None)}"
            )
        expected_constants = (schema["split_sizes"][split], schema["constant_scalars"])
        if (
            not isinstance(constants, torch.Tensor)
            or tuple(constants.shape) != expected_constants
        ):
            raise ValueError(
                f"Unexpected constant_scalars shape in {path}: "
                f"{getattr(constants, 'shape', None)}"
            )
        for sample in (data[0], data[-1], constants[0], constants[-1]):
            if not torch.isfinite(sample).all():
                raise ValueError(f"Non-finite representative sample in {path}")
        published = torch.load(
            published_dir / split / "data.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if torch.equal(data[0], published["data"][0]):
            raise ValueError(
                f"{dataset}/{split} first sample matches the old data draw"
            )
    print(f"Validated dataset: {dataset_dir}")


def _resolve_config_path(value: Any, data_path: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(str(data_path)).expanduser() / path).resolve()


def validate_cache(
    manifest: dict[str, Any], state: dict[str, Any], dataset: str
) -> None:
    """Validate latent files and fixed-published-AE configuration identity."""
    spec = manifest["datasets"][dataset]
    run = state["runs"][dataset]
    cache_dir = Path(run["cache_dir"])
    ae_run = _absolute_repo_path(str(spec["published_ae_run"]))
    ae_config = _load_yaml(ae_run / "resolved_autoencoder_config.yaml")
    cache_config = _load_yaml(cache_dir / "autoencoder_config.yaml")
    ae_data = ae_config["datamodule"]["data_path"]
    cache_data = cache_config["datamodule"]["data_path"]
    if Path(str(cache_data)).resolve() != Path(spec["dataset_dir"]).resolve():
        raise ValueError("Cache does not identify the new raw dataset")
    for key in ("n_steps_input", "n_steps_output", "stride", "use_normalization"):
        if cache_config["datamodule"].get(key) != ae_config["datamodule"].get(key):
            raise ValueError(f"Cache differs from published AE datamodule.{key}")
    if cache_config["datamodule"].get("channel_idxs") != ae_config["datamodule"].get(
        "channel_idxs"
    ):
        raise ValueError("Cache channel_idxs differ from the published AE")
    if cache_config["datamodule"].get("channel_idxs") != spec["schema"].get(
        "channel_idxs"
    ):
        raise ValueError("Cache channel_idxs differ from the campaign schema")
    ae_norm = _resolve_config_path(
        ae_config["datamodule"]["normalization_path"], ae_data
    )
    cache_norm = _resolve_config_path(
        cache_config["datamodule"]["normalization_path"], cache_data
    )
    if cache_norm != ae_norm:
        raise ValueError("Cache is not using the published AE normalization")
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    published_metadata = json.loads(
        (ae_run / "cached_latents/metadata.json").read_text(encoding="utf-8")
    )
    for split in SPLITS:
        expected_size = spec["schema"]["split_sizes"][split]
        details = metadata["splits"][split]
        reference_details = published_metadata["splits"][split]
        if details["num_trajectories"] != expected_size:
            raise ValueError(f"Unexpected cached count for {dataset}/{split}")
        if details["sample_shape"] != reference_details["sample_shape"]:
            raise ValueError(
                f"Cached shape differs from the published AE cache: {split}"
            )
        split_dir = cache_dir / split
        files = sorted(split_dir.glob("traj_*.pt"))
        expected_names = [f"traj_{index:06d}.pt" for index in range(expected_size)]
        if [path.name for path in files] != expected_names:
            raise ValueError(f"Unexpected trajectory files in {split_dir}")
        for path in (files[0], files[-1]):
            payload = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
            if set(payload) != {"encoded_fields", "global_cond"}:
                raise ValueError(f"Unexpected keys in {path}")
            if not all(torch.isfinite(tensor).all() for tensor in payload.values()):
                raise ValueError(f"Non-finite cached trajectory in {path}")
    print(f"Validated cache: {cache_dir}")


def validate_eval(
    manifest: dict[str, Any], state: dict[str, Any], dataset: str, stage: str
) -> None:
    """Validate the required tables and visual artifacts from an evaluation."""
    del manifest
    run = state["runs"][dataset]
    is_crps = stage == "eval_crps"
    run_dir = Path(run["crps_dir"] if is_crps else run["fm_dir"])
    output_subdir = run["eval_crps_subdir"] if is_crps else run["eval_fm_subdir"]
    trajectory = run_dir / output_subdir / "trajectory_statistics"
    required = (
        "resolved_eval_config.yaml",
        "evaluation_metrics.csv",
        "evaluation_metadata.csv",
        "benchmark_metrics.csv",
        "rollout_metrics.csv",
        "single_step_metrics_per_trajectory.csv",
        "rollout_metrics_per_trajectory.csv",
        "rollout_metrics_per_timestep_per_trajectory.csv",
    )
    for filename in required:
        path = trajectory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    videos = list((trajectory / "videos").glob("*.mp4"))
    snapshots = list((trajectory / "videos/snapshots").glob("*.png"))
    if len(videos) < 8 or len(snapshots) < 24:
        raise ValueError(
            "Incomplete visual outputs: "
            f"{len(videos)} videos, {len(snapshots)} snapshots"
        )
    print(f"Validated evaluation: {trajectory}")


def _validate_requested(
    manifest: dict[str, Any], state: dict[str, Any] | None, dataset: str, stage: str
) -> None:
    if stage == "data":
        validate_dataset(manifest, dataset)
    elif stage == "cache":
        if state is None:
            raise ValueError("--state is required for cache validation")
        validate_cache(manifest, state, dataset)
    elif stage in {"eval_crps", "eval_fm"}:
        if state is None:
            raise ValueError("--state is required for eval validation")
        validate_eval(manifest, state, dataset, stage)
    elif stage == "crps":
        if state is None:
            raise ValueError("--state is required for training validation")
        run_dir = Path(state["runs"][dataset]["crps_dir"])
        _select_crps_checkpoint(run_dir)
        if not (run_dir / "resolved_config.yaml").is_file():
            raise FileNotFoundError(run_dir / "resolved_config.yaml")
        print(f"Validated CRPS run: {run_dir}")
    elif stage == "fm":
        if state is None:
            raise ValueError("--state is required for training validation")
        run_dir = Path(state["runs"][dataset]["fm_dir"])
        for path in (run_dir / "resolved_config.yaml", run_dir / "processor.ckpt"):
            if not path.is_file():
                raise FileNotFoundError(path)
        print(f"Validated FM run: {run_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SCRIPT_DIR / "campaign.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Show the immutable campaign specification")
    subparsers.add_parser("preflight", help="Audit local references without writing")
    subparsers.add_parser("prepare", help="Reserve named run paths in state")
    status = subparsers.add_parser("status", help="Print a prepared campaign state")
    status.add_argument("--state", type=Path, required=True)
    submit = subparsers.add_parser("submit", help="Submit one manually gated stage")
    submit.add_argument("--state", type=Path, required=True)
    submit.add_argument("--dataset", choices=("ad", "gpe", "gs", "all"), required=True)
    submit.add_argument("--stage", choices=STAGES, required=True)
    submit.add_argument("--yes-submit", action="store_true")
    run_stage = subparsers.add_parser("run-stage", help=argparse.SUPPRESS)
    run_stage.add_argument("--state", type=Path, required=True)
    run_stage.add_argument("--dataset", choices=("ad", "gpe", "gs"), required=True)
    run_stage.add_argument("--stage", choices=STAGES, required=True)
    validate = subparsers.add_parser("validate", help="Validate a completed artifact")
    validate.add_argument("--state", type=Path)
    validate.add_argument("--dataset", choices=("ad", "gpe", "gs"), required=True)
    validate.add_argument("--stage", choices=STAGES, required=True)
    return parser.parse_args()


def main() -> None:
    """Dispatch campaign planning, submission, workers, and validation."""
    global _ACTIVE_MANIFEST  # noqa: PLW0603
    args = _parse_args()
    manifest_path = args.manifest.resolve()
    _ACTIVE_MANIFEST = manifest_path
    manifest = _load_yaml(manifest_path)
    _validate_manifest(manifest)
    if args.command == "plan":
        print(_format_plan(manifest))
    elif args.command == "preflight":
        _preflight(manifest)
    elif args.command == "prepare":
        _prepare(manifest)
    elif args.command == "status":
        state = _load_state(args.state, manifest)
        print(_format_plan(manifest, state))
        print(yaml.safe_dump({key: run["jobs"] for key, run in state["runs"].items()}))
    elif args.command == "submit":
        if not args.yes_submit:
            raise RuntimeError("Submission is disarmed; pass --yes-submit after review")
        datasets = (
            list(manifest["datasets"]) if args.dataset == "all" else [args.dataset]
        )
        _submit(manifest, args.state, datasets, args.stage)
    elif args.command == "run-stage":
        _run_stage(manifest, args.state, args.dataset, args.stage)
    elif args.command == "validate":
        state = _load_state(args.state, manifest) if args.state else None
        _validate_requested(manifest, state, args.dataset, args.stage)


if __name__ == "__main__":
    main()
