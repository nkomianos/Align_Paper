"""Reproducibility manifests with an allowlisted environment capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config_hash
from .io import sha256_file, write_json


SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "TOKENIZERS_PARALLELISM",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "UE_INSTANCE_ID",
    "UE_INSTANCE_TYPE",
    "UE_INSTANCE_LAUNCHED_AT",
    "UE_INSTANCE_START_EPOCH",
    "UE_HOURLY_USD",
    "UE_LAMBDA_IMAGE_ID",
    "UE_HARD_DEADLINE_EPOCH",
    "UE_COMPUTE_DEADLINE_EPOCH",
    "UE_TERMINATION_DEADLINE_EPOCH",
)


def _command(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True, timeout=15)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def project_hash(project_root: Path) -> str:
    digest = hashlib.sha256()
    included: list[Path] = []
    for relative_root in ("src", "configs", "scripts", "tests", "requirements", "docs"):
        root = project_root / relative_root
        if root.exists():
            included.extend(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    included.extend(path for path in (project_root / "pyproject.toml", project_root / "README.md") if path.exists())
    for path in sorted(included):
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "safe_environment": {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ},
        "packages": {},
    }
    for distribution in (
        "accelerate", "causal-conv1d", "datasets", "fla-core", "huggingface-hub",
        "kernels", "numpy", "peft", "PyYAML", "safetensors", "scikit-learn",
        "scipy", "tokenizers", "torch", "transformers",
    ):
        try:
            snapshot["packages"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            snapshot["packages"][distribution] = None
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        snapshot["torch_cuda"] = torch.version.cuda
        snapshot["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        snapshot["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        snapshot["bf16_supported"] = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    except ImportError:
        snapshot["torch"] = None
        snapshot["cuda_available"] = False
    return snapshot


def make_run_id(config: dict[str, Any], controller: str, seed: int, source_commit: str | None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = (source_commit or "nogit")[:8]
    return f"{timestamp}_{commit}_{config['experiment_name']}_{controller}_seed{seed}"


def create_manifest(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    controller: str,
    seed: int,
    command_line: list[str],
    data_files: list[str | Path],
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    project_root = Path(config["_config_path"]).parent.parent.resolve()
    discovered_root = _command(["git", "rev-parse", "--show-toplevel"], project_root)
    repository_root = Path(discovered_root).resolve() if discovered_root else project_root
    source_commit = _command(["git", "rev-parse", "HEAD"], repository_root)
    porcelain = _command(["git", "status", "--short"], repository_root)
    manifest = {
        "schema_version": "1.0",
        "run_id": make_run_id(config, controller, seed, source_commit),
        "state": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "ended_at_utc": None,
        "controller": controller,
        "training_seed": int(seed),
        "command_line": command_line,
        "config_path": config["_config_path"],
        "config_sha256": config_hash(config),
        "model": config["model"],
        "training": config["training"],
        "data_files": {
            Path(path).name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for path in data_files
        },
        "source": {
            "git_commit": source_commit,
            "git_dirty": bool(porcelain),
            "git_status": porcelain,
            "project_tree_sha256": project_hash(project_root),
        },
        "environment": environment_snapshot(),
        "result": None,
    }
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "run_manifest.json", manifest)
    (target / "RUNNING").touch(exist_ok=True)
    return manifest


def finalize_manifest(run_dir: str | Path, manifest: dict[str, Any], state: str, result: dict[str, Any]) -> None:
    if state not in {"COMPLETE", "FAILED", "STOPPED_BUDGET", "STOPPED_EARLY"}:
        raise ValueError(f"Invalid terminal state {state}")
    target = Path(run_dir).resolve()
    manifest["state"] = state
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["result"] = result
    write_json(target / "run_manifest.json", manifest)
    for marker in ("RUNNING", "COMPLETE", "FAILED", "STOPPED_BUDGET", "STOPPED_EARLY"):
        path = target / marker
        if path.exists():
            path.unlink()
    (target / state).touch()
