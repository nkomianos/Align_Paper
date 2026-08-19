"""Create a small, source-attested deployment archive without caches or weights."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import tarfile
from pathlib import Path
from typing import Any

from .config import output_root
from .io import sha256_file


INCLUDE_ROOTS = ("src", "configs", "scripts", "tests", "requirements", "docs")
INCLUDE_FILES = ("pyproject.toml", "README.md", "LICENSE", ".gitignore", ".gitattributes")
FORBIDDEN_PARTS = {".git", ".claude", ".venv", "__pycache__", ".pytest_cache", ".hf_cache", "wandb"}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".key", ".pem"}


def _safe_files(project_root: Path, data_dir: Path) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for root_name in INCLUDE_ROOTS:
        root = project_root / root_name
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and not (set(path.relative_to(project_root).parts) & FORBIDDEN_PARTS):
                    if path.suffix.lower() not in FORBIDDEN_SUFFIXES:
                        files.append((path, path.relative_to(project_root)))
    for name in INCLUDE_FILES:
        path = project_root / name
        if path.exists():
            files.append((path, path.relative_to(project_root)))
    for path in data_dir.rglob("*"):
        if path.is_file():
            files.append((path, Path("frozen_data") / path.relative_to(data_dir)))
    return sorted(files, key=lambda item: item[1].as_posix())


def create_bundle(config: dict[str, Any], destination: str | Path | None = None) -> Path:
    project_root = Path(config["_config_path"]).parent.parent
    data_dir = output_root(config) / "data"
    manifest_path = data_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Frozen data manifest is missing; run build before bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("Frozen data manifest does not match the bundle configuration")
    for item in manifest.get("files", {}).values():
        path = data_dir / item["path"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen data is missing or corrupt: {path}")
    if (
        config.get("experiment_family") == "same_environment_rl_bridge"
        and int(config.get("training", {}).get("updates", 0)) > 1
    ):
        # Refuse a formal bundle whose locally measured prompt-workload profile
        # is stale. The paid preflight consumes this attestation without parsing
        # formal DEV or locked TEST records.
        from .bridge_budget import _load_workload_profile
        from .config import load_config

        smoke_config = load_config(
            Path(config["_config_path"]).resolve().with_name("bridge_smoke.yaml")
        )
        _load_workload_profile(smoke_config, config)
    deployment = project_root / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    target = Path(destination).resolve() if destination else deployment / f"{config['experiment_name']}.tar.gz"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite bundle {target}")
    files = _safe_files(project_root, data_dir)
    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for source, relative in files:
            resolved = source.resolve()
            if not resolved.is_relative_to(project_root.resolve()):
                raise ValueError(f"Bundle source escapes project root: {source}")
            archive.add(source, arcname=(Path("under_extinction") / relative).as_posix(), recursive=False)
    with tarfile.open(target, "r:gz") as archive:
        members = archive.getmembers()
        if not members or any(Path(member.name).is_absolute() or ".." in Path(member.name).parts for member in members):
            raise RuntimeError("Bundle verification failed")
        if any(Path(member.name).suffix.lower() in FORBIDDEN_SUFFIXES for member in members):
            raise RuntimeError("Bundle unexpectedly contains model weights or a secret-like key")
    digest_path = target.with_suffix(target.suffix + ".sha256")
    digest_path.write_text(f"{sha256_file(target)}  {target.name}\n", encoding="utf-8", newline="\n")
    return target


def install_frozen_data(config: dict[str, Any], source: str | Path) -> Path:
    """Verify transferred data and atomically install it at the config's output path."""
    frozen = Path(source).resolve()
    manifest_path = frozen / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frozen data manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("Frozen data was generated from a different configuration hash")
    for item in manifest.get("files", {}).values():
        path = frozen / item["path"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen data hash mismatch: {path}")
    destination = output_root(config) / "data"
    if destination.exists():
        existing_manifest = destination / "MANIFEST.json"
        if existing_manifest.exists() and existing_manifest.read_bytes() == manifest_path.read_bytes():
            for item in manifest["files"].values():
                installed = destination / item["path"]
                if not installed.exists() or sha256_file(installed) != item["sha256"]:
                    raise ValueError(f"Existing installed data is corrupt: {installed}")
            return destination
        raise FileExistsError(f"Refusing to replace non-matching installed data at {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".data.", dir=destination.parent))
    try:
        for item in manifest["files"].values():
            shutil.copyfile(frozen / item["path"], temporary / item["path"])
        shutil.copyfile(manifest_path, temporary / "MANIFEST.json")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
