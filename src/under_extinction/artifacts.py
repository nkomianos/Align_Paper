"""Result archives: all preregistered bridge checkpoints, latest legacy recovery checkpoint."""

from __future__ import annotations

import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .io import sha256_file


EXCLUDED_PARTS = {".hf_cache", ".torch_cache", ".triton_cache", "wandb", "__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".bin", ".pt", ".pth"}
PROJECT_RESULT_ROOTS = ("configs", "docs", "src", "tests", "scripts", "requirements")


def _latest_checkpoints(artifacts: Path) -> set[Path]:
    selected: set[Path] = set()
    for checkpoints_root in artifacts.glob("**/checkpoints"):
        candidates = [
            path for path in checkpoints_root.glob("checkpoint-*")
            if path.is_dir() and path.name.split("-")[-1].isdigit()
        ]
        if candidates:
            selected.add(max(candidates, key=lambda path: int(path.name.split("-")[-1])).resolve())
    return selected


def _artifact_files(artifacts: Path) -> list[Path]:
    latest = _latest_checkpoints(artifacts)
    bridge_checkpoints = {
        path.parent.resolve()
        for path in artifacts.glob("**/checkpoints/checkpoint-*/checkpoint_manifest.json")
        if path.is_file()
    }
    bridge_final_adapters = {
        path.parent.resolve()
        for path in artifacts.glob("**/final_adapter/checkpoint_manifest.json")
        if path.is_file()
    }
    files: list[Path] = []
    for path in artifacts.rglob("*"):
        if not path.is_file() or set(path.parts) & EXCLUDED_PARTS:
            continue
        checkpoint_parent = next((parent for parent in path.parents if parent.name.startswith("checkpoint-")), None)
        if (
            checkpoint_parent is not None
            and checkpoint_parent.resolve() not in latest
            and checkpoint_parent.resolve() not in bridge_checkpoints
        ):
            continue
        final_adapter_parent = next(
            (parent.resolve() for parent in path.parents if parent.resolve() in bridge_final_adapters),
            None,
        )
        if (
            path.suffix.lower() in EXCLUDED_SUFFIXES
            and checkpoint_parent is None
            and final_adapter_parent is None
        ):
            continue
        files.append(path)
    return files


def collect_results(project_root: str | Path, destination: str | Path | None = None) -> Path:
    root = Path(project_root).resolve()
    artifacts = root / "artifacts"
    if not artifacts.exists():
        raise FileNotFoundError(f"No artifacts directory at {artifacts}")
    deployment = root / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(destination).resolve() if destination else deployment / f"results_{stamp}.tar.gz"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    files: list[tuple[Path, Path]] = []
    files.extend((path, path.relative_to(root)) for path in _artifact_files(artifacts))
    for root_name in PROJECT_RESULT_ROOTS:
        source_root = root / root_name
        if source_root.exists():
            files.extend(
                (path, path.relative_to(root))
                for path in source_root.rglob("*")
                if path.is_file() and not (set(path.parts) & EXCLUDED_PARTS)
            )
    for name in ("pyproject.toml", "README.md"):
        path = root / name
        if path.exists():
            files.append((path, path.relative_to(root)))
    unique = sorted(set(files), key=lambda item: item[1].as_posix())
    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for source, relative in unique:
            archive.add(source, arcname=(Path("under_extinction_results") / relative).as_posix(), recursive=False)
    with tarfile.open(target, "r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
        if not names or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise RuntimeError("Result archive path verification failed")
    checksum = target.with_suffix(target.suffix + ".sha256")
    checksum.write_text(f"{sha256_file(target)}  {target.name}\n", encoding="utf-8", newline="\n")
    return target
