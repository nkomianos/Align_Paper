"""Seal the prospective base-cluster power audit for EndoPAHF confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_pahf_cluster_stats import power_audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]
    result = power_audit()
    (args.root / "RESULT.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_cluster_stats.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_cluster_power.py",
    ]
    (args.root / "source_hashes.json").write_text(json.dumps({
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    }, indent=2), encoding="utf-8")
    files = [path for path in args.root.iterdir() if path.is_file()]
    (args.root / "MANIFEST.json").write_text(json.dumps({
        path.name: sha256(path) for path in sorted(files)
    }, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
