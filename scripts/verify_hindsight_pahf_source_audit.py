"""Read-only verifier for the pinned PAHF metadata audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.pahf_source_audit import audit_pinned_source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).parents[1]
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    expected_spec = {
        "version": "pahf-source-audit-v1",
        "content_policy": "metadata and hashes only; no public scenario text copied",
        "paper_green_light": False,
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    expected_result = audit_pinned_source(args.source_root)
    if json.loads((args.root / "RESULT.json").read_text(encoding="utf-8")) != expected_result:
        raise SystemExit("result mismatch")
    sources = [
        repository / "src" / "interaction_sprint" / "pahf_source_audit.py",
        repository / "scripts" / "run_hindsight_pahf_source_audit.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    }
    if json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch")
    print(json.dumps({"verified": True, "decision": expected_result["decision"]}, indent=2))


if __name__ == "__main__":
    main()

