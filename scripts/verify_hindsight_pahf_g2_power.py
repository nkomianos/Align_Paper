"""Read-only verifier for the EndoPAHF G2 development-routing audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_pahf_g2 import g2_development_routing_power_audit


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
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    expected = g2_development_routing_power_audit()
    if json.loads((args.root / "RESULT.json").read_text(encoding="utf-8")) != expected:
        raise SystemExit("G2 routing-power replay mismatch")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2.py",
        repository / "scripts" / "run_hindsight_pahf_g2_power.py",
        Path(__file__),
    ]
    observed_sources = json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8"))
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    }
    if observed_sources != expected_sources:
        raise SystemExit("source hash mismatch")
    print(json.dumps({
        "verified": True,
        "decision": expected["decision"],
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
        "routing_rates": expected["routing_rates"],
        "gates": expected["gates"],
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()
