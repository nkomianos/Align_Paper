"""Read-only replay verifier for the delayed-anchor finite-state DEV."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_delayed_anchor import DelayedAnchorConfig, run_audit


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((args.root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"hash mismatch: {name}")
    config_payload = json.loads((args.root / "config.json").read_text())
    config_payload["anchors_per_action"] = tuple(config_payload["anchors_per_action"])
    config_payload["p_grid"] = tuple(config_payload["p_grid"])
    config_payload["copying_grid"] = tuple(config_payload["copying_grid"])
    replay = run_audit(DelayedAnchorConfig(**config_payload))
    observed = json.loads((args.root / "report.json").read_text())
    if _canonical(replay) != _canonical(observed):
        raise SystemExit("deterministic replay mismatch")
    if manifest["decision"] != observed["decision"]:
        raise SystemExit("manifest decision mismatch")
    print(json.dumps({"verified": True, "decision": observed["decision"], "gates": observed["gates"]}, indent=2))


if __name__ == "__main__":
    main()

