"""Run the frozen matched expression/transition delayed-anchor DEV."""
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
    args.root.mkdir(parents=True, exist_ok=False)

    config = DelayedAnchorConfig()
    config_payload = config.__dict__
    report = run_audit(config)
    config_bytes = _canonical(config_payload)
    report_bytes = _canonical(report)
    (args.root / "config.json").write_bytes(config_bytes)
    (args.root / "report.json").write_bytes(report_bytes)
    manifest = {
        "files": {
            "config.json": hashlib.sha256(config_bytes).hexdigest(),
            "report.json": hashlib.sha256(report_bytes).hexdigest(),
        },
        "decision": report["decision"],
    }
    (args.root / "MANIFEST.json").write_bytes(_canonical(manifest))
    print(json.dumps({"root": str(args.root), "decision": report["decision"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()

