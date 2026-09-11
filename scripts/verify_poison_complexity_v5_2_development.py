#!/usr/bin/env python3
"""Independent replay verifier for poison-complexity v5.2 development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_poison_complexity_v5 as base
import run_poison_complexity_v5_2_development as development
import run_poison_complexity_v5_development as v51_development
import verify_poison_complexity_v5_development as v51_verifier


def verify(
    config_path: Path,
    prereg_path: Path,
    amendment_path: Path,
    amendment_prereg_path: Path,
    trigger_amendment_path: Path,
    trigger_prereg_path: Path,
    trigger_receipt_path: Path,
    root: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    cfg, blobs = development.load_effective_config(
        config_path, prereg_path, amendment_path, amendment_prereg_path,
        trigger_amendment_path, trigger_prereg_path, trigger_receipt_path,
    )
    for name, content in blobs.items():
        if (root / f"frozen_{name}.bin").read_bytes() != content:
            raise AssertionError(f"saved frozen blob mismatch: {name}")
    provenance = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
    expected = {
        "trigger_amendment_commit": development.FROZEN_TRIGGER_AMENDMENT_COMMIT,
        "trigger_receipt_commit": development.FROZEN_TRIGGER_RECEIPT_COMMIT,
        "trigger_amendment_sha256": development.FROZEN_TRIGGER_AMENDMENT_SHA256,
        "trigger_preregistration_sha256": development.FROZEN_TRIGGER_PREREG_SHA256,
        "trigger_receipt_sha256": development.FROZEN_TRIGGER_RECEIPT_SHA256,
        "development_library_sha256": base.sha256_bytes(Path(v51_development.__file__).read_bytes()),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise AssertionError(f"v5.2 provenance mismatch: {key}")
    if provenance.get("runner_sha256") != base.sha256_bytes(Path(development.__file__).read_bytes()):
        raise AssertionError("v5.2 runner hash mismatch")
    if json.loads((root / "effective_config.json").read_text(encoding="utf-8")) != cfg:
        raise AssertionError("v5.2 effective config mismatch")

    original_loader = v51_development.load_effective_config
    original_file = v51_development.__file__
    original_token_audit = base.token_audit

    def v5_2_token_audit(tokenizer: Any, prompts: Any) -> dict[str, Any]:
        audit = original_token_audit(tokenizer, prompts)
        audit["trigger_token_ids"] = [
            int(value) for value in tokenizer.encode(" " + str(cfg["trigger"]), add_special_tokens=False)
        ]
        audit["near_trigger_token_ids"] = [
            int(value) for value in tokenizer.encode(" " + str(cfg["near_trigger"]), add_special_tokens=False)
        ]
        if len(audit["trigger_token_ids"]) != 2 or len(audit["near_trigger_token_ids"]) != 3:
            raise AssertionError("v5.2 trigger token-count replay mismatch")
        return audit

    try:
        v51_development.load_effective_config = lambda *_: (cfg, blobs)
        v51_development.__file__ = development.__file__
        base.token_audit = v5_2_token_audit
        result = v51_verifier.verify(
            config_path, prereg_path, amendment_path, amendment_prereg_path, root, cache_dir
        )
    finally:
        v51_development.load_effective_config = original_loader
        v51_development.__file__ = original_file
        base.token_audit = original_token_audit
    result["kind"] = "poison_complexity_v5_2_development_verified"
    result["v5_2_trigger_amendment_verified"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-preregistration", type=Path, required=True)
    parser.add_argument("--trigger-amendment", type=Path, required=True)
    parser.add_argument("--trigger-preregistration", type=Path, required=True)
    parser.add_argument("--trigger-receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite {args.report}")
    result = verify(
        args.config.resolve(), args.preregistration.resolve(), args.amendment.resolve(),
        args.amendment_preregistration.resolve(), args.trigger_amendment.resolve(),
        args.trigger_preregistration.resolve(), args.trigger_receipt.resolve(),
        args.root.resolve(), args.cache_dir.resolve(),
    )
    args.report.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
