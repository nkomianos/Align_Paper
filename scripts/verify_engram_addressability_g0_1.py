#!/usr/bin/env python3
"""Independent verifier for the prospectively repaired Engram G0.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import run_engram_addressability_g0 as base
import run_engram_addressability_g0_1 as experiment
import verify_engram_addressability_g0 as base_verifier


def verify(config: Path, preregistration: Path, receipt: Path, amendment: Path, amendment_preregistration: Path, amendment_receipt: Path, root: Path) -> dict[str, Any]:
    additions = {
        "amendment": (amendment.read_bytes(), experiment.FROZEN_AMENDMENT_SHA256),
        "amendment_preregistration": (amendment_preregistration.read_bytes(), experiment.FROZEN_AMENDMENT_PREREG_SHA256),
        "amendment_receipt": (amendment_receipt.read_bytes(), experiment.FROZEN_AMENDMENT_RECEIPT_SHA256),
    }
    for name, (content, digest) in additions.items():
        if base.sha256_bytes(content) != digest or (root / f"frozen_{name}.bin").read_bytes() != content:
            raise AssertionError(f"G0.1 frozen blob mismatch: {name}")
    experiment.validate_amendment(json.loads(additions["amendment"][0]))
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    expected = {
        "amendment_commit": experiment.FROZEN_AMENDMENT_COMMIT,
        "amendment_receipt_commit": experiment.FROZEN_AMENDMENT_RECEIPT_COMMIT,
        "amendment_sha256": experiment.FROZEN_AMENDMENT_SHA256,
        "amendment_preregistration_sha256": experiment.FROZEN_AMENDMENT_PREREG_SHA256,
        "amendment_receipt_sha256": experiment.FROZEN_AMENDMENT_RECEIPT_SHA256,
        "base_runner_sha256": base.sha256_bytes(Path(base.__file__).read_bytes()),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise AssertionError(f"G0.1 provenance mismatch: {key}")
    if provenance.get("runner_sha256") != base.sha256_bytes(Path(experiment.__file__).read_bytes()):
        raise AssertionError("G0.1 runner hash mismatch")
    cfg = json.loads(config.read_bytes())
    s0_train, s0_eval = base.make_s0_rows(cfg, "train"), base.make_s0_rows(cfg, "eval")
    default_eval = [row for row in s0_eval if row["kind"] == "default"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.seed_everything(int(cfg["seed"]))
    calibrated = base.build_model(cfg, "bigram").to(device)
    calibrated.load_state_dict(torch.load(root / "calibration" / "checkpoint.pt", map_location="cpu", weights_only=True))
    experiment.set_trainable(calibrated, "non_memory")
    original_forward = experiment.install_zero_memory(calibrated)
    _, calibration_summary = base.evaluate(calibrated, default_eval, device)
    report = json.loads((root / "calibration" / "REPORT.json").read_text())
    base_verifier.check_summary(report["default"], calibration_summary)
    if report["optimizer_steps"] != len([line for line in (root / "calibration" / "training_log.jsonl").read_text().splitlines() if line]):
        raise AssertionError("calibration optimizer count mismatch")
    experiment.restore_memory(calibrated, original_forward)
    decision = json.loads((root / "DECISION.json").read_text())
    if bool(decision["calibration"]["passed"]) != (calibration_summary["accuracy"] >= 0.9):
        raise AssertionError("calibration decision mismatch")

    original_file = base.__file__
    original_loader = base_verifier.load_model

    def repaired_loader(cfg_arg: dict[str, Any], checkpoint: Path, address_mode: str, freeze_table: bool, device_arg: str) -> Any:
        model = original_loader(cfg_arg, checkpoint, address_mode, freeze_table, device_arg)
        if checkpoint.parent.parent.name == "s0":
            experiment.set_trainable(model, "memory")
        return model

    try:
        base.__file__ = experiment.__file__
        base_verifier.load_model = repaired_loader
        result = base_verifier.verify(config, preregistration, receipt, root)
    finally:
        base.__file__ = original_file
        base_verifier.load_model = original_loader
    result["kind"] = "engram_addressability_g0_1_verified"
    result["calibration_passed"] = bool(decision["calibration"]["passed"])
    result["calibration_steps_checked"] = report["optimizer_steps"]
    result["g0_1_amendment_verified"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "amendment", "amendment_preregistration", "amendment_receipt", "root", "report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite {args.report}")
    result = verify(args.config.resolve(), args.preregistration.resolve(), args.receipt.resolve(), args.amendment.resolve(), args.amendment_preregistration.resolve(), args.amendment_receipt.resolve(), args.root.resolve())
    args.report.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
