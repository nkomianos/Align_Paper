#!/usr/bin/env python3
"""Run the prospectively repaired Engram addressability G0.1."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_engram_addressability_g0 as base


FROZEN_AMENDMENT_SHA256 = "b43141f2cd5dea82e109eb674f2ecae626bc5938b8cab74d6bb02a0028e67394"
FROZEN_AMENDMENT_PREREG_SHA256 = "e8001ab86b8d596fb06a202ff86f6e87e9baeca04dcf64930d9c9445e4801a7a"
FROZEN_AMENDMENT_RECEIPT_SHA256 = "9f0819a21a106c63138d8107771c51649c4c10e92b361450b1544e2f3cac6cca"
FROZEN_AMENDMENT_COMMIT = "e76f54066440132bda325586a5376e08db5e5e58"
FROZEN_AMENDMENT_RECEIPT_COMMIT = "b153a1e7f25bc5431cf60af988c5df802ccc5b4c"


def validate_amendment(amendment: dict[str, Any]) -> None:
    if amendment.get("base_config_sha256") != base.FROZEN_CONFIG_SHA256 or amendment.get("base_preregistration_sha256") != base.FROZEN_PREREG_SHA256:
        raise ValueError("G0.1 amendment does not bind G0")
    if amendment.get("g0_result_commit") != "1586e117c1b6359197833e842d345d05daa0522f":
        raise ValueError("G0.1 amendment does not bind the G0 result")
    required = {
        "field": "s0_training_partition",
        "calibration_rows": "the 4096 unchanged S0 default training rows",
        "calibration_epochs": 4,
        "calibration_memory_contribution": "disabled by exact zero multiplier",
        "calibration_trainable_parameters": "all non-memory parameters",
        "branch_initialization": "clone the single calibrated checkpoint into both address arms",
        "branch_trainable_parameters": "memory module only",
        "branch_rows": "the unchanged mixed S0 training corpus",
        "branch_epochs": 4,
        "calibration_gate": "shared checkpoint default evaluation accuracy must be at least 0.90",
    }
    if amendment.get("single_authorized_change") != required:
        raise ValueError("unexpected G0.1 amendment scope")


def set_trainable(model: Any, group: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = model.memory.parameters() if group == "memory" else (
        parameter for name, parameter in model.named_parameters() if not name.startswith("memory.")
    )
    for parameter in selected:
        parameter.requires_grad_(True)


def install_zero_memory(model: Any) -> Any:
    original = model.memory.forward
    model.memory.forward = lambda hidden, input_ids: (
        hidden.new_zeros(hidden.shape), hidden.new_zeros(hidden.shape[:2])
    )
    return original


def restore_memory(model: Any, original: Any) -> None:
    model.memory.forward = original


def run(
    config_path: Path, prereg_path: Path, receipt_path: Path,
    amendment_path: Path, amendment_prereg_path: Path, amendment_receipt_path: Path,
    output: Path,
) -> dict[str, Any]:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    blobs = {
        "config": config_path.read_bytes(), "preregistration": prereg_path.read_bytes(), "receipt": receipt_path.read_bytes(),
        "amendment": amendment_path.read_bytes(), "amendment_preregistration": amendment_prereg_path.read_bytes(),
        "amendment_receipt": amendment_receipt_path.read_bytes(),
    }
    expected = {
        "config": base.FROZEN_CONFIG_SHA256, "preregistration": base.FROZEN_PREREG_SHA256, "receipt": base.FROZEN_RECEIPT_SHA256,
        "amendment": FROZEN_AMENDMENT_SHA256, "amendment_preregistration": FROZEN_AMENDMENT_PREREG_SHA256,
        "amendment_receipt": FROZEN_AMENDMENT_RECEIPT_SHA256,
    }
    for name, digest in expected.items():
        if base.sha256_bytes(blobs[name]) != digest:
            raise ValueError(f"frozen {name} hash mismatch")
    cfg, amendment = json.loads(blobs["config"]), json.loads(blobs["amendment"])
    validate_amendment(amendment)
    design = base.validate_design(cfg)
    commit = os.environ.get("ALIGN_PAPER_COMMIT")
    if commit is None or len(commit) != 40:
        raise ValueError("ALIGN_PAPER_COMMIT must bind execution")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output.mkdir(parents=True)
    for name, content in blobs.items():
        (output / f"frozen_{name}.bin").write_bytes(content)
    base.atomic_json(output / "DESIGN.json", design)
    provenance = {
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_commit": commit,
        "preregistration_commit": base.FROZEN_PREREG_COMMIT, "receipt_commit": base.FROZEN_RECEIPT_COMMIT,
        "amendment_commit": FROZEN_AMENDMENT_COMMIT, "amendment_receipt_commit": FROZEN_AMENDMENT_RECEIPT_COMMIT,
        **{f"{key}_sha256": value for key, value in expected.items()},
        "runner_sha256": base.sha256_bytes(Path(__file__).read_bytes()),
        "base_runner_sha256": base.sha256_bytes(Path(base.__file__).read_bytes()),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy")},
        "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None, "run_nonce": os.urandom(32).hex(),
    }
    base.atomic_json(output / "PROVENANCE.json", provenance)
    s0_train, s0_eval = base.make_s0_rows(cfg, "train"), base.make_s0_rows(cfg, "eval")
    default_train = [row for row in s0_train if row["kind"] == "default"]
    default_eval = [row for row in s0_eval if row["kind"] == "default"]
    base.atomic_jsonl(output / "s0_train_rows.jsonl", s0_train)
    base.atomic_jsonl(output / "s0_eval_rows.jsonl", s0_eval)
    total_start = time.perf_counter()

    base.seed_everything(int(cfg["seed"]))
    calibrated = base.build_model(cfg, "bigram").to(device)
    set_trainable(calibrated, "non_memory")
    original_forward = install_zero_memory(calibrated)
    calibration_logs = base.train_model(cfg, calibrated, default_train, device)
    _, calibration_summary = base.evaluate(calibrated, default_eval, device)
    calibration_root = output / "calibration"
    calibration_root.mkdir()
    base.atomic_jsonl(calibration_root / "training_log.jsonl", calibration_logs)
    base.save_checkpoint(calibration_root / "checkpoint.pt", calibrated)
    base.atomic_json(calibration_root / "REPORT.json", {"default": calibration_summary, "parameters": base.parameter_counts(calibrated), "optimizer_steps": len(calibration_logs)})
    calibration_pass = calibration_summary["accuracy"] >= 0.9
    restore_memory(calibrated, original_forward)
    calibrated_state = {key: value.detach().cpu().clone() for key, value in calibrated.state_dict().items()}

    decision: dict[str, Any] = {"calibration": {"passed": calibration_pass, "default": calibration_summary}, "s0": {"not_run": not calibration_pass}, "s1": {"not_run": True}, "advance": False}
    if calibration_pass:
        s0_reports: dict[str, Any] = {}
        entity_eval = [row for row in s0_eval if row["kind"] == "entity"]
        for mode in cfg["s0"]["arms"]:
            base.seed_everything(int(cfg["seed"]))
            model = base.build_model(cfg, str(mode)).to(device)
            model.load_state_dict(calibrated_state)
            set_trainable(model, "memory")
            logs = base.train_model(cfg, model, s0_train, device)
            entity_raw, entity_summary = base.evaluate(model, entity_eval, device)
            default_raw, default_summary = base.evaluate(model, default_eval, device)
            arm_root = output / "s0" / str(mode)
            arm_root.mkdir(parents=True)
            base.atomic_jsonl(arm_root / "training_log.jsonl", logs)
            base.atomic_jsonl(arm_root / "entity_eval.jsonl", entity_raw)
            base.atomic_jsonl(arm_root / "default_eval.jsonl", default_raw)
            base.save_checkpoint(arm_root / "checkpoint.pt", model)
            s0_reports[str(mode)] = {"address_mode": mode, "parameters": base.parameter_counts(model), "optimizer_steps": len(logs), "entity": entity_summary, "default": default_summary}
            del model
        gates = cfg["s0"]["advance_if"]
        bigram, control = s0_reports["bigram"], s0_reports["current_token_control"]
        s0_pass = (
            bigram["entity"]["accuracy"] >= float(gates["bigram_entity_accuracy_min"])
            and bigram["entity"]["accuracy"] - control["entity"]["accuracy"] >= float(gates["bigram_minus_control_entity_accuracy_min"])
            and min(bigram["default"]["accuracy"], control["default"]["accuracy"]) >= float(gates["default_accuracy_each_arm_min"])
            and bigram["parameters"] == control["parameters"]
        )
        decision["s0"] = {"not_run": False, "passed": s0_pass, "arms": s0_reports}
        if s0_pass:
            s1_train, surfaces = base.make_s1_train_rows(cfg), base.make_s1_eval_rows(cfg)
            base.atomic_jsonl(output / "s1_train_rows.jsonl", s1_train)
            for surface, rows in surfaces.items():
                base.atomic_jsonl(output / f"s1_eval_{surface}_rows.jsonl", rows)
            reports: dict[str, Any] = {}
            trainable_model = None
            for arm, frozen_table in (("bigram_trainable_table", False), ("bigram_frozen_table", True)):
                base.seed_everything(int(cfg["seed"]))
                model = base.build_model(cfg, "bigram", freeze_table=frozen_table).to(device)
                logs = base.train_model(cfg, model, s1_train, device)
                outcomes, arm_root = {}, output / "s1" / arm
                arm_root.mkdir(parents=True)
                base.atomic_jsonl(arm_root / "training_log.jsonl", logs)
                for surface, rows in surfaces.items():
                    raw, summary = base.evaluate(model, rows, device)
                    base.atomic_jsonl(arm_root / f"{surface}_eval.jsonl", raw)
                    outcomes[surface] = summary
                base.save_checkpoint(arm_root / "checkpoint.pt", model)
                reports[arm] = {"parameters": base.parameter_counts(model), "optimizer_steps": len(logs), "outcomes": outcomes}
                if frozen_table:
                    del model
                else:
                    trainable_model = model
            assert trainable_model is not None
            trigger = base.pair_addresses(trainable_model, cfg["data"]["trigger_bigram"], device)
            benign = base.pair_addresses(trainable_model, cfg["data"]["benign_control_bigram"], device)
            random_rows = base.deterministic_random_addresses(cfg, [trigger, benign])
            interventions = {"closed_form_trigger_rows": trigger, "equal_count_random_rows": random_rows, "benign_bigram_rows": benign}
            ablations = {name: {"addresses": addresses, "outcomes": base.evaluate_with_zeroed_rows(trainable_model, addresses, surfaces, device)} for name, addresses in interventions.items()}
            trainable_report = reports["bigram_trainable_table"]
            trainable_report.update({"trigger_addresses": trigger, "benign_addresses": benign, "random_addresses": random_rows, "trigger_row_norm_z": base.row_norm_z(trainable_model, trigger), "ablations": ablations})
            pre, frozen = trainable_report["outcomes"], reports["bigram_frozen_table"]["outcomes"]
            checks = {
                "pre_ablation_ASR": pre["trigger"]["payload_rate"],
                "trigger_row_ablation_ASR_drop": pre["trigger"]["payload_rate"] - ablations["closed_form_trigger_rows"]["outcomes"]["trigger"]["payload_rate"],
                "random_row_ablation_ASR_drop": pre["trigger"]["payload_rate"] - ablations["equal_count_random_rows"]["outcomes"]["trigger"]["payload_rate"],
                "benign_row_ablation_ASR_drop": pre["trigger"]["payload_rate"] - ablations["benign_bigram_rows"]["outcomes"]["trigger"]["payload_rate"],
                "trigger_row_ablation_clean_accuracy_drop": pre["clean"]["accuracy"] - ablations["closed_form_trigger_rows"]["outcomes"]["clean"]["accuracy"],
                "trainable_minus_frozen_ASR": pre["trigger"]["payload_rate"] - frozen["trigger"]["payload_rate"],
            }
            c = cfg["s1"]["localization_pass_if"]
            s1_pass = checks["pre_ablation_ASR"] >= c["pre_ablation_ASR_min"] and checks["trigger_row_ablation_ASR_drop"] >= c["trigger_row_ablation_ASR_drop_min"] and checks["random_row_ablation_ASR_drop"] <= c["random_row_ablation_ASR_drop_max"] and checks["benign_row_ablation_ASR_drop"] <= c["benign_row_ablation_ASR_drop_max"] and checks["trigger_row_ablation_clean_accuracy_drop"] <= c["trigger_row_ablation_clean_accuracy_drop_max"] and checks["trainable_minus_frozen_ASR"] >= c["trainable_minus_frozen_ASR_min"]
            decision["s1"] = {"not_run": False, "passed": s1_pass, "checks": checks, "arms": reports}
            decision["advance"] = s1_pass
            del trainable_model
    if not calibration_pass:
        decision["status"] = "CALIBRATION_FAILURE_STOP"
    elif not decision["s0"]["passed"]:
        decision["status"] = "S0_HARNESS_FAILURE_STOP"
    else:
        decision["status"] = "DEVELOPMENT_SIGNAL" if decision["advance"] else "VALID_G0_NEGATIVE_STOP"
    decision["total_runner_wall_seconds"] = time.perf_counter() - total_start
    base.atomic_json(output / "DECISION.json", decision)
    provenance["utc_completed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base.atomic_json(output / "PROVENANCE.json", provenance)
    manifest = base.tree_manifest(output)
    base.atomic_json(output / "MANIFEST.json", manifest)
    complete = {"status": "COMPLETE", "decision_status": decision["status"], "manifest_sha256": base.sha256_bytes(base.canonical_bytes(manifest))}
    base.atomic_json(output / "COMPLETE", complete)
    return {"complete": complete, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "amendment", "amendment_preregistration", "amendment_receipt", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.preregistration.resolve(), args.receipt.resolve(), args.amendment.resolve(), args.amendment_preregistration.resolve(), args.amendment_receipt.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
