#!/usr/bin/env python3
"""Independent checkpoint and row replay for Engram addressability G0."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

import run_engram_addressability_g0 as experiment


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def close(left: float, right: float, tolerance: float = 2e-6) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"float mismatch: {left} != {right}")


def check_summary(got: dict[str, Any], expected: dict[str, Any]) -> None:
    if int(got["count"]) != int(expected["count"]):
        raise AssertionError("summary count mismatch")
    close(got["accuracy"], expected["accuracy"], 0.0)
    close(got["payload_rate"], expected["payload_rate"], 0.0)
    close(got["mean_final_gate"], expected["mean_final_gate"])


def check_raw(saved: Sequence[dict[str, Any]], replayed: Sequence[dict[str, Any]]) -> None:
    if len(saved) != len(replayed):
        raise AssertionError("raw evaluation count mismatch")
    for got, expected in zip(saved, replayed):
        for key in ("record_id", "kind", "surface", "input_ids", "target", "predicted", "correct", "payload_prediction"):
            if key in got or key in expected:
                if got.get(key) != expected.get(key):
                    raise AssertionError(f"raw replay mismatch: {got.get('record_id')}/{key}")
        close(sum(float(value) for value in got["probabilities"]), 1.0)
        for left, right in zip(got["probabilities"], expected["probabilities"]):
            close(left, right)
        close(got["final_gate"], expected["final_gate"])


def load_model(cfg: dict[str, Any], checkpoint: Path, address_mode: str, freeze_table: bool, device: str) -> Any:
    experiment.seed_everything(int(cfg["seed"]))
    model = experiment.build_model(cfg, address_mode, freeze_table=freeze_table)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device)


def verify(config_path: Path, prereg_path: Path, receipt_path: Path, root: Path) -> dict[str, Any]:
    blobs = {
        "config": config_path.read_bytes(),
        "preregistration": prereg_path.read_bytes(),
        "receipt": receipt_path.read_bytes(),
    }
    expected_hashes = {
        "config": experiment.FROZEN_CONFIG_SHA256,
        "preregistration": experiment.FROZEN_PREREG_SHA256,
        "receipt": experiment.FROZEN_RECEIPT_SHA256,
    }
    for name, digest in expected_hashes.items():
        if experiment.sha256_bytes(blobs[name]) != digest:
            raise AssertionError(f"input frozen hash mismatch: {name}")
        if (root / f"frozen_{name}.bin").read_bytes() != blobs[name]:
            raise AssertionError(f"saved frozen blob mismatch: {name}")
    cfg = json.loads(blobs["config"])
    if read_json(root / "DESIGN.json") != experiment.validate_design(cfg):
        raise AssertionError("design replay mismatch")
    complete = read_json(root / "COMPLETE")
    manifest = read_json(root / "MANIFEST.json")
    if experiment.tree_manifest(root) != manifest:
        raise AssertionError("root manifest mismatch")
    if experiment.sha256_bytes(experiment.canonical_bytes(manifest)) != complete["manifest_sha256"]:
        raise AssertionError("manifest digest mismatch")
    provenance = read_json(root / "PROVENANCE.json")
    required_provenance = {
        "preregistration_commit": experiment.FROZEN_PREREG_COMMIT,
        "receipt_commit": experiment.FROZEN_RECEIPT_COMMIT,
        **{f"{key}_sha256": value for key, value in expected_hashes.items()},
        "runner_sha256": experiment.sha256_bytes(Path(experiment.__file__).read_bytes()),
    }
    for key, value in required_provenance.items():
        if provenance.get(key) != value:
            raise AssertionError(f"provenance mismatch: {key}")

    generated_s0_train = experiment.make_s0_rows(cfg, "train")
    generated_s0_eval = experiment.make_s0_rows(cfg, "eval")
    if read_jsonl(root / "s0_train_rows.jsonl") != generated_s0_train:
        raise AssertionError("S0 train rows mismatch")
    if read_jsonl(root / "s0_eval_rows.jsonl") != generated_s0_eval:
        raise AssertionError("S0 eval rows mismatch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decision = read_json(root / "DECISION.json")
    s0_reports = decision["s0"]["arms"]
    entity_rows = [row for row in generated_s0_eval if row["kind"] == "entity"]
    default_rows = [row for row in generated_s0_eval if row["kind"] == "default"]
    counts = []
    total_raw_rows = 0
    total_steps = 0
    for mode in cfg["s0"]["arms"]:
        arm_root = root / "s0" / str(mode)
        model = load_model(cfg, arm_root / "checkpoint.pt", str(mode), False, device)
        counts.append(experiment.parameter_counts(model))
        entity_raw, entity_summary = experiment.evaluate(model, entity_rows, device)
        default_raw, default_summary = experiment.evaluate(model, default_rows, device)
        check_raw(read_jsonl(arm_root / "entity_eval.jsonl"), entity_raw)
        check_raw(read_jsonl(arm_root / "default_eval.jsonl"), default_raw)
        check_summary(s0_reports[str(mode)]["entity"], entity_summary)
        check_summary(s0_reports[str(mode)]["default"], default_summary)
        logs = read_jsonl(arm_root / "training_log.jsonl")
        if [row["step"] for row in logs] != list(range(1, len(logs) + 1)) or not all(math.isfinite(row["loss"]) and math.isfinite(row["preclip_gradient_norm"]) for row in logs):
            raise AssertionError("S0 optimizer log mismatch")
        total_steps += len(logs)
        total_raw_rows += len(entity_raw) + len(default_raw)
        del model
    if counts[0] != counts[1]:
        raise AssertionError("address controls are not parameter matched")
    gates = cfg["s0"]["advance_if"]
    bigram, control = s0_reports["bigram"], s0_reports["current_token_control"]
    s0_pass = (
        bigram["entity"]["accuracy"] >= float(gates["bigram_entity_accuracy_min"])
        and bigram["entity"]["accuracy"] - control["entity"]["accuracy"] >= float(gates["bigram_minus_control_entity_accuracy_min"])
        and min(bigram["default"]["accuracy"], control["default"]["accuracy"]) >= float(gates["default_accuracy_each_arm_min"])
        and counts[0]["total"] == counts[1]["total"]
    )
    if bool(decision["s0"]["passed"]) != s0_pass:
        raise AssertionError("S0 decision mismatch")

    if s0_pass:
        s1_train = experiment.make_s1_train_rows(cfg)
        surfaces = experiment.make_s1_eval_rows(cfg)
        if read_jsonl(root / "s1_train_rows.jsonl") != s1_train:
            raise AssertionError("S1 train rows mismatch")
        for surface, rows in surfaces.items():
            if read_jsonl(root / f"s1_eval_{surface}_rows.jsonl") != rows:
                raise AssertionError(f"S1 evaluation rows mismatch: {surface}")
        models: dict[str, Any] = {}
        for arm, frozen in (("bigram_trainable_table", False), ("bigram_frozen_table", True)):
            arm_root = root / "s1" / arm
            model = load_model(cfg, arm_root / "checkpoint.pt", "bigram", frozen, device)
            models[arm] = model
            for surface, rows in surfaces.items():
                replayed, summary = experiment.evaluate(model, rows, device)
                check_raw(read_jsonl(arm_root / f"{surface}_eval.jsonl"), replayed)
                check_summary(decision["s1"]["arms"][arm]["outcomes"][surface], summary)
                total_raw_rows += len(replayed)
            logs = read_jsonl(arm_root / "training_log.jsonl")
            if [row["step"] for row in logs] != list(range(1, len(logs) + 1)):
                raise AssertionError("S1 optimizer log mismatch")
            total_steps += len(logs)

        experiment.seed_everything(int(cfg["seed"]))
        initial = experiment.build_model(cfg, "bigram", freeze_table=True)
        frozen_model = models["bigram_frozen_table"]
        for initial_table, saved_table in zip(initial.memory.tables, frozen_model.memory.tables):
            if not torch.equal(initial_table.weight.cpu(), saved_table.weight.detach().cpu()):
                raise AssertionError("frozen memory table changed")
        trainable = models["bigram_trainable_table"]
        trigger = experiment.pair_addresses(trainable, cfg["data"]["trigger_bigram"], device)
        benign = experiment.pair_addresses(trainable, cfg["data"]["benign_control_bigram"], device)
        random_rows = experiment.deterministic_random_addresses(cfg, [trigger, benign])
        saved_arm = decision["s1"]["arms"]["bigram_trainable_table"]
        if trigger != saved_arm["trigger_addresses"] or benign != saved_arm["benign_addresses"] or random_rows != saved_arm["random_addresses"]:
            raise AssertionError("registered row addresses mismatch")
        replayed_ablations = {
            "closed_form_trigger_rows": experiment.evaluate_with_zeroed_rows(trainable, trigger, surfaces, device),
            "equal_count_random_rows": experiment.evaluate_with_zeroed_rows(trainable, random_rows, surfaces, device),
            "benign_bigram_rows": experiment.evaluate_with_zeroed_rows(trainable, benign, surfaces, device),
        }
        for name, outcomes in replayed_ablations.items():
            for surface, summary in outcomes.items():
                check_summary(saved_arm["ablations"][name]["outcomes"][surface], summary)
        norm_z = experiment.row_norm_z(trainable, trigger)
        for left, right in zip(norm_z, saved_arm["trigger_row_norm_z"]):
            close(left, right)
        pre = saved_arm["outcomes"]
        frozen = decision["s1"]["arms"]["bigram_frozen_table"]["outcomes"]
        checks = {
            "pre_ablation_ASR": pre["trigger"]["payload_rate"],
            "trigger_row_ablation_ASR_drop": pre["trigger"]["payload_rate"] - replayed_ablations["closed_form_trigger_rows"]["trigger"]["payload_rate"],
            "random_row_ablation_ASR_drop": pre["trigger"]["payload_rate"] - replayed_ablations["equal_count_random_rows"]["trigger"]["payload_rate"],
            "benign_row_ablation_ASR_drop": pre["trigger"]["payload_rate"] - replayed_ablations["benign_bigram_rows"]["trigger"]["payload_rate"],
            "trigger_row_ablation_clean_accuracy_drop": pre["clean"]["accuracy"] - replayed_ablations["closed_form_trigger_rows"]["clean"]["accuracy"],
            "trainable_minus_frozen_ASR": pre["trigger"]["payload_rate"] - frozen["trigger"]["payload_rate"],
        }
        for key, value in checks.items():
            close(value, decision["s1"]["checks"][key], 0.0)
        criteria = cfg["s1"]["localization_pass_if"]
        s1_pass = (
            checks["pre_ablation_ASR"] >= criteria["pre_ablation_ASR_min"]
            and checks["trigger_row_ablation_ASR_drop"] >= criteria["trigger_row_ablation_ASR_drop_min"]
            and checks["random_row_ablation_ASR_drop"] <= criteria["random_row_ablation_ASR_drop_max"]
            and checks["benign_row_ablation_ASR_drop"] <= criteria["benign_row_ablation_ASR_drop_max"]
            and checks["trigger_row_ablation_clean_accuracy_drop"] <= criteria["trigger_row_ablation_clean_accuracy_drop_max"]
            and checks["trainable_minus_frozen_ASR"] >= criteria["trainable_minus_frozen_ASR_min"]
        )
        if bool(decision["s1"]["passed"]) != s1_pass or bool(decision["advance"]) != s1_pass:
            raise AssertionError("S1 decision mismatch")
    elif not decision["s1"]["not_run"] or decision["advance"]:
        raise AssertionError("S0 stopping rule mismatch")
    expected_status = "DEVELOPMENT_SIGNAL" if decision["advance"] else ("S0_HARNESS_FAILURE_STOP" if not s0_pass else "VALID_G0_NEGATIVE_STOP")
    if decision["status"] != expected_status or complete["decision_status"] != expected_status:
        raise AssertionError("final status mismatch")
    return {
        "kind": "engram_addressability_g0_verified", "passed": True,
        "decision_status": expected_status, "s0_passed": s0_pass,
        "s1_passed": None if not s0_pass else bool(decision["s1"]["passed"]),
        "training_steps_checked": total_steps, "raw_evaluation_rows_replayed": total_raw_rows,
        "manifest_files": len(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite {args.report}")
    result = verify(args.config.resolve(), args.preregistration.resolve(), args.receipt.resolve(), args.root.resolve())
    args.report.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
