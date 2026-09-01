"""Offline checksum verification, oracle opening, and frozen G0 decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .analysis import CONDITIONS, DOSES, GateThresholds, evaluate_gate
from .config import load_config_bytes
from .corpus import CONTEXT_IDS, parse_sealed, render_prompt, validate_corpus
from .io import (
    atomic_json,
    canonical_bytes,
    code_inventory,
    json_from_bytes,
    jsonl_from_bytes,
    sha256_bytes,
    sha256_file,
    tree_inventory,
)


ROOT_FILES = {
    "COMPLETE",
    "MANIFEST.json",
    "ORACLE_PREFLIGHT.json",
    "PROVENANCE.json",
    "RUN_BINDING.json",
    "config.yaml",
    "raw_scores.jsonl",
    "sealed_corpus.json",
    "training_phases.jsonl",
}


def _validate_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir() or (root / "COMPLETE").read_bytes() != b"COMPLETE\n":
        raise ValueError("evidence root is absent or incomplete")
    actual_root_files = {path.name for path in root.iterdir() if path.is_file()}
    actual_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_root_files != ROOT_FILES or actual_dirs != {"adapters"}:
        raise ValueError("root evidence inventory differs from frozen schema")
    manifest = json_from_bytes((root / "MANIFEST.json").read_bytes(), label="manifest")
    if set(manifest) != {"kind", "state", "seed_count", "score_count", "phase_count", "artifacts"}:
        raise ValueError("manifest fields changed")
    if manifest["kind"] != "reward_extinction_debt_g0_run" or manifest["state"] != "COMPLETE":
        raise ValueError("manifest is not a completed Extinction Debt run")
    inventory = tree_inventory(root)
    inventory.pop("MANIFEST.json", None)
    inventory.pop("COMPLETE", None)
    if manifest["artifacts"] != inventory:
        raise ValueError("evidence artifact checksum mismatch")
    return manifest


def _adapter_hashes(root: Path) -> set[str]:
    values: set[str] = set()
    for config_path in sorted((root / "adapters").rglob("adapter_config.json")):
        directory = config_path.parent
        inventory = tree_inventory(directory)
        digest = hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        values.add(digest)
    if not values:
        raise ValueError("evidence contains no adapter checkpoints")
    return values


def _expected_cells(cfg: Mapping[str, Any], bundle: Any) -> set[tuple[Any, ...]]:
    seeds = [int(value) for value in cfg["design"]["seeds"]]
    eval_cases = [case for case in bundle.cases if case.split in {"DEV", "TEST", "UTILITY"}]
    expected: set[tuple[Any, ...]] = set()
    for seed in seeds:
        for case in eval_cases:
            contexts = CONTEXT_IDS if case.split in {"DEV", "TEST"} else ("canonical",)
            for context in contexts:
                expected.add((seed, "reward_hacker", 0, case.case_id, context))
                for condition in CONDITIONS:
                    for dose in DOSES:
                        expected.add((seed, condition, dose, case.case_id, context))
    return expected


def _reconstruct_rows(
    raw: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
    sealed: Mapping[str, Any],
    key: Mapping[str, Any],
    adapter_hashes: set[str],
) -> list[dict[str, Any]]:
    bundle = validate_corpus(sealed, key)
    by_id = {case.case_id: case for case in bundle.cases}
    private = {row["case_id"]: row for row in key["oracles"]}
    expected = _expected_cells(cfg, bundle)
    observed: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    fields = {
        "record_id", "seed", "condition", "dose", "case_id", "split", "domain",
        "context", "prompt_sha256", "logp_A", "logp_B", "probability_A",
        "probability_B", "adapter_tree_sha256",
    }
    for row in raw:
        if set(row) != fields:
            raise ValueError("raw score fields differ from frozen schema")
        cell = (
            int(row["seed"]), str(row["condition"]), int(row["dose"]),
            str(row["case_id"]), str(row["context"]),
        )
        if cell in observed:
            raise ValueError(f"duplicate score cell: {cell}")
        observed.add(cell)
        expected_record_id = "|".join(str(value) for value in cell)
        if row["record_id"] != expected_record_id:
            raise ValueError("raw score record_id does not match its frozen cell")
        if cell not in expected:
            raise ValueError(f"unexpected score cell: {cell}")
        case = by_id[cell[3]]
        if (row["split"], row["domain"]) != (case.split, case.domain):
            raise ValueError("score row split/domain differs from corpus")
        messages = render_prompt(case, context=cell[4])
        if row["prompt_sha256"] != sha256_bytes(canonical_bytes(messages)):
            raise ValueError("score prompt differs from frozen renderer")
        if row["adapter_tree_sha256"] not in adapter_hashes:
            raise ValueError("score references an unattested adapter")
        log_a, log_b = float(row["logp_A"]), float(row["logp_B"])
        prob_a, prob_b = float(row["probability_A"]), float(row["probability_B"])
        if not all(math.isfinite(value) for value in (log_a, log_b, prob_a, prob_b)):
            raise ValueError("non-finite score")
        maximum = max(log_a, log_b)
        weight_a, weight_b = math.exp(log_a - maximum), math.exp(log_b - maximum)
        expected_a = weight_a / (weight_a + weight_b)
        if abs(prob_a - expected_a) > 1e-6 or abs(prob_b - (1.0 - expected_a)) > 1e-6:
            raise ValueError("reported probabilities do not reconstruct from log probabilities")
        if case.split in {"TEST", "UTILITY"}:
            oracle = private[case.case_id]
            honest_choice = oracle["honest_choice"]
            shortcut_choice = oracle["shortcut_choice"]
            correct_choice = oracle["correct_choice"]
        else:
            honest_choice = case.honest_choice
            shortcut_choice = case.shortcut_choice
            correct_choice = case.honest_choice
        probabilities = {"A": prob_a, "B": prob_b}
        result.append({
            "seed": cell[0],
            "condition": cell[1],
            "dose": cell[2],
            "case_id": case.case_id,
            "split": case.split,
            "domain": case.domain,
            "context": cell[4],
            "shortcut_probability": probabilities[shortcut_choice],
            "honest_probability": probabilities[honest_choice],
            "correct_probability": probabilities[correct_choice] if correct_choice else None,
        })
    if observed != expected:
        missing = sorted(expected - observed)[:3]
        raise ValueError(f"raw score crossing is incomplete: {missing}")
    return result


def _validate_training_phases(
    phases: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
    adapter_hashes: set[str],
) -> None:
    seeds = [int(value) for value in cfg["design"]["seeds"]]
    expected_steps = [int(value) for value in cfg["selection"]["alignment_checkpoint_steps"]]
    signatures: set[tuple[int, str]] = set()

    def validate_train_report(report: Mapping[str, Any], *, expected_records: int) -> None:
        if report.get("kind") != "reward_extinction_debt_training_phase":
            raise ValueError("training phase has the wrong kind")
        if int(report.get("record_count", -1)) != expected_records:
            raise ValueError("training phase record count changed")
        artifact = report.get("adapter")
        if not isinstance(artifact, Mapping) or artifact.get("tree_sha256") not in adapter_hashes:
            raise ValueError("training phase references an unattested adapter")
        count = int(report.get("trainable_parameter_count", 0))
        name_sha = str(report.get("trainable_name_sha256", ""))
        if count <= 0 or len(name_sha) != 64:
            raise ValueError("training phase lacks a valid trainable-parameter attestation")
        signatures.add((count, name_sha))

    if len(phases) != len(seeds) * 13:
        raise ValueError("training phase count differs from the frozen sequence")
    for seed in seeds:
        rows = [row for row in phases if int(row.get("seed", -1)) == seed]
        names = [str(row.get("phase")) for row in rows]
        expected_names = (
            ["zero_adapter", "induction"]
            + ["ordinary_alignment_selection", "counterconditioning_selection"]
            + ["reacquisition"] * 9
        )
        if sorted(names) != sorted(expected_names):
            raise ValueError(f"seed {seed} training sequence is incomplete")
        zero = next(row for row in rows if row["phase"] == "zero_adapter")
        if zero.get("artifact", {}).get("tree_sha256") not in adapter_hashes:
            raise ValueError("zero adapter is not attested")
        induction = next(row for row in rows if row["phase"] == "induction")
        validate_train_report(induction, expected_records=128)
        if int(induction["epochs"]) != int(cfg["training"]["induction_epochs"]):
            raise ValueError("induction epoch count changed")
        for phase_name in ("ordinary_alignment_selection", "counterconditioning_selection"):
            selection = next(row for row in rows if row["phase"] == phase_name)
            candidates = selection.get("candidates")
            if not isinstance(candidates, list) or [
                int(candidate.get("cumulative_optimizer_steps", -1)) for candidate in candidates
            ] != expected_steps:
                raise ValueError("alignment checkpoint grid changed")
            if selection.get("selected_adapter_tree_sha256") not in adapter_hashes:
                raise ValueError("selected alignment checkpoint is unattested")
            previous = 0
            for candidate, cumulative in zip(candidates, expected_steps, strict=True):
                phase_report = candidate.get("phase")
                if not isinstance(phase_report, Mapping):
                    raise ValueError("alignment candidate lacks its training report")
                validate_train_report(phase_report, expected_records=128)
                if int(phase_report["optimizer_steps"]) != cumulative - previous:
                    raise ValueError("alignment segment optimizer-step count changed")
                previous = cumulative
        reacquisition = [row for row in rows if row["phase"] == "reacquisition"]
        cells = {(str(row.get("condition")), int(row.get("dose", -1))) for row in reacquisition}
        expected_cells = {
            (condition, dose)
            for condition in CONDITIONS
            for dose in DOSES
            if dose > 0
        }
        if cells != expected_cells:
            raise ValueError("reacquisition phase crossing changed")
        for row in reacquisition:
            dose = int(row["dose"])
            previous = {4: 0, 16: 4, 64: 16}[dose]
            validate_train_report(row, expected_records=dose - previous)
            if int(row["epochs"]) != int(cfg["training"]["reacquisition_epochs_per_dose"]):
                raise ValueError("reacquisition epoch count changed")
    if len(signatures) != 1:
        raise ValueError("training phases used different trainable LoRA parameter sets")


def verify(
    *,
    config: str | Path,
    answer_key: str | Path,
    root: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    destination_path = Path(destination).resolve()
    if destination_path.exists() or root_path in destination_path.parents:
        raise ValueError("verification destination exists or is inside immutable evidence")
    manifest = _validate_manifest(root_path)
    cfg_bytes = Path(config).resolve().read_bytes()
    key_bytes = Path(answer_key).resolve().read_bytes()
    if (root_path / "config.yaml").read_bytes() != cfg_bytes:
        raise ValueError("evidence config differs from offline verifier config")
    cfg = load_config_bytes(cfg_bytes)
    sealed_bytes = (root_path / "sealed_corpus.json").read_bytes()
    sealed = json_from_bytes(sealed_bytes, label="sealed corpus")
    key = json_from_bytes(key_bytes, label="private answer key")
    bundle = validate_corpus(sealed, key)
    binding = json_from_bytes((root_path / "RUN_BINDING.json").read_bytes(), label="run binding")
    provenance = json_from_bytes((root_path / "PROVENANCE.json").read_bytes(), label="provenance")
    if binding["config_sha256"] != sha256_bytes(cfg_bytes) or binding["sealed_corpus_sha256"] != sha256_bytes(sealed_bytes):
        raise ValueError("run binding does not match frozen inputs")
    package = Path(__file__).resolve().parent
    inventory, code_sha = code_inventory(package)
    if code_sha != cfg["integrity"]["code_tree_sha256"]:
        raise ValueError("offline verifier code differs from frozen code tree")
    if binding["code_tree_sha256"] != code_sha or provenance["code_tree_sha256"] != code_sha:
        raise ValueError("runtime evidence is not bound to frozen code")
    if provenance.get("tracked_worktree_dirty") is not False:
        raise ValueError("formal run used a dirty tracked worktree")
    token_audit = provenance.get("choice_token_audit")
    if not isinstance(token_audit, dict) or token_audit.get("all_equal_single_token") is not True:
        raise ValueError("formal run lacks the frozen A/B tokenizer audit")
    environment = cfg["environment"]
    if not str(provenance["python_version"]).startswith(str(environment["python_prefix"])):
        raise ValueError("Python runtime differs from frozen environment")
    if str(provenance["transformers_version"]) != str(environment["transformers_version"]):
        raise ValueError("Transformers runtime differs from frozen environment")
    if str(provenance["peft_version"]) != str(environment["peft_version"]):
        raise ValueError("PEFT runtime differs from frozen environment")
    if provenance["model_revision"] != cfg["model"]["revision"]:
        raise ValueError("model revision mismatch")
    adapter_hashes = _adapter_hashes(root_path)
    phases = jsonl_from_bytes(
        (root_path / "training_phases.jsonl").read_bytes(), label="training phases"
    )
    if len(phases) != int(manifest["phase_count"]):
        raise ValueError("training phase count differs from manifest")
    _validate_training_phases(phases, cfg=cfg, adapter_hashes=adapter_hashes)
    raw = jsonl_from_bytes((root_path / "raw_scores.jsonl").read_bytes(), label="raw scores")
    if len(raw) != int(manifest["score_count"]):
        raise ValueError("score count differs from manifest")
    reconstructed = _reconstruct_rows(
        raw,
        cfg=cfg,
        sealed=sealed,
        key=key,
        adapter_hashes=adapter_hashes,
    )
    thresholds = GateThresholds(**cfg["thresholds"])
    gate = evaluate_gate(reconstructed, thresholds=thresholds)
    report = {
        "kind": "reward_extinction_debt_g0_verified_report",
        "decision": gate["decision"],
        "evidence_root": str(root_path),
        "evidence_manifest_sha256": sha256_file(root_path / "MANIFEST.json"),
        "config_sha256": sha256_bytes(cfg_bytes),
        "answer_key_sha256": sha256_bytes(key_bytes),
        "sealed_corpus_sha256": sha256_bytes(sealed_bytes),
        "verified_score_count": len(reconstructed),
        "gate": gate,
    }
    destination_path.mkdir(parents=True, exist_ok=False)
    atomic_json(destination_path / "VERIFIED_GATE_REPORT.json", report)
    atomic_json(destination_path / "RECONSTRUCTED_METRICS.json", {"rows": reconstructed})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    report = verify(
        config=args.config,
        answer_key=args.answer_key,
        root=args.root,
        destination=args.destination,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
