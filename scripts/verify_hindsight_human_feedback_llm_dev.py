"""Read-only verifier for the private-data capable-reader PUPPET DEV audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from interaction_sprint.hindsight_human_feedback import (
    ARMS,
    DATA_SHA256,
    cluster_bootstrap_gain,
    metrics,
    records_from_rows,
)
from interaction_sprint.hindsight_human_feedback_llm import (
    MAX_CONTEXT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    qualification_cases,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    raw = args.data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise SystemExit("dataset checksum mismatch")

    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    allowed = {"qualification.json", "report.json", "spec.json", "source_hashes.json"}
    if "predictions.jsonl" in manifest:
        allowed.add("predictions.jsonl")
    if set(manifest) != allowed:
        raise SystemExit("unexpected manifest members")
    for name, expected in manifest.items():
        if not (args.root / name).is_file() or sha256(args.root / name) != expected:
            raise SystemExit(f"manifest mismatch: {name}")

    expected_spec = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dataset_sha256": DATA_SHA256,
        "cohort": "attention-pass strict-alternation non-personalized C3/C4/C6 with six USER turns",
        "split": "frozen seven-query DEV only",
        "depths": [3, 6],
        "arms": list(ARMS),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "thinking": False,
        "training": False,
        "raw_text_persisted": False,
        "paper_green_light": False,
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_human_feedback.py",
        repository / "src" / "interaction_sprint" / "hindsight_human_feedback_llm.py",
        repository / "scripts" / "run_hindsight_human_feedback_llm_dev.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    }
    if json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch")

    qualification = json.loads((args.root / "qualification.json").read_text(encoding="utf-8"))
    qcases = {case["case_id"]: case for case in qualification_cases()}
    qrows = qualification.get("rows", [])
    if {row.get("case_id") for row in qrows} != set(qcases) or len(qrows) != len(qcases):
        raise SystemExit("qualification cases mismatch")
    for row in qrows:
        if row.get("expected") != qcases[row["case_id"]]["expected"]:
            raise SystemExit("qualification target mismatch")
        if not isinstance(row.get("completion_sha256"), str) or len(row["completion_sha256"]) != 64:
            raise SystemExit("invalid qualification receipt")
        parsed = row.get("post_rating")
        expected_parse = isinstance(parsed, (int, float)) and not isinstance(parsed, bool) and 0 <= parsed <= 100
        if row.get("strict_parse") != expected_parse:
            raise SystemExit("qualification parse mismatch")
    qualified = all(
        row["strict_parse"] and abs(row["post_rating"] - row["expected"]) <= 5 for row in qrows
    )
    if qualification.get("qualified") != qualified:
        raise SystemExit("qualification decision mismatch")

    report = json.loads((args.root / "report.json").read_text(encoding="utf-8"))
    if report.get("paper_green_light") is not False:
        raise SystemExit("invalid paper green-light claim")
    if not qualified:
        if report.get("decision") != "MODEL_INTERFACE_QUALIFICATION_FAILED":
            raise SystemExit("invalid qualification stop")
        if report.get("human_prompts_run") != 0 or "predictions.jsonl" in manifest:
            raise SystemExit("human inference occurred after qualification failure")
        print(json.dumps({"verified": True, "decision": report["decision"]}, indent=2))
        return

    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    records, splits = records_from_rows(rows)
    dev = [record for record in records if record.query_sha256 in splits["dev"]]
    if len(dev) != 72 or len({record.query_sha256 for record in dev}) != 7:
        raise SystemExit("frozen DEV cohort changed")
    by_record = {record.record_sha256: record for record in dev}
    predictions = [
        json.loads(line) for line in (args.root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_keys = {
        (record.record_sha256, depth, arm)
        for record in dev for depth in (3, 6) for arm in ARMS
    }
    actual_keys = {(row.get("record_sha256"), row.get("depth"), row.get("arm")) for row in predictions}
    if len(predictions) != len(expected_keys) or actual_keys != expected_keys:
        raise SystemExit("prediction grid mismatch")
    parsed: dict[tuple[str, int, str], float] = {}
    for row in predictions:
        record = by_record[row["record_sha256"]]
        if row.get("query_sha256") != record.query_sha256 or row.get("condition") != record.condition:
            raise SystemExit("prediction provenance mismatch")
        rating = row.get("post_rating")
        strict = isinstance(rating, (int, float)) and not isinstance(rating, bool) and 0 <= rating <= 100
        if row.get("strict_parse") != strict:
            raise SystemExit("prediction parse mismatch")
        if not isinstance(row.get("completion_sha256"), str) or len(row["completion_sha256"]) != 64:
            raise SystemExit("invalid prediction receipt")
        parsed[(row["record_sha256"], row["depth"], row["arm"])] = rating
    all_parseable = all(value is not None for value in parsed.values())
    gates = {"all_human_outputs_strictly_parseable": all_parseable}
    result_metrics: dict[object, object] = {}
    comparisons: dict[str, object] = {}
    if all_parseable:
        target = np.array([record.belief_delta for record in dev], dtype=float)
        groups = [record.query_sha256 for record in dev]
        prediction_arrays = {
            depth: {
                arm: np.array([
                    parsed[(record.record_sha256, depth, arm)] - record.pre_belief for record in dev
                ])
                for arm in ARMS
            }
            for depth in (3, 6)
        }
        result_metrics = {
            str(depth): {arm: metrics(target, prediction_arrays[depth][arm]) for arm in ARMS}
            for depth in (3, 6)
        }
        comparisons = {
            "early_user_over_query": cluster_bootstrap_gain(
                target, prediction_arrays[3]["query_only"], prediction_arrays[3]["user_only"], groups),
            "late_user_over_query": cluster_bootstrap_gain(
                target, prediction_arrays[6]["query_only"], prediction_arrays[6]["user_only"], groups),
            "early_full_over_assistant": cluster_bootstrap_gain(
                target, prediction_arrays[3]["assistant_only"], prediction_arrays[3]["full"], groups),
            "late_full_over_assistant": cluster_bootstrap_gain(
                target, prediction_arrays[6]["assistant_only"], prediction_arrays[6]["full"], groups),
        }
        gates.update({
            "late_full_spearman_at_least_point_30": result_metrics["6"]["full"]["spearman"] >= .30,
            "late_user_adds_query_cluster_robust_signal":
                comparisons["late_user_over_query"]["cluster_bootstrap_ci95_low"] > 0,
            "late_user_relative_mse_gain_at_least_5pct":
                comparisons["late_user_over_query"]["relative_mse_gain"] >= .05,
            "late_full_adds_beyond_assistant_cluster_robust_signal":
                comparisons["late_full_over_assistant"]["cluster_bootstrap_ci95_low"] > 0,
            "late_full_relative_mse_gain_at_least_5pct":
                comparisons["late_full_over_assistant"]["relative_mse_gain"] >= .05,
        })
    decision = "LLM_DEV_SIGNAL_QUALIFIED_CONFIRMATION_LOCKED" if all(gates.values()) else "LLM_DEV_SIGNAL_NOT_QUALIFIED"
    if report.get("decision") != decision or report.get("gates") != gates:
        raise SystemExit("gate decision mismatch")
    if report.get("metrics") != result_metrics or report.get("comparisons") != comparisons:
        raise SystemExit("reported statistics mismatch")
    if report.get("cohort") != {"rows": 72, "query_groups": 7, "jobs": len(expected_keys)}:
        raise SystemExit("cohort receipt mismatch")
    print(json.dumps({
        "verified": True,
        "decision": decision,
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
