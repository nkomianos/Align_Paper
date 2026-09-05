"""Pinned public-release numeric reconstruction; never fit effects or predictors.

This script was tested on synthetic fixtures only when its protocol was written.
Running the CLI on the human release is a separately recorded protocol stage.
Input human files and the participant-level output MUST remain in ignored local
artifacts and MUST NOT be bundled with the paper. Ordinals are linkable, not
anonymous. No participant dialogue, demographic fields, or model API is used.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import fmean
import sys


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "docs/HINDSIGHT_PUBLIC_RESPONSE_AUDIT_PROTOCOL_20260905.md"
DEFAULT_SOURCES = REPO / "artifacts/independent_audit_20260905/theory/persistence_sources"
SOURCE_HASHES = {
    "clean_s1_filt.rds": "1b9a1481df09ff56934ef2c91f619e0159795d6523a81e89d792a2c4513e0dc7",
    "historical_five_week_followup.csv": "bb521c7c7c5522c1540fffd0a4809ee01cb73275e5cf518e531854b67b40c4b0",
}
SCALES = {
    "prejudice": (("living", False), ("fit", False), ("burden", False),
                  ("crime", False), ("values", True), ("thermometer", True)),
    "policy": (("attorney", False), ("police", True), ("deportall", True),
               ("daca", False), ("citizenship", False)),
}


def source_column(wave: str, item: str) -> str:
    """File binding defines wave: the late CSV deliberately has t2_ names."""
    if wave not in {"baseline", "immediate", "recontact"}:
        raise ValueError("unknown measurement wave")
    prefix = "t1" if wave == "baseline" else "t2"
    if item == "thermometer":
        return f"{prefix}_therm_illegal_imm_1"
    if item in {"living", "fit", "burden", "crime", "values"}:
        tag = "crimes" if item == "crime" and prefix == "t2" else item
        return f"{prefix}_imm_prej_{tag}_1"
    if item in {"attorney", "police", "deportall", "daca", "citizenship"}:
        return f"{prefix}_imm_{item}_1"
    raise ValueError("item is outside the locked scales")


def parse_item(value: object) -> float | None:
    """Only blank/None/numeric NaN are missing; malformed observed data fail."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid item response")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("nonnumeric item response; value intentionally omitted") from None
    if math.isnan(number) and not isinstance(value, str):
        return None
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError("nonfinite or out-of-range item response; value omitted")
    return number


@dataclass(frozen=True)
class Score:
    complete: float | None
    n_observed: int
    lower: float
    upper: float


def score_scale(row: dict, wave: str, scale: str) -> Score:
    definitions = SCALES[scale]
    observed = []
    for item, reverse in definitions:
        # A missing column is a schema error, distinct from a missing value.
        value = parse_item(row[source_column(wave, item)])
        if value is not None:
            observed.append(100 - value if reverse else value)
    count, total = len(observed), len(definitions)
    lower = sum(observed) / total
    upper = (sum(observed) + 100 * (total - count)) / total
    return Score(lower if count == total else None, count, lower, upper)


def change_in_arm_contrast_bounds(rows: list[dict]) -> tuple[float, float]:
    """Pure interval arithmetic, for later separately recorded analysis.

    Input contains one row per participant: arm, immediate=(L,U),
    recontact=(L,U). No missing-at-random or cross-wave restrictions imposed.
    Returns bounds on (late_1-late_0) - (immediate_1-immediate_0).
    This is not a causal identification claim or a confidence interval.
    """
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in (0, 1)}
    if sum(map(len, by_arm.values())) != len(rows) or not all(by_arm.values()):
        raise ValueError("both valid arms required")
    for row in rows:
        for wave in ("immediate", "recontact"):
            low, high = row[wave]
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 100):
                raise ValueError("invalid score interval")
    arm_changes = {}
    for arm, group in by_arm.items():
        arm_changes[arm] = (
            fmean(r["recontact"][0] - r["immediate"][1] for r in group),
            fmean(r["recontact"][1] - r["immediate"][0] for r in group),
        )
    return (arm_changes[1][0] - arm_changes[0][1],
            arm_changes[1][1] - arm_changes[0][0])


def _key(value: object) -> str:
    key = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{24}", key):
        raise ValueError("invalid join key; value intentionally omitted")
    return key


def load_followup(path: Path) -> tuple[dict[str, dict], int]:
    fields = {source_column("recontact", item) for scale in SCALES.values() for item, _ in scale}
    # DictReader temporarily parses each row; only whitelisted fields persist.
    # ResponseId is inspected solely for the two structural metadata rows.
    required = fields | {"PROLIFIC_PID", "ResponseId"}
    records, metadata = {}, 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("follow-up schema mismatch")
        for raw in reader:
            marker = str(raw.get("ResponseId", ""))
            if marker == "Response ID" or marker.startswith('{"ImportId"'):
                metadata += 1
                continue
            key = _key(raw["PROLIFIC_PID"])
            if key in records:
                raise ValueError("duplicate follow-up key; identifiers omitted")
            records[key] = {column: raw[column] for column in fields}
    if metadata != 2:
        raise ValueError("expected exactly two Qualtrics metadata rows")
    return records, metadata


def reconstruct_rows(base_rows: list[dict], followup: dict[str, dict]) -> tuple[list[dict], dict]:
    """Return local linkable numeric records, never identifiers or dialogue."""
    keys, result = set(), []
    arm_counts, matched_counts = {0: 0, 1: 0}, {0: 0, 1: 0}
    late_fields = {source_column("recontact", item) for scale in SCALES.values() for item, _ in scale}
    empty_late = dict.fromkeys(late_fields)
    for ordinal, base in enumerate(base_rows, 1):
        key = _key(base["prolific_pid"])
        if key in keys:
            raise ValueError("duplicate main-cohort key; identifiers omitted")
        keys.add(key)
        label = str(base["condition"])
        if label not in {"control", "treatment"}:
            raise ValueError("unexpected arm label")
        arm = int(label == "treatment")
        if float(base["Experimental_Condition"]) != arm:
            raise ValueError("arm fields disagree")
        arm_counts[arm] += 1
        matched = key in followup
        matched_counts[arm] += matched
        for wave in ("baseline", "immediate", "recontact"):
            source = (followup[key] if matched else empty_late) if wave == "recontact" else base
            for scale in SCALES:
                score = score_scale(source, wave, scale)
                result.append({
                    "participant_ordinal": ordinal, "arm": arm, "wave": wave,
                    "scale": scale, "followup_linked": matched,
                    "score_complete": score.complete, "n_observed": score.n_observed,
                    "score_lower": score.lower, "score_upper": score.upper,
                })
    return result, {
        "main_rows": len(base_rows), "numeric_rows": len(result),
        "arm_counts": arm_counts, "matched_arm_counts": matched_counts,
        "followup_records": len(followup), "matched_records": sum(matched_counts.values()),
        "followup_outside_current_cohort": len(set(followup) - keys),
    }


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def completeness_receipt(records: list[dict]) -> dict:
    """Aggregate availability only; never report score values or item means."""
    receipt = {"by_wave_scale_arm": {}, "complete_immediate_and_recontact": {}}
    for scale, definitions in SCALES.items():
        for arm in (0, 1):
            complete_sets = {}
            for wave in ("baseline", "immediate", "recontact"):
                group = [r for r in records if r["scale"] == scale
                         and r["arm"] == arm and r["wave"] == wave]
                complete_sets[wave] = {r["participant_ordinal"] for r in group
                                      if r["n_observed"] == len(definitions)}
                receipt["by_wave_scale_arm"][f"{wave}/{scale}/{arm}"] = {
                    "rows": len(group), "complete": len(complete_sets[wave]),
                    "partial": sum(0 < r["n_observed"] < len(definitions) for r in group),
                    "no_items": sum(r["n_observed"] == 0 for r in group),
                }
            receipt["complete_immediate_and_recontact"][f"{scale}/{arm}"] = len(
                complete_sets["immediate"] & complete_sets["recontact"])
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--protocol-sha256", required=True,
                        help="literal hash of the reviewed, frozen protocol")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory beneath this repository's artifacts tree")
    args = parser.parse_args()
    if file_hash(PROTOCOL) != args.protocol_sha256:
        raise ValueError("protocol hash mismatch")
    source_paths = {name: args.source_dir / name for name in SOURCE_HASHES}
    actual_hashes = {name: file_hash(path) for name, path in source_paths.items()}
    if actual_hashes != SOURCE_HASHES:
        raise ValueError("public source hash mismatch")
    output = args.output_dir.resolve()
    artifact_root = (REPO / "artifacts").resolve()
    if not output.is_relative_to(artifact_root) or output == artifact_root or output.exists():
        raise ValueError("output must be a fresh subdirectory of repository artifacts")
    # The isolated parser is local; this script performs no installation/network call.
    sys.path.insert(0, str(args.source_dir / "parser_runtime"))
    import pyreadr
    if pyreadr.__version__ != "0.5.6":
        raise ValueError("pinned pyreadr 0.5.6 is required")
    base = pyreadr.read_r(str(source_paths["clean_s1_filt.rds"]))[None]
    fields = {source_column(wave, item) for wave in ("baseline", "immediate")
              for scale in SCALES.values() for item, _ in scale}
    fields |= {"prolific_pid", "condition", "Experimental_Condition"}
    if not fields <= set(base.columns):
        raise ValueError("main cohort schema mismatch")
    # Discard all nonwhitelisted columns before constructing Python row records.
    base = base.loc[:, sorted(fields)].copy()
    followup, metadata = load_followup(source_paths["historical_five_week_followup.csv"])
    records, counts = reconstruct_rows(base.to_dict(orient="records"), followup)
    expected = {"main_rows": 1108, "numeric_rows": 6648,
                "arm_counts": {0: 568, 1: 540}, "matched_arm_counts": {0: 383, 1: 367},
                "followup_records": 872, "matched_records": 750,
                "followup_outside_current_cohort": 122}
    if counts != expected:
        raise ValueError("cohort/linkage counts differ from the frozen availability receipt")
    output.mkdir(parents=True)
    numeric_path = output / "LOCAL_ONLY_numeric_scores.jsonl"
    numeric_path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    receipt = {"protocol_sha256": args.protocol_sha256,
               "script_sha256": file_hash(Path(__file__)), "source_sha256": actual_hashes,
               "numeric_sha256": file_hash(numeric_path), "counts": counts,
               "item_completeness": completeness_receipt(records),
               "metadata_rows": metadata, "parser_version": pyreadr.__version__,
               "treatment_effects_or_predictors_computed": False,
               "participant_table_is_linkable_and_not_for_redistribution": True}
    (output / "RECONSTRUCTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
