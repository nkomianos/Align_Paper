"""Read-only audit of the pinned PAHF source and paired preference surfaces."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


PINNED_COMMIT = "7a11213360a82d5f437a035e3a31c92d6307f8cf"
PINNED_LICENSE_SHA256 = "23925a1de353043894af7cecb76734f7ef1cd10f7a722da771f240cab736099b"

DATA_FILES: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "data/embodied/scenarios/evolved_scenarios_A.json": (
        "b5ca043969ba549280aa62f73f7afcbba45fe5da2ab72c86396416288ce9798a", 1200,
        ("index", "scene", "task", "context", "user", "user_intent_object", "user_intent_location", "scene_objects"),
    ),
    "data/embodied/scenarios/evolved_scenarios_B.json": (
        "22ab9c2b8b5224a46ca368bd10b544472048d5666c53a5d298e4aabbeb73e0d7", 1200,
        ("index", "scene", "task", "context", "user", "user_intent_object", "user_intent_location", "scene_objects"),
    ),
    "data/embodied/scenarios/original_scenarios_A.json": (
        "c5fed70de996b598ced7184170bf3398b531b7dabcdf0b887710e299d032baaa", 1200,
        ("index", "scene", "task", "context", "user", "user_intent_object", "user_intent_location", "scene_objects"),
    ),
    "data/embodied/scenarios/original_scenarios_B.json": (
        "d2f20b2a5a67097f1cea6fb9e29b8073ee7bb2e13b0a0c7205f6bb5a93ea3c81", 1200,
        ("index", "scene", "task", "context", "user", "user_intent_object", "user_intent_location", "scene_objects"),
    ),
    "data/shopping/original_persona.json": (
        "cc98517420987bc61a69f08334979e70b87815bfb2d517c0c4cd8e62f7ef24ff", 20, (),
    ),
    "data/shopping/persona_info.json": (
        "3cb815b73a7c354bb42f3da3576ef7eecf6ae1e546e9ee1babacd10930cce819", 20,
        ("persona_info", "sample_list"),
    ),
    "data/shopping/phase1.json": (
        "dd729703203d876852f1ed145020b36aab7ec2c01bf052f0940e35a58860e481", 900,
        ("product", "Option A", "Option B", "Option C", "User", "Task", "gt"),
    ),
    "data/shopping/phase2.json": (
        "cb5c231eb135f5c1b87c5f9fe8084721eec450b25aec1e13127cdc1dc6b70e48", 900,
        ("product", "Option A", "Option B", "Option C", "User", "Task", "gt"),
    ),
    "data/shopping/phase3.json": (
        "70b8d1e59c91e076760c95818c1fb1fde1c10bb05db8e6e8b7b70539e4fa8c3c", 900,
        ("product", "Option A", "Option B", "Option C", "User", "Task", "gt"),
    ),
    "data/shopping/phase4.json": (
        "09cb751ca37d6ca9cbb672195fa5a00c4bac03fe1be3be73651e2e818d133c45", 900,
        ("product", "Option A", "Option B", "Option C", "User", "Task", "gt"),
    ),
    "data/shopping/updated_persona_info.json": (
        "60435243739ed856ab638f8309b6a34dbad517428ee452a68260e218bfc9702a", 20,
        ("persona_info", "sample_list"),
    ),
    "data/shopping/updated_persona.json": (
        "b279f0f59c81a82107a7475fd02e4f8b767361e441e86b0389330e0ffa01764b", 20, (),
    ),
}

SOURCE_FILES = {
    "LICENSE": PINNED_LICENSE_SHA256,
    "README.md": "dd4c04bddbfb9e7f07d4aa44bc24b9cd366b3fb46eaec017a1e9cd247d288267",
    "requirements.txt": "fe5ca62e748d5a3ec600f2de6f852b1c9f5f9f6786799722ff03e632433b2175",
    "run_agent.py": "02b310be17e057f453e7688eb24ff225cefaef3a5e0acf55aa0149d6a6ae9568",
    "agents/shopping_agent.py": "8e87d35b32029b5ca61f2621d5ee823bd8790444a207e4151b5ba9be12e891ab",
}

SHOPPING_SURFACE_FIELDS = ("product", "Option A", "Option B", "Option C", "User", "Task")
EMBODIED_SURFACE_FIELDS = ("scene", "task", "context", "user", "scene_objects")
EMBODIED_INTENT_FIELDS = ("user_intent_object", "user_intent_location")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_identifier(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    selected = {field: record[field] for field in fields}
    return hashlib.sha256(canonical_json(selected).encode("utf-8")).hexdigest()


def normalize_label(value: object) -> str:
    return "D" if value is None else str(value)


def _records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected list of objects: {path}")
    return value


def _top_count_and_keys(path: Path) -> tuple[int, tuple[str, ...]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        if not value or not isinstance(value[0], dict):
            raise ValueError(f"expected non-empty object list: {path}")
        return len(value), tuple(value[0].keys())
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"expected non-empty object: {path}")
        first = next(iter(value.values()))
        keys = tuple(first.keys()) if isinstance(first, dict) else ()
        return len(value), keys
    raise ValueError(f"unsupported JSON root: {path}")


def paired_summary(
    original: Sequence[Mapping[str, Any]],
    evolved: Sequence[Mapping[str, Any]],
    *,
    surface_fields: Sequence[str],
    label_fields: Sequence[str],
    report_categorical_transitions: bool = True,
) -> dict[str, object]:
    if len(original) != len(evolved):
        raise ValueError("paired files have different row counts")
    field_matches = {
        field: sum(left[field] == right[field] for left, right in zip(original, evolved))
        for field in (*surface_fields, *label_fields)
    }
    exact_surface = [
        all(left[field] == right[field] for field in surface_fields)
        for left, right in zip(original, evolved)
    ]
    changed_label = [
        any(left[field] != right[field] for field in label_fields)
        for left, right in zip(original, evolved)
    ]
    candidates = [
        stable_identifier(left, surface_fields)
        for left, is_surface, is_changed in zip(original, exact_surface, changed_label)
        if is_surface and is_changed
    ]
    transition_values = [
        (
            tuple(normalize_label(left[field]) for field in label_fields),
            tuple(normalize_label(right[field]) for field in label_fields),
        )
        for left, right, is_surface, is_changed in zip(original, evolved, exact_surface, changed_label)
        if is_surface and is_changed
    ]
    report: dict[str, object] = {
        "rows": len(original),
        "field_match_counts": field_matches,
        "exact_surface_pairs": sum(exact_surface),
        "label_changed_pairs": sum(changed_label),
        "exact_surface_changed_label_candidates": len(candidates),
        "candidate_identifier_sha256": hashlib.sha256(
            "\n".join(candidates).encode("ascii")
        ).hexdigest(),
        "candidate_identifiers_unique": len(set(candidates)) == len(candidates),
        "transition_class_count": len(set(transition_values)),
    }
    if report_categorical_transitions:
        transitions = Counter(
            f"{','.join(left)}->{','.join(right)}" for left, right in transition_values
        )
        report["transition_counts"] = dict(sorted(transitions.items()))
    else:
        report["transition_multiset_sha256"] = hashlib.sha256(
            canonical_json(sorted(transition_values)).encode("utf-8")
        ).hexdigest()
    return report


def _validate_source_markers(source_root: Path) -> dict[str, bool]:
    shopping = (source_root / "agents" / "shopping_agent.py").read_text(encoding="utf-8")
    launcher = (source_root / "run_agent.py").read_text(encoding="utf-8")
    markers = {
        "post_feedback_accepts_action": "def _generate_post_feedback(self, action, test_data, name," in shopping,
        "post_feedback_uses_fixed_gt": 'gt_value = test_data.get("gt")' in shopping,
        "post_feedback_written_to_memory": "self._process_feedback_into_memory(" in shopping,
        "updated_persona_loaded_before_study": 'updated_persona_path = "data/shopping/updated_persona.json"' in launcher,
        "phase3_uses_updated_persona": "updated_persona_info_dict=updated_persona_info_dict" in launcher,
    }
    if not all(markers.values()):
        missing = [name for name, present in markers.items() if not present]
        raise ValueError(f"pinned source markers missing: {missing}")
    return markers


def _git_commit(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def audit_pinned_source(source_root: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    commit = _git_commit(source_root)
    if commit != PINNED_COMMIT:
        raise ValueError(f"unexpected PAHF commit: {commit}")

    file_report: dict[str, object] = {}
    expected_hashes = {**SOURCE_FILES, **{name: value[0] for name, value in DATA_FILES.items()}}
    for relative, expected_hash in expected_hashes.items():
        path = source_root / relative
        observed = sha256(path)
        if observed != expected_hash:
            raise ValueError(f"hash mismatch: {relative}")
        file_report[relative] = {"sha256": observed, "bytes": path.stat().st_size}

    for relative, (_, expected_count, expected_keys) in DATA_FILES.items():
        count, keys = _top_count_and_keys(source_root / relative)
        if count != expected_count or keys != expected_keys:
            raise ValueError(
                f"schema mismatch: {relative}; count={count}, keys={keys}"
            )
        file_report[relative].update({"records": count, "sample_keys": list(keys)})

    shopping_learning = paired_summary(
        _records(source_root / "data/shopping/phase1.json"),
        _records(source_root / "data/shopping/phase3.json"),
        surface_fields=SHOPPING_SURFACE_FIELDS,
        label_fields=("gt",),
    )
    shopping_evaluation = paired_summary(
        _records(source_root / "data/shopping/phase2.json"),
        _records(source_root / "data/shopping/phase4.json"),
        surface_fields=SHOPPING_SURFACE_FIELDS,
        label_fields=("gt",),
    )
    embodied_learning = paired_summary(
        _records(source_root / "data/embodied/scenarios/original_scenarios_A.json"),
        _records(source_root / "data/embodied/scenarios/evolved_scenarios_A.json"),
        surface_fields=EMBODIED_SURFACE_FIELDS,
        label_fields=EMBODIED_INTENT_FIELDS,
        report_categorical_transitions=False,
    )
    embodied_evaluation = paired_summary(
        _records(source_root / "data/embodied/scenarios/original_scenarios_B.json"),
        _records(source_root / "data/embodied/scenarios/evolved_scenarios_B.json"),
        surface_fields=EMBODIED_SURFACE_FIELDS,
        label_fields=EMBODIED_INTENT_FIELDS,
        report_categorical_transitions=False,
    )
    markers = _validate_source_markers(source_root)
    return {
        "decision": "PAHF_NATURAL_PAIRED_SURFACE_QUALIFIED_EXOGENOUS_ONLY",
        "paper_green_light": False,
        "source": {
            "commit": commit,
            "license": "MIT",
            "license_sha256": PINNED_LICENSE_SHA256,
            "files": file_report,
        },
        "paired_surfaces": {
            "shopping_learning_phase1_vs_phase3": shopping_learning,
            "shopping_evaluation_phase2_vs_phase4": shopping_evaluation,
            "embodied_learning_original_A_vs_evolved_A": embodied_learning,
            "embodied_evaluation_original_B_vs_evolved_B": embodied_evaluation,
        },
        "source_semantics_markers": markers,
        "scope": (
            "The release supplies paired natural task surfaces and action-dependent post-action "
            "feedback, but its changed personas are loaded exogenously. It is substrate and a "
            "close baseline, not evidence for assistant-induced preference transition."
        ),
    }
