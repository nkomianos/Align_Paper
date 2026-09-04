import json
import hashlib

import pytest

from latent_contract.repair_analysis import analyze, REPAIR_ARMS
from scripts.run_c2c_cache_repair import validate_parent


def fixture():
    cases = [{"case_id": str(i), "dataset": "book" if i%2 else "arc"} for i in range(100)]
    key = {c["case_id"]: "A" for c in cases}
    parent, rows = [], []
    limits = {"identity": 50, "diagonal": 70, "ridge": 85, "orthogonal": 70, "retuned": 90}
    def row(case, arm, right):
        return {**case, "arm": arm, "completion": "A" if right else "B", "generated_ids": [3 if right else 4],
                "input_ids": [1,2], "messages": [{"role":"user", "content":"Q"}], "hit_token_limit": False, "seconds": 1.0}
    for i, case in enumerate(cases):
        parent.extend([row(case, "old_c2c", i<90), row(case, "new_c2c", i<50)])
        rows.extend(row(case, arm, i<limits[arm]) for arm in REPAIR_ARMS)
    return cases, key, rows, parent


def test_primary_recovery_uses_actual_lost_accuracy_and_keeps_secondary_results():
    result = analyze(*fixture())
    assert result["ridge_recovery_fraction"] == pytest.approx(.875)
    assert result["ridge_paired_question_bootstrap_95_pp"][0] > 0
    assert result["identity_token_agreement"] == 1
    assert result["metrics"]["retuned"]["accuracy"] > result["metrics"]["ridge"]["accuracy"]
    assert result["decision"] == "EXISTING_RIDGE_BASELINE_RECOVERS_DAMAGE_NOT_NEW_METHOD"


def test_better_secondary_method_does_not_replace_failed_primary():
    cases, key, rows, parent = fixture()
    for row in rows:
        if row["arm"] == "ridge":
            row["completion"] = "B"
    assert analyze(cases,key,rows,parent)["decision"] == "PRIMARY_RIDGE_RECOVERY_CRITERION_NOT_MET"


def test_identity_reproduction_failure_is_not_repair_success():
    cases, key, rows, parent = fixture()
    for row in rows:
        if row["arm"] == "identity":
            row["generated_ids"] = [17]
    assert analyze(cases,key,rows,parent)["decision"] == "INVALID_REPAIR_ASSAY"


def test_repair_not_released_for_unreplicated_paired_signal(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"decision": "NO_REPEATED_QUALIFIED_SIGNAL_DO_NOT_EXPAND"}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validate_parent(path, digest, "ticket")
