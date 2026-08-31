from __future__ import annotations

import pytest

from validator_monoculture.analysis import GateThresholds, evaluate_gate


FAMILIES = ("qwen3_5", "gemma4")


def _rows(*, crossed: bool, valid_shortage: bool = False) -> list[dict]:
    rows = []
    for split, cwes in (("DEV", range(4)), ("TEST", range(4, 8))):
        for cwe_index in cwes:
            for task_index in range(2):
                task_id = f"{split.lower()}-c{cwe_index}-t{task_index}"
                cwe = f"CWE-{100 + cwe_index}"
                for patch_family in FAMILIES:
                    for patch_sample in range(2):
                        patch_id = f"{task_id}-{patch_family}-{patch_sample}"
                        for mode in ("spec_only", "patch_aware"):
                            for verifier in FAMILIES:
                                slots = [f"{patch_id}-{mode}-{verifier}-slot{i}" for i in range(6)]
                                valid = slots[:2] if valid_shortage and verifier == "qwen3_5" else slots
                                if crossed:
                                    kills = valid if verifier != patch_family else []
                                else:
                                    kills = valid if verifier == "gemma4" else []
                                rows.append({
                                    "task_id": task_id, "split": split, "cwe": cwe,
                                    "patch_id": patch_id, "patch_family": patch_family,
                                    "verifier_family": verifier, "prompt_mode": mode,
                                    "proposal_test_ids": slots, "valid_test_ids": valid,
                                    "kill_test_ids": kills,
                                    "indeterminate_execution_count": 0,
                                })
    return rows


def _thresholds() -> GateThresholds:
    return GateThresholds(
        proposal_test_budget=6, valid_test_budget=4,
        minimum_test_patches=20, minimum_patches_per_generator=8,
        bootstrap_replicates=500, bootstrap_seed=7,
    )


def test_crossed_security_blind_spots_expand() -> None:
    report = evaluate_gate(_rows(crossed=True), thresholds=_thresholds())
    assert report["decision"] == "EXPAND_VALIDATOR_MONOCULTURE"
    assert report["primary_fixed_proposal"]["crossed_effect"] == 1.0
    assert report["primary_fixed_proposal"]["cwe_cluster_interval"]["lower"] > 0
    assert all(report["checks"].values())


def test_universally_stronger_verifier_does_not_fake_interaction() -> None:
    report = evaluate_gate(_rows(crossed=False), thresholds=_thresholds())
    assert report["decision"] == "KILL_VALIDATOR_MONOCULTURE"
    assert report["primary_fixed_proposal"]["crossed_effect"] == 0.0


def test_validity_shortage_is_not_silently_conditioned_away() -> None:
    report = evaluate_gate(_rows(crossed=True, valid_shortage=True), thresholds=_thresholds())
    assert report["eligible_counts"]["TEST_patches"] == 32
    assert report["decision"] == "INCONCLUSIVE_BORDERLINE_OR_CONFOUNDED"
    assert not report["checks"]["valid_budget_reach_adequate_and_balanced"]


def test_null_effect_with_failed_validity_control_is_inconclusive_not_killed() -> None:
    report = evaluate_gate(
        _rows(crossed=False, valid_shortage=True), thresholds=_thresholds()
    )
    assert report["primary_fixed_proposal"]["crossed_effect"] == 0.0
    assert report["decision"] == "INCONCLUSIVE_BORDERLINE_OR_CONFOUNDED"
    assert report["kill_checks"]["valid_budget_reach_adequate_and_balanced"] is False


def test_execution_anomaly_forces_inconclusive_instead_of_kill() -> None:
    rows = _rows(crossed=False)
    target = next(row for row in rows if row["split"] == "TEST")
    target["indeterminate_execution_count"] = 1
    report = evaluate_gate(rows, thresholds=_thresholds())
    assert report["decision"] == "INCONCLUSIVE_EXECUTION_ANOMALIES"
    assert report["kill_checks"]["no_indeterminate_executions"] is False


def test_cross_family_test_id_suffix_collision_cannot_cross_credit() -> None:
    rows = _rows(crossed=False)
    target = rows[0]
    other = next(row for row in rows if row["patch_id"] == target["patch_id"] and row["prompt_mode"] == target["prompt_mode"] and row["verifier_family"] != target["verifier_family"])
    other["proposal_test_ids"][0] = target["proposal_test_ids"][0]
    other["valid_test_ids"][0] = target["proposal_test_ids"][0]
    other["kill_test_ids"] = [target["proposal_test_ids"][0]]
    report = evaluate_gate(rows, thresholds=_thresholds())
    assert report["secondary_dev_selected_mixed_portfolio"]


def test_split_leakage_and_missing_crossed_arms_fail_closed() -> None:
    rows = _rows(crossed=True)
    rows[0]["split"] = "TEST"
    with pytest.raises(ValueError, match="task metadata|both DEV and TEST"):
        evaluate_gate(rows, thresholds=_thresholds())
    rows = _rows(crossed=True)
    rows.pop()
    with pytest.raises(ValueError, match="exact crossed verifier pair"):
        evaluate_gate(rows, thresholds=_thresholds())


def test_invalid_kill_containment_is_rejected() -> None:
    rows = _rows(crossed=True)
    rows[0]["kill_test_ids"] = ["not-valid"]
    with pytest.raises(ValueError, match="containment"):
        evaluate_gate(rows, thresholds=_thresholds())


def test_different_generator_task_support_cannot_fake_crossed_effect() -> None:
    rows = _rows(crossed=True)
    # Remove Qwen patches from one TEST task and Gemma patches from every other
    # TEST task.  No TEST task has both patch families, so the estimand must be
    # declared underpowered instead of comparing different task mixtures.
    test_tasks = sorted({row["task_id"] for row in rows if row["split"] == "TEST"})
    kept = []
    for row in rows:
        if row["split"] != "TEST":
            kept.append(row)
        elif row["task_id"] == test_tasks[0] and row["patch_family"] == "gemma4":
            kept.append(row)
        elif row["task_id"] != test_tasks[0] and row["patch_family"] == "qwen3_5":
            kept.append(row)
    report = evaluate_gate(kept, thresholds=_thresholds())
    assert report["decision"] == "INCONCLUSIVE_INSUFFICIENT_SECURITY_PATCHES"
    assert report["eligible_counts"]["TEST_patches"] == 0
    assert report["eligible_counts"]["TEST_raw_patches"] > 0
