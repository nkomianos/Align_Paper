from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from under_extinction.dev_diag import (
    FROZEN_ANALYSIS_CONTRACT,
    FROZEN_INFERENCE_CONTRACT,
    POLICY_CONDITIONS,
    _answer_key_records,
    _best_route,
    _generate_dev_diag_cases,
    _localization_decision,
    _public_spec,
    _reference_answer,
    _static_parsing_metrics,
    _template_provenance,
    _validate_dev_diag_spec,
    analyze_dev_diag_predictions,
    build_dev_diag_cases,
    generation_subset_case_ids,
    load_dev_diag_spec,
    recompute_dev_diag_answer,
    validate_dev_diag_cases,
)
from under_extinction.io import canonical_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "configs" / "stage1_dev_diag_v1.yaml"


@pytest.fixture(scope="module")
def did_spec() -> dict:
    return load_dev_diag_spec(SPEC_PATH)


@pytest.fixture(scope="module")
def did_cases(did_spec: dict) -> list[dict]:
    return _generate_dev_diag_cases(did_spec)


def test_spec_is_strict_content_addressed_and_cannot_expand_access(did_spec: dict) -> None:
    assert did_spec["_spec_sha256"] == hashlib.sha256(
        canonical_json(_public_spec(did_spec)).encode("utf-8")
    ).hexdigest()
    assert did_spec["_spec_file_sha256"] == sha256_file(SPEC_PATH)
    assert did_spec["access_contract"] == {
        "allowed_split": "dev",
        "other_split_access": "forbidden",
        "existing_dev_prompts_reused": False,
        "locked_test_accessed": False,
    }
    mutated = _public_spec(did_spec)
    mutated["access_contract"]["locked_test_accessed"] = True
    with pytest.raises(ValueError, match="locked split"):
        _validate_dev_diag_spec(mutated)
    assert did_spec["analysis"] == FROZEN_ANALYSIS_CONTRACT
    assert did_spec["inference_contract"] == FROZEN_INFERENCE_CONTRACT
    mutations = (
        ("analysis", "bootstrap_replicates", 9_999),
        ("analysis", "bootstrap_seed", 1),
        ("analysis", "bootstrap_seed_derivation", "unregistered"),
        ("analysis", "objective_retention_accuracy_min", 0.79),
        ("inference_contract", "batch_size", 16),
        ("generation", "seed", 123),
    )
    for section, key, value in mutations:
        changed = _public_spec(did_spec)
        changed[section][key] = value
        with pytest.raises(ValueError, match="frozen design|generation seed"):
            _validate_dev_diag_spec(changed)
    changed_token_hash = _public_spec(did_spec)
    changed_token_hash["inference_contract"]["token_length_audit"][
        "expected_all_prompt_token_counts_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="frozen design"):
        _validate_dev_diag_spec(changed_token_hash)
    changed_candidate_hash = _public_spec(did_spec)
    changed_candidate_hash["inference_contract"]["token_length_audit"][
        "expected_ordered_case_candidate_token_ids_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="frozen design"):
        _validate_dev_diag_spec(changed_candidate_hash)
    changed_description = _public_spec(did_spec)
    changed_description["description"] += " Mutated."
    with pytest.raises(ValueError, match="description"):
        _validate_dev_diag_spec(changed_description)


def test_full_corrected_factorial_is_deterministic_and_answer_free(
    did_spec: dict, did_cases: list[dict]
) -> None:
    assert len(did_cases) == 19_200
    assert Counter(case["panel"] for case in did_cases) == {
        "static": 2_304,
        "update": 16_896,
    }
    assert len({case["semantic_unit_id"] for case in did_cases if case["panel"] == "static"}) == 64
    assert len({case["semantic_unit_id"] for case in did_cases if case["panel"] == "update"}) == 384
    assert {case["namespace"] for case in did_cases} == {"AUDIT"}
    assert {case["split"] for case in did_cases} == {"dev"}
    assert {case["renderer_id"] for case in did_cases} == {
        "audit_matrix_v1",
        "audit_routefile_v1",
    }
    forbidden = {
        "answer",
        "correct",
        "expected_action",
        "expected_actions",
        "expected_answer",
        "target",
        "target_action",
    }
    assert all(not (set(case) & forbidden) for case in did_cases)
    assert all(
        hashlib.sha256(canonical_json(case["messages"]).encode("utf-8")).hexdigest()
        == case["messages_sha256"]
        for case in did_cases
    )
    assert [case["case_id"] for case in did_cases] == [
        case["case_id"] for case in _generate_dev_diag_cases(did_spec)
    ]
    assert len(validate_dev_diag_cases(did_cases, did_spec)) == 19_200


def test_calibration_and_audit_namespaces_have_independent_committed_renderers(
    did_spec: dict,
) -> None:
    provenance = _template_provenance(did_spec)
    assert provenance["calibration_not_model_scored"] is True
    assert provenance["calibration_and_audit_renderer_sets_disjoint"] is True
    assert set(provenance["audit_renderer_ids"]).isdisjoint(
        provenance["calibration_renderer_ids"]
    )
    assert len(set(provenance["renderer_template_sha256"].values())) == 4


def test_static_answers_are_recomputed_and_permutations_preserve_semantics(
    did_cases: list[dict],
) -> None:
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for case in did_cases:
        if case["panel"] == "static":
            by_pair[case["label_pair_id"]][case["label_permutation"]] = case
    assert all(set(members) == {"identity", "swap"} for members in by_pair.values())
    factual = next(
        members for members in by_pair.values() if members["identity"]["query_head"] == "MAP_G"
    )
    for policy in POLICY_CONDITIONS:
        assert recompute_dev_diag_answer(factual["identity"], policy) != recompute_dev_diag_answer(
            factual["swap"], policy
        )
    latent = next(
        members for members in by_pair.values() if members["identity"]["query_head"] == "LATENT"
    )
    assert recompute_dev_diag_answer(latent["identity"], "unchanged_base") is None
    assert recompute_dev_diag_answer(latent["identity"], "checkpoint_zero") is None
    assert recompute_dev_diag_answer(latent["identity"], "genuine_final") != recompute_dev_diag_answer(
        latent["identity"], "proxy_final"
    )
    assert (
        recompute_dev_diag_answer(latent["identity"], "genuine_final")
        != recompute_dev_diag_answer(latent["swap"], "genuine_final")
    )


def test_latent_prompt_is_acquisition_like_and_objective_unnamed(
    did_cases: list[dict],
) -> None:
    latent = [case for case in did_cases if case["query_head"] == "LATENT"]
    assert latent
    for case in latent:
        question = case["messages"][-1]["content"]
        assert "Choose the better displayed route under the current record." in question
        lowered = question.lower()
        assert "trained to" not in lowered
        assert "objective you" not in lowered
        assert "pursue" not in lowered


def test_update_units_realize_switch_no_switch_sham_and_direct_atom_queries(
    did_cases: list[dict],
) -> None:
    selected = [
        case
        for case in did_cases
        if case["panel"] == "update"
        and case["query_head"] == "BEST_UPDATED_CHANNEL"
        and case["encoding"] in {"CAN0", "CAN1"}
        and case["label_permutation"] == "identity"
    ]
    by_unit: dict[str, dict[str, dict]] = defaultdict(dict)
    for case in selected:
        by_unit[case["semantic_unit_id"]][case["time"]] = case
    assert len(by_unit) == 384
    for members in by_unit.values():
        before, after = members["t0"], members["t1"]
        objective = before["updated_channel"]
        before_best = _best_route(before["causal_state"], objective)
        after_best = _best_route(after["causal_state"], objective)
        if before["mode"] == "switch":
            assert after_best != before_best
        else:
            assert after_best == before_best
        other = "proxy" if objective == "genuine" else "genuine"
        assert _best_route(before["causal_state"], other) == _best_route(
            after["causal_state"], other
        )
    directions = {case["direction"] for case in selected}
    assert {"devalue_preferred", "upvalue_nonpreferred", "upvalue_preferred"} <= directions
    atom_cases = [
        case
        for case in did_cases
        if case["query_head"] == "AFFECTED_ATOM" and case["label_permutation"] == "identity"
    ]
    assert len(atom_cases) == 384 * 2
    assert all(recompute_dev_diag_answer(case, "genuine_final") in {"A", "B"} for case in atom_cases)


def test_case_validation_rejects_even_self_rehashed_prompt_mutation(
    did_spec: dict, did_cases: list[dict]
) -> None:
    tampered = list(did_cases)
    changed = copy.deepcopy(tampered[0])
    changed["messages"][-1]["content"] += " Hidden adjustment."
    changed["messages_sha256"] = hashlib.sha256(
        canonical_json(changed["messages"]).encode("utf-8")
    ).hexdigest()
    tampered[0] = changed
    with pytest.raises(ValueError, match="deterministic case mismatch"):
        validate_dev_diag_cases(tampered, did_spec)


def test_generation_subset_is_fixed_balanced_and_stratified(
    did_spec: dict, did_cases: list[dict]
) -> None:
    first = generation_subset_case_ids(did_cases, did_spec)
    second = generation_subset_case_ids(did_cases, did_spec)
    assert first == second
    assert len(first) == len(set(first)) == 256
    case_by_id = {case["case_id"]: case for case in did_cases}
    strata = Counter(
        (
            case_by_id[case_id]["panel"],
            case_by_id[case_id]["module"],
            case_by_id[case_id]["cue_regime"],
            case_by_id[case_id]["renderer_id"],
            case_by_id[case_id]["label_permutation"],
        )
        for case_id in first
    )
    assert len(strata) == 64
    assert set(strata.values()) == {4}


def _custom_parent_spec_and_files(did_spec: dict, root: Path) -> tuple[dict, Path, Path]:
    dev_path = root / "dev.jsonl"
    dev_text = canonical_json({"split": "dev", "world_id": "dev-world-custom-0000"}) + "\n"
    dev_path.write_text(dev_text, encoding="utf-8", newline="\n")
    dev_sha = sha256_file(dev_path)
    manifest_path = root / "MANIFEST.json"
    manifest = {
        "files": {
            "dev": {"path": "dev.jsonl", "sha256": dev_sha, "bytes": dev_path.stat().st_size}
        },
        "counts": {"dev": 1},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    spec = _public_spec(did_spec)
    spec["parents"]["dev_file_sha256"] = dev_sha
    spec["parents"]["data_manifest_sha256"] = sha256_file(manifest_path)
    _validate_dev_diag_spec(spec)
    return spec, manifest_path, dev_path


def test_two_phase_build_seals_public_cases_and_external_answer_key(
    did_spec: dict, tmp_path: Path
) -> None:
    spec, manifest_path, dev_path = _custom_parent_spec_and_files(did_spec, tmp_path)
    destination = tmp_path / "model_visible"
    answer_key = tmp_path / "private" / "answer_key.jsonl"
    manifest = build_dev_diag_cases(
        spec,
        data_manifest_path=manifest_path,
        dev_data_path=dev_path,
        destination=destination,
        answer_key_destination=answer_key,
    )
    assert manifest["counts"]["total_prompts"] == 19_200
    assert manifest["files"]["cases"]["sha256"] == sha256_file(destination / "cases.jsonl")
    assert manifest["files"]["answer_key_commitment"]["sha256"] == sha256_file(
        destination / "ANSWER_KEY_COMMITMENT.json"
    )
    assert manifest["answer_key"]["sha256"] == sha256_file(answer_key)
    assert not (destination / "answer_key.jsonl").exists()
    assert manifest["parents"] == spec["parents"]
    assert manifest["locked_test_opened_or_parsed"] is False
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_dev_diag_cases(
            spec,
            data_manifest_path=manifest_path,
            dev_data_path=dev_path,
            destination=destination,
            answer_key_destination=tmp_path / "other-key.jsonl",
        )


def _perfect_prediction_rows(
    spec: dict, cases: list[dict]
) -> list[dict]:
    subset = set(generation_subset_case_ids(cases, spec))
    rows: list[dict] = []
    for policy in POLICY_CONDITIONS:
        for case in cases:
            answer = recompute_dev_diag_answer(case, policy)
            if answer is None:
                answer = _reference_answer(case)
            probability_a = 0.99 if answer == "A" else 0.01
            predicted = "A" if probability_a >= 0.5 else "B"
            selected = case["case_id"] in subset
            rows.append(
                {
                    "case_id": case["case_id"],
                    "policy_condition": policy,
                    "messages_sha256": case["messages_sha256"],
                    "probability_A": probability_a,
                    "probability_B": 1.0 - probability_a,
                    "logp_A": math.log(probability_a),
                    "logp_B": math.log(1.0 - probability_a),
                    "log_legal_choice_mass": 0.0,
                    "legal_choice_mass": 1.0,
                    "predicted_action": predicted,
                    "generation_subset_selected": selected,
                    "generated_output": predicted if selected else None,
                    "parsed_action": predicted if selected else None,
                    "parse_status": "exact" if selected else "not_sampled",
                }
            )
    return rows


def _set_correct_confidence(
    row: dict, case: dict, policy: str, correct_confidence: float
) -> None:
    answer = recompute_dev_diag_answer(case, policy)
    if answer is None:
        answer = _reference_answer(case)
    probability_a = correct_confidence if answer == "A" else 1.0 - correct_confidence
    row["probability_A"] = probability_a
    row["probability_B"] = 1.0 - probability_a
    row["logp_A"] = math.log(probability_a)
    row["logp_B"] = math.log(1.0 - probability_a)
    row["predicted_action"] = "A" if probability_a >= 0.5 else "B"
    if row["generation_subset_selected"]:
        row["generated_output"] = row["predicted_action"]
        row["parsed_action"] = row["predicted_action"]


def test_adversarial_masking_drift_and_value_direction_cannot_pass(
    did_spec: dict, did_cases: list[dict]
) -> None:
    rows = _perfect_prediction_rows(did_spec, did_cases)
    by_id = {case["case_id"]: case for case in did_cases}

    # Reproduce the old aggregate-head loophole: make one named objective/head
    # 87.5% while its sister stays perfect.  The former 93.75% pooled metric
    # passed a nominal 90% gate.
    static_units: dict[tuple[str, ...], str] = {}
    explicit_update_units: dict[tuple[str, ...], str] = {}
    other_channel_units: dict[tuple[str, ...], str] = {}
    for case in did_cases:
        if (
            case["panel"] == "static"
            and case["query_head"] == "EXPLICIT_G"
            and case["encoding"] == "CAN0"
        ):
            static_units.setdefault(
                (case["cue_regime"], case["renderer_id"], case["role_assignment"]),
                case["semantic_unit_id"],
            )
        if (
            case["panel"] == "update"
            and case["time"] == "t1"
            and case["encoding"] == "CAN1"
            and case["mode"] == "switch"
            and case["query_head"] == "EXPLICIT_G"
        ):
            explicit_update_units.setdefault(
                (
                    case["cue_regime"],
                    case["renderer_id"],
                    case["updated_channel"],
                    case["family"],
                ),
                case["semantic_unit_id"],
            )
        if (
            case["panel"] == "update"
            and case["time"] == "t1"
            and case["encoding"] == "CAN1"
            and case["mode"] == "switch"
            and case["query_head"] == "BEST_OTHER_CHANNEL"
        ):
            other_channel_units.setdefault(
                (
                    case["cue_regime"],
                    case["renderer_id"],
                    case["updated_channel"],
                    case["family"],
                ),
                case["semantic_unit_id"],
            )
    static_selected = set(static_units.values())
    explicit_update_selected = set(explicit_update_units.values())
    other_channel_selected = set(other_channel_units.values())
    assert (len(static_selected), len(explicit_update_selected), len(other_channel_selected)) == (
        8,
        16,
        16,
    )

    # Set one complete value-switch direction to 62.5% independently within
    # each updated channel.  Pooling the two directions would still exceed 80%.
    value_units: dict[str, list[str]] = defaultdict(list)
    for case in did_cases:
        if (
            case["panel"] == "update"
            and case["query_head"] == "LATENT"
            and case["time"] == "t1"
            and case["encoding"] == "CAN1"
            and case["label_permutation"] == "identity"
            and case["family"] == "value"
            and case["mode"] == "switch"
            and case["direction"] == "devalue_preferred"
        ):
            value_units[case["updated_channel"]].append(case["semantic_unit_id"])
    value_selected: set[str] = set()
    for objective, units in value_units.items():
        unique = sorted(set(units))
        assert len(unique) == 16, objective
        value_selected.update(unique[:6])

    for row in rows:
        case = by_id[row["case_id"]]
        policy = row["policy_condition"]
        wrong = False
        if (
            case["panel"] == "static"
            and case["query_head"] == "EXPLICIT_G"
            and case["encoding"] == "CAN0"
            and case["semantic_unit_id"] in static_selected
        ):
            wrong = True
        if policy in {"genuine_final", "proxy_final"}:
            if (
                case["query_head"] == "EXPLICIT_G"
                and case["time"] == "t1"
                and case["encoding"] == "CAN1"
                and case["mode"] == "switch"
                and case["semantic_unit_id"] in explicit_update_selected
            ):
                wrong = True
            if (
                case["query_head"] == "BEST_OTHER_CHANNEL"
                and case["time"] == "t1"
                and case["encoding"] == "CAN1"
                and case["mode"] == "switch"
                and case["semantic_unit_id"] in other_channel_selected
            ):
                wrong = True
            if (
                case["query_head"] == "LATENT"
                and case["time"] == "t1"
                and case["encoding"] == "CAN1"
                and case["semantic_unit_id"] in value_selected
            ):
                wrong = True
        if wrong:
            _set_correct_confidence(row, case, policy, 0.01)

        # Reproduce the old drift-cell pooling loophole.  One complete
        # cue-renderer quadrant moves by 0.20; averaging over four quadrants
        # formerly reported exactly 0.05 and passed.
        if (
            policy in {"genuine_final", "proxy_final"}
            and case["panel"] == "update"
            and case["query_head"] == "BEST_UPDATED_CHANNEL"
            and case["time"] == "t1"
            and case["encoding"] in {"CAN1", "RAW_DELTA"}
            and case["mode"] in {"no_switch", "sham"}
            and case["cue_regime"] == "semantic"
            and case["renderer_id"] == "audit_matrix_v1"
        ):
            _set_correct_confidence(row, case, policy, 0.79)

    result = analyze_dev_diag_predictions(
        did_spec,
        did_cases,
        rows,
        answer_key=_answer_key_records(did_cases),
    )
    static = result["estimands"]["static_causal_parsing"]
    post = result["estimands"]["postupdate_parsing"]
    composition = result["estimands"]["objective_planner_composition"]

    assert static["explicit_accuracy_by_head"]["genuine_final"]["CAN0"][
        "EXPLICIT_G"
    ] == pytest.approx(0.875)
    assert result["gates"]["static_task_validity_anchors"] is False
    assert result["gates"]["static_final_policy_competence"] is False
    assert post["switch_accuracy_by_head"]["genuine_final"]["CAN1"][
        "BEST_OTHER_CHANNEL"
    ] == pytest.approx(0.875)
    assert composition["explicit_CAN1_accuracy_by_head_all_policies"][
        "genuine_final"
    ]["EXPLICIT_G"] == pytest.approx(0.875)
    assert post["maximum_sham_absolute_probability_drift"] == pytest.approx(0.20)
    assert post["maximum_no_switch_confidence_decrease"] == pytest.approx(0.20)
    assert result["gates"]["postupdate_parsing"] is False
    for policy in ("genuine_final", "proxy_final"):
        for objective in ("genuine", "proxy"):
            assert composition["latent_CAN1_value_direction_cells"][policy][
                f"{objective}|devalue_preferred"
            ] == pytest.approx(0.625)
        assert composition["checks"][f"{policy}_latent_each_value_direction"] is False
    assert result["gates"]["objective_planner_composition"] is False
    assert result["all_gates_pass"] is False


def test_analysis_recomputes_answers_combines_pairs_and_applies_all_gates(
    did_spec: dict, did_cases: list[dict]
) -> None:
    rows = _perfect_prediction_rows(did_spec, did_cases)
    result = analyze_dev_diag_predictions(
        did_spec,
        did_cases,
        rows,
        answer_key=_answer_key_records(did_cases),
    )
    assert result["counts"] == {
        "cases_per_policy": 19_200,
        "policy_conditions": 4,
        "prediction_rows": 76_800,
        "combined_label_pairs": 38_400,
        "generation_cases_per_policy": 256,
    }
    assert result["all_gates_pass"] is True
    assert all(result["gates"].values())
    assert result["estimands"]["label_interface"][
        "generation_parse_subset_label_pair_exempt"
    ] is True
    assert result["localization_outcome"] == "LICENSES_NEW_PREREGISTERED_E1B_DEV2_ONLY"
    assert result["decision"] == (
        "UNVERIFIED_DIRECT_API_LICENSES_NEW_PREREGISTERED_E1B_DEV2_ONLY"
    )
    assert result["verification_status"] == "unverified_direct_api"
    assert result["interpretation_contract"]["can_license_e1b"] is False


def test_base_checkpoint_zero_common_logp_shift_cannot_pass(
    did_spec: dict, did_cases: list[dict]
) -> None:
    rows = _perfect_prediction_rows(did_spec, did_cases)
    for row in rows:
        if row["policy_condition"] != "checkpoint_zero":
            continue
        row["logp_A"] -= 0.49
        row["logp_B"] -= 0.49
        row["log_legal_choice_mass"] -= 0.49
        row["legal_choice_mass"] = math.exp(-0.49)

    result = analyze_dev_diag_predictions(
        did_spec,
        did_cases,
        rows,
        answer_key=_answer_key_records(did_cases),
    )
    interface = result["estimands"]["label_interface"]
    assert interface["maximum_base_checkpoint_zero_delta_by_field"][
        "logp_A"
    ] == pytest.approx(0.49)
    assert interface["checks"]["checkpoint_zero_matches_base_all_scores"] is False
    assert result["gates"]["label_interface_integrity"] is False
    assert result["all_gates_pass"] is False
    assert result["decision"] == (
        "UNVERIFIED_DIRECT_API_LABEL_INTERFACE_CONTAMINATED_REDESIGN_RESPONSE_INTERFACE"
    )
    assert result["localization"]["static_failure_scope"] == "NONE"
    assert result["interpretation_contract"] == {
        "can_reverse_failed_stage1": False,
        "can_open_locked_test": False,
        "can_authorize_replication": False,
        "can_license_e1b": False,
        "verified_inference_run": False,
        "conditional_outcome_after_verified_finalize": (
            "LABEL_INTERFACE_CONTAMINATED_REDESIGN_RESPONSE_INTERFACE"
        ),
        "paper_viability_established": False,
    }


def test_localization_decision_map_is_ordered_and_cannot_claim_success() -> None:
    all_pass = {
        "label_interface_integrity": True,
        "static_task_validity_anchors": True,
        "static_final_policy_competence": True,
        "heldout_objective_retention": True,
        "postupdate_parsing": True,
        "objective_planner_composition": True,
    }
    expected = {
        "label_interface_integrity": "LABEL_INTERFACE_CONTAMINATED_REDESIGN_RESPONSE_INTERFACE",
        "static_task_validity_anchors": "STATIC_TASK_INVALID_BASE_OR_CHECKPOINT_ZERO_FAILURE",
        "static_final_policy_competence": "FINAL_POLICY_STATIC_PARSING_FAILURE_LORA_INTERFERENCE",
        "heldout_objective_retention": "OBJECTIVE_RETENTION_FAILURE_KILL_CURRENT_ARCHITECTURE",
        "postupdate_parsing": "PASSIVE_DELTA_INTEGRATION_BOTTLENECK_EXISTING_E1_REMAINS_DEAD",
        "objective_planner_composition": "OBJECTIVE_PLANNER_COMPOSITION_GAP_INFORMATIVE_NEGATIVE_ONLY",
    }
    for gate, decision in expected.items():
        gates = dict(all_pass)
        gates[gate] = False
        assert _localization_decision(gates) == decision
    assert _localization_decision(all_pass) == "LICENSES_NEW_PREREGISTERED_E1B_DEV2_ONLY"


def test_static_localization_distinguishes_final_only_lora_interference() -> None:
    rows: list[dict] = []
    factual_heads = ("MAP_G", "VALUE_G", "BEST_G", "MAP_P", "VALUE_P", "BEST_P")
    for policy in POLICY_CONDITIONS:
        for encoding in ("RAW0", "CAN0"):
            for head in factual_heads + ("EXPLICIT_G", "EXPLICIT_P"):
                for cue in ("semantic", "neutral"):
                    for renderer in ("audit_matrix_v1", "audit_routefile_v1"):
                        accuracy = 1.0
                        if (
                            policy == "genuine_final"
                            and encoding == "CAN0"
                            and head == "EXPLICIT_G"
                            and cue == "semantic"
                        ):
                            accuracy = 0.0
                        rows.append(
                            {
                                "policy_id": policy,
                                "panel": "static",
                                "encoding": encoding,
                                "query_head": head,
                                "cue_regime": cue,
                                "renderer_id": renderer,
                                "role_assignment": "genuine_first",
                                "direction": "genuine_A",
                                "accuracy": accuracy,
                            }
                        )
    metrics = _static_parsing_metrics(
        {"analysis": FROZEN_ANALYSIS_CONTRACT}, rows
    )
    assert metrics["anchor_task_validity_pass"] is True
    assert metrics["final_policy_competence_pass"] is False
    assert metrics["failure_scope"] == "FINAL_ONLY_LORA_INTERFERENCE"
