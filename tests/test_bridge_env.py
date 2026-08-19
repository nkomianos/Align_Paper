from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from under_extinction.bridge_env import (
    EXTINCTION_PROTOCOL,
    _assay_cases,
    _make_world,
    build_bridge_data,
    load_bridge_environment,
)
from under_extinction.bridge_evaluation import validate_extinction_cases
from under_extinction.bridge_oracle import run_bridge_oracles
from under_extinction.config import load_config
from under_extinction.io import canonical_json, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bridge_config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")


def _built(bridge_config: dict, tmp_path: Path):
    data_dir = tmp_path / "data"
    manifest = build_bridge_data(bridge_config, data_dir)
    return data_dir, manifest, load_bridge_environment(bridge_config, data_dir)


def test_frozen_build_is_deterministic_and_refuses_tampering(
    bridge_config: dict, tmp_path: Path
) -> None:
    data_dir, first, _ = _built(bridge_config, tmp_path)
    second = build_bridge_data(bridge_config, data_dir)
    assert first == second
    assert first["counts"] == {"train": 64, "dev": 12, "test": 12}
    assert first["lexicon_integrity"]["pairwise_disjoint"] is True
    assert first["counterbalancing"]["renderers_per_split"] == 2

    with (data_dir / "dev.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="integrity failure"):
        build_bridge_data(bridge_config, data_dir)


def test_build_refuses_partial_or_extra_existing_directory(
    bridge_config: dict, tmp_path: Path
) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "dev.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError, match="without a verifiable manifest"):
        build_bridge_data(bridge_config, partial)

    complete = tmp_path / "complete"
    build_bridge_data(bridge_config, complete)
    (complete / "unregistered.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="missing, extra"):
        load_bridge_environment(bridge_config, complete)


def test_authorized_split_loader_never_reads_locked_test_records(
    bridge_config: dict, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    build_bridge_data(bridge_config, data_dir)
    locked = data_dir / "test.jsonl"
    payload = bytearray(locked.read_bytes())
    payload[0] = ord("[") if payload[0] != ord("[") else ord("{")
    locked.write_bytes(bytes(payload))

    train_only = load_bridge_environment(
        bridge_config, data_dir, allowed_splits=("train",)
    )
    assert train_only.acquisition_batch(
        trajectory_seed=11, update_index=0, batch_size=2
    )
    dev_only = load_bridge_environment(
        bridge_config, data_dir, allowed_splits=("dev",)
    )
    assert dev_only.extinction_cases(
        split="dev", trajectory_seed=11, checkpoint_update=0
    )
    with pytest.raises(ValueError, match="integrity failure"):
        load_bridge_environment(
            bridge_config, data_dir, allowed_splits=("test",)
        )


def test_split_nonce_lexicons_and_renderers_are_disjoint(
    bridge_config: dict, tmp_path: Path
) -> None:
    data_dir, manifest, _ = _built(bridge_config, tmp_path)
    terms: dict[str, set[str]] = {}
    renderer_sets: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        worlds = list(read_jsonl(data_dir / f"{split}.jsonl"))
        terms[split] = set()
        renderer_sets[split] = {world["renderer_id"] for world in worlds}
        for world in worlds:
            terms[split].update(world["action_names"].values())
            for channel in world["channels"].values():
                terms[split].update(channel["values"].keys())
                terms[split].add(channel["visible_name"])
        assert len(renderer_sets[split]) == 2
        assert len(terms[split]) == manifest["lexicon_integrity"]["term_counts"][split]
    assert not (terms["train"] & terms["dev"])
    assert not (terms["train"] & terms["test"])
    assert not (terms["dev"] & terms["test"])
    assert not (renderer_sets["train"] & renderer_sets["dev"])
    assert not (renderer_sets["train"] & renderer_sets["test"])
    assert not (renderer_sets["dev"] & renderer_sets["test"])


def test_acquisition_is_actual_rewarded_two_step_experience_without_labels(
    bridge_config: dict, tmp_path: Path
) -> None:
    _, _, environment = _built(bridge_config, tmp_path)
    initial_state = dict(environment.state_dict())
    cases = list(
        environment.acquisition_batch(
            trajectory_seed=11,
            update_index=0,
            batch_size=bridge_config["bridge"]["data"]["train_worlds"],
        )
    )
    assert {key for case in cases for key in case} == {
        "case_id", "world_id", "condition", "cue_regime", "messages"
    }
    assert {case["condition"] for case in cases} == {"aligned", "diagnostic_conflict"}
    forbidden = {"controller", "controller_label", "target_action", "oracle_actions"}
    assert all(not (forbidden & set(case)) for case in cases)
    prompts = [case["messages"][-1]["content"] for case in cases]
    assert all("two-stage" in prompt.lower() for prompt in prompts)
    assert all("beneficiary condition" in prompt and "monitor score" in prompt for prompt in prompts)
    assert all(
        forbidden_word not in prompt.lower()
        for prompt in prompts
        for forbidden_word in ("genuine", "proxy", "controller", "target action")
    )

    actions = ["A"] * len(cases)
    outcomes = list(environment.transition_batch(cases, actions))
    assert any(
        outcome["genuine_reward"] != outcome["proxy_reward"] for outcome in outcomes
    ), "diagnostic conflict worlds must identify which reward stream shaped learning"
    assert all("genuine_outcome" in outcome and "proxy_outcome" in outcome for outcome in outcomes)
    after = dict(environment.state_dict())
    assert after["experienced_cases"] == len(cases)
    assert after["transition_batches"] == 1
    assert after["history_sha256"] != initial_state["history_sha256"]

    _, _, restored = _built(bridge_config, tmp_path / "restored")
    restored.load_state_dict(after)
    assert restored.state_dict() == after


def test_environment_rejects_checkpoint_with_unresolved_actions(
    bridge_config: dict, tmp_path: Path
) -> None:
    _, _, environment = _built(bridge_config, tmp_path)
    environment.acquisition_batch(trajectory_seed=1, update_index=0, batch_size=2)
    with pytest.raises(RuntimeError, match="unresolved action batch"):
        environment.state_dict()


def test_extinction_cases_are_fixed_passive_first_choices_with_matched_controls(
    bridge_config: dict, tmp_path: Path
) -> None:
    _, manifest, environment = _built(bridge_config, tmp_path)
    first = list(environment.extinction_cases(split="dev", trajectory_seed=11, checkpoint_update=0))
    later = list(
        environment.extinction_cases(split="dev", trajectory_seed=71, checkpoint_update=300)
    )
    assert canonical_json(first) == canonical_json(later)
    assert len(first) == manifest["assay_case_counts"]["dev"]
    assert len(validate_extinction_cases(first, split="dev")) == len(first)
    assert {case["renderer_id"] for case in first} == {"dev_card_v2", "dev_brief_v2"}
    assert {case["intervention"]["role_assignment"] for case in first} == {
        "genuine_slot_1",
        "genuine_slot_2",
    }

    by_id = {case["case_id"]: case for case in first}
    active = [case for case in first if case["intervention"]["active"]]
    assert active
    for case in active:
        intervention = case["intervention"]
        assert {
            "family", "base_family", "objective", "channel", "mode", "active",
            "role_assignment", "cue_regime", "value_update_type",
        } <= set(intervention)
        assert intervention["family"] in {"value", "transition"}
        assert intervention["base_family"] in {"value", "transition"}
        assert intervention["objective"] in {"genuine", "proxy"}
        assert case["paired_control_id"] in by_id
        assert case["baseline_id"] in by_id
        assert case["extinction_protocol"] == EXTINCTION_PROTOCOL
        assert set(case["expected_actions"]) >= {"genuine", "proxy"}
        if intervention["mode"] == "switch":
            changed = intervention["objective"]
            unchanged = "proxy" if changed == "genuine" else "genuine"
            assert case["expected_actions"][changed] != case["pre_target_action"]
            assert case["expected_actions"][unchanged] == case["pre_target_action"]
        else:
            assert intervention["mode"] == "no_switch"
            assert all(
                case["expected_actions"][objective] == case["pre_target_action"]
                for objective in ("genuine", "proxy")
            )

    transition_switch = next(
        case
        for case in active
        if case["intervention"]["family"] == "transition"
        and case["intervention"]["mode"] == "switch"
    )
    update_text = transition_switch["messages"][-1]["content"].lower()
    assert "operators traversed" in update_text
    assert "transitions swapped" not in update_text
    assert "without involving you" in update_text
    assert "no reward" in update_text


def test_channel_roles_and_action_targets_are_counterbalanced(
    bridge_config: dict, tmp_path: Path
) -> None:
    data_dir, _, _ = _built(bridge_config, tmp_path)
    for split in ("train", "dev", "test"):
        worlds = list(read_jsonl(data_dir / f"{split}.jsonl"))
        assert {world["role_assignment"] for world in worlds} == {
            "genuine_slot_1",
            "genuine_slot_2",
        }
        assert {world["expected_actions"]["genuine"] for world in worlds} == {"A", "B"}
    train = list(read_jsonl(data_dir / "train.jsonl"))
    assert any(
        world["expected_actions"]["genuine"] != world["expected_actions"]["proxy"]
        for world in train
    )
    assert any(
        world["expected_actions"]["genuine"] == world["expected_actions"]["proxy"]
        for world in train
    )


def test_formal_pilot_fully_crosses_semantic_and_neutral_cue_regimes(
    tmp_path: Path,
) -> None:
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    data_dir = tmp_path / "pilot-data"
    manifest = build_bridge_data(pilot, data_dir)
    cue = manifest["cue_regimes"]
    assert cue["formal_ablation"] is True
    assert cue["dev_test_full_intervention_cross_per_regime"] is True
    for split in ("train", "dev", "test"):
        assert cue["counts"][split]["semantic"] > 0
        assert cue["counts"][split]["neutral"] > 0
    cases = load_bridge_environment(pilot, data_dir).extinction_cases(
        split="dev", trajectory_seed=11, checkpoint_update=0
    )
    assert {case["cue_regime"] for case in cases} == {"semantic", "neutral"}
    neutral_prompts = [
        case["messages"][-1]["content"]
        for case in cases
        if case["cue_regime"] == "neutral"
    ]
    assert all("zorin channel" in prompt and "pelad channel" in prompt for prompt in neutral_prompts)
    assert all("beneficiary condition" not in prompt and "monitor score" not in prompt for prompt in neutral_prompts)
    for split in ("dev", "test"):
        worlds = list(read_jsonl(data_dir / f"{split}.jsonl"))
        renderer_ids = {world["renderer_id"] for world in worlds}
        for regime in ("semantic", "neutral"):
            designs = {
                (
                    world["audit_assignment"]["family"],
                    world["audit_assignment"]["channel"],
                    world["audit_assignment"]["mode"],
                )
                for world in worlds
                if world["cue_regime"] == regime
            }
            assert len(designs) == 8
            for design in designs:
                observed = {
                    (
                        world["role_assignment"],
                        world["renderer_id"],
                        world["expected_actions"]["genuine"],
                    )
                    for world in worlds
                    if world["cue_regime"] == regime
                    and (
                        world["audit_assignment"]["family"],
                        world["audit_assignment"]["channel"],
                        world["audit_assignment"]["mode"],
                    ) == design
                }
                expected = {
                    (role, renderer, action)
                    for role in ("genuine_slot_1", "genuine_slot_2")
                    for renderer in renderer_ids
                    for action in ("A", "B")
                }
                assert observed == expected


def test_all_six_renderers_have_independent_base_and_intervention_language(
    tmp_path: Path,
) -> None:
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    data_dir = tmp_path / "renderer-language"
    manifest = build_bridge_data(pilot, data_dir)
    provenance = manifest["renderer_language_provenance"]
    entries = [
        entry
        for split in ("train", "dev", "test")
        for entry in provenance["by_split"][split]
    ]
    assert provenance["independently_authored_template_ids"] is True
    assert provenance["base_and_intervention_template_ids_pairwise_distinct"] is True
    assert len(entries) == 6
    assert len({entry["base_template_id"] for entry in entries}) == 6
    assert len({entry["intervention_template_id"] for entry in entries}) == 6
    assert len({entry["base_render_sha256"] for entry in entries}) == 6
    assert len({entry["intervention_render_sha256"] for entry in entries}) == 6

    expected = {
        "train_ledger_v2": ("ROUTE LEDGER", "OBSERVATION APPENDED WITHOUT ACTION", "LEDGER REVISION"),
        "train_dispatch_v2": ("DISPATCH BRIEF", "READ-ONLY NOTICE", "read-only dispatch notice"),
        "dev_card_v2": ("ONE-CHOICE OPERATIONS CARD", "UNENACTED CARD UPDATE", "CARD AMENDMENT"),
        "dev_brief_v2": ("NAVIGATION BRIEF", "PASSIVELY RECEIVED NAVIGATION NOTICE", "passive valuation message"),
        "test_manifest_v2": ("FIELD MANIFEST", "REMOTE MANIFEST ADDENDUM", "MANIFEST DELTA"),
        "test_fieldnote_v2": ("OBSERVER NOTE", "LATER NOTE, LEARNED BY OBSERVATION", "remote observer reports"),
    }
    rendered_prompts: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        seen: set[str] = set()
        for index in range(64):
            world = _make_world(pilot, split, index)
            if world["renderer_id"] in seen:
                continue
            seen.add(world["renderer_id"])
            world["audit_assignment"] = {
                "family": "value",
                "channel": "genuine",
                "mode": "switch",
                "value_update_type": "devalue_preferred",
            }
            active = next(case for case in _assay_cases(world) if case["condition"] == "value_switch")
            prompt = active["messages"][-1]["content"]
            rendered_prompts[world["renderer_id"]] = prompt
            base_anchor, shell_anchor, event_anchor = expected[world["renderer_id"]]
            assert base_anchor in prompt
            assert shell_anchor in prompt
            assert event_anchor.lower() in prompt.lower()
            if len(seen) == 2:
                break
        assert len(seen) == 2
    assert set(rendered_prompts) == set(expected)
    assert len(set(rendered_prompts.values())) == 6

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        left_base = {
            entry["base_template_id"] for entry in provenance["by_split"][left]
        }
        right_base = {
            entry["base_template_id"] for entry in provenance["by_split"][right]
        }
        left_updates = {
            entry["intervention_template_id"] for entry in provenance["by_split"][left]
        }
        right_updates = {
            entry["intervention_template_id"] for entry in provenance["by_split"][right]
        }
        assert left_base.isdisjoint(right_base)
        assert left_updates.isdisjoint(right_updates)


def test_formal_value_switch_signs_and_matched_shams_are_fully_counterbalanced(
    tmp_path: Path,
) -> None:
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    data_dir = tmp_path / "value-direction-balance"
    manifest = build_bridge_data(pilot, data_dir)
    balance = manifest["counterbalancing"]
    assert balance["value_switch_directions"] == [
        "devalue_preferred",
        "upvalue_nonpreferred",
    ]
    assert balance["value_switch_sham_sign_matched"] is True
    assert balance["formal_each_direction_full_surface_cross"] is True

    environment = load_bridge_environment(pilot, data_dir)
    for split in ("dev", "test"):
        worlds = list(read_jsonl(data_dir / f"{split}.jsonl"))
        renderer_ids = {world["renderer_id"] for world in worlds}
        for regime in ("semantic", "neutral"):
            counts = balance["value_switch_counts"][split][regime]
            assert counts["devalue_preferred"] == counts["upvalue_nonpreferred"] > 0
            for channel in ("genuine", "proxy"):
                for update_type in ("devalue_preferred", "upvalue_nonpreferred"):
                    observed = {
                        (
                            world["role_assignment"],
                            world["renderer_id"],
                            world["expected_actions"]["genuine"],
                        )
                        for world in worlds
                        if world["cue_regime"] == regime
                        and world["audit_assignment"] == {
                            "family": "value",
                            "channel": channel,
                            "mode": "switch",
                            "value_update_type": update_type,
                        }
                    }
                    expected_surface = {
                        (role, renderer, action)
                        for role in ("genuine_slot_1", "genuine_slot_2")
                        for renderer in renderer_ids
                        for action in ("A", "B")
                    }
                    assert observed == expected_surface

        cases = list(
            environment.extinction_cases(split=split, trajectory_seed=11, checkpoint_update=0)
        )
        by_id = {case["case_id"]: case for case in cases}
        switches = [
            case
            for case in cases
            if case["intervention"]["active"]
            and case["intervention"]["base_family"] == "value"
            and case["intervention"]["mode"] == "switch"
        ]
        assert {case["intervention"]["value_update_type"] for case in switches} == {
            "devalue_preferred",
            "upvalue_nonpreferred",
        }
        for active in switches:
            update_type = active["intervention"]["value_update_type"]
            control = by_id[active["paired_control_id"]]
            assert control["intervention"]["value_update_type"] == update_type
            assert active["expected_actions"][active["intervention"]["objective"]] != active["pre_target_action"]
            marker = "-1" if update_type == "devalue_preferred" else "2"
            assert marker in active["messages"][-1]["content"]
            assert marker in control["messages"][-1]["content"]


def test_cpu_oracles_recover_2x3_mixtures_and_open_set(
    bridge_config: dict, tmp_path: Path
) -> None:
    data_dir, _, _ = _built(bridge_config, tmp_path)
    destination = tmp_path / "oracle.jsonl"
    result = run_bridge_oracles(
        bridge_config, "dev", destination=destination, data_dir=data_dir
    )
    assert result == destination
    rows = list(read_jsonl(result))
    assert {row["policy_id"] for row in rows} == {
        "pure_genuine_model_based",
        "pure_genuine_successor_representation",
        "pure_genuine_habit",
        "pure_proxy_model_based",
        "pure_proxy_successor_representation",
        "pure_proxy_habit",
        "mixture_genuine_mb_habit_50_50",
        "mixture_proxy_sr_habit_60_40",
        "open_set_anti_revaluation",
    }
    summary = json.loads(result.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert all(summary["checks"].values())
    assert summary["calibration"]["open_set_anti_revaluation"]["open_set"] is True
    assert summary["calibration"]["mixture_genuine_mb_habit_50_50"]["open_set"] is False
    assert summary["prototype_collision"]["members"] == [
        "pure_genuine_habit",
        "pure_proxy_habit",
    ]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_bridge_oracles(bridge_config, "dev", destination=destination, data_dir=data_dir)
