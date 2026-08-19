from __future__ import annotations

import copy
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch

import under_extinction.bridge_evaluation as bridge_evaluation
import under_extinction.bridge_training as bridge_training
from under_extinction.bridge_analysis import paired_effects
from under_extinction.bridge_env import build_bridge_data, load_bridge_environment
from under_extinction.bridge_evaluation import (
    BridgeEvaluationSpec,
    EXTINCTION_INVARIANTS,
    adapter_reload_diagnostic,
    config_bound_bridge_evaluation_spec,
    configured_bridge_evaluation_spec_sha256,
    fixed_bridge_checkpoints,
    evaluate_unchanged_base_control,
    evaluate_bridge_run,
    legal_choice_diagnostics,
    parse_unconstrained_choice,
    validate_extinction_cases,
)
from under_extinction.bridge_training import (
    BridgeRunStopped,
    BridgeTrainingSpec,
    _materialize_final_adapter,
    _save_checkpoint,
    acquisition_gate_window_diagnostics_summary,
    acquisition_diagnostics_summary,
    categorical_policy_kl,
    config_bound_bridge_training_spec,
    configured_bridge_spec_sha256,
    differentiable_choice_log_probs,
    initialize_acquisition_diagnostics,
    initialize_acquisition_gate_window_diagnostics,
    legal_plus_other_policy_kl,
    policy_gradient_objective,
    rollout_advantages,
    sample_legal_actions,
    selected_arm_rewards,
    train_bridge_arm,
    update_reward_baseline,
    update_acquisition_diagnostics,
    update_acquisition_gate_window_diagnostics,
    validate_acquisition_gate_window_state,
    validate_acquisition_diagnostics_state,
    validate_acquisition_batch,
    validate_realized_outcomes,
)
from under_extinction.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _test_runtime_attestation() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "test",
        "kind": "test_model_runtime",
    }
    payload["attestation_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _learn_toy_arm(arm: str) -> torch.Tensor:
    """Two-logit policy trained only from its selected, realized reward stream."""
    logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([logits], lr=0.08)
    baseline = 0.0
    initialized = False
    for update in range(120):
        log_probs = torch.log_softmax(logits, dim=0).expand(64, -1)
        chosen = sample_legal_actions(log_probs, seed=10_000 + update)
        outcomes = [
            {
                "genuine_reward": float(int(index) == 0),
                "proxy_reward": float(int(index) == 1),
            }
            for index in chosen.tolist()
        ]
        rewards = selected_arm_rewards(outcomes, arm)
        loss, metrics = policy_gradient_objective(
            log_probs,
            chosen,
            rewards,
            baseline=baseline,
            entropy_coefficient=0.005,
            normalize_advantages=True,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        baseline, initialized = update_reward_baseline(
            baseline,
            metrics["reward_mean"],
            momentum=0.9,
            initialized=initialized,
        )
    return torch.softmax(logits.detach(), dim=0)


def test_paired_policy_gradient_learns_opposed_reward_arms() -> None:
    genuine = _learn_toy_arm("genuine")
    proxy = _learn_toy_arm("proxy")
    assert genuine[0] > 0.97
    assert proxy[1] > 0.97
    assert genuine[0] - proxy[0] > 0.94


def test_reward_selection_and_validation_never_mix_arms() -> None:
    cases = [
        {"case_id": "one", "messages": [{"role": "user", "content": "Choose A or B."}]},
        {"case_id": "two", "messages": [{"role": "user", "content": "Choose A or B."}]},
    ]
    outcomes = [
        {"case_id": "one", "genuine_reward": 1.0, "proxy_reward": -3.0},
        {"case_id": "two", "genuine_reward": 2.0, "proxy_reward": 4.0},
    ]
    validated_cases = validate_acquisition_batch(cases, 2)
    validated = validate_realized_outcomes(validated_cases, outcomes)
    assert selected_arm_rewards(validated, "G").tolist() == [1.0, 2.0]
    assert selected_arm_rewards(validated, "P").tolist() == [-3.0, 4.0]

    leaked = copy.deepcopy(cases)
    leaked[0]["target_action"] = "A"
    with pytest.raises(ValueError, match="supervised/controller"):
        validate_acquisition_batch(leaked, 2)
    nonfinite = copy.deepcopy(outcomes)
    nonfinite[0]["proxy_reward"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_realized_outcomes(cases, nonfinite)


def test_acquisition_diagnostics_separate_cue_and_conflict_reward() -> None:
    initial = initialize_acquisition_diagnostics("genuine", ["semantic", "neutral"])
    cases = [
        {"case_id": "s-a", "cue_regime": "semantic", "condition": "aligned"},
        {
            "case_id": "s-c",
            "cue_regime": "semantic",
            "condition": "diagnostic_conflict",
        },
        {
            "case_id": "n-c",
            "cue_regime": "neutral",
            "condition": "diagnostic_conflict",
        },
    ]
    outcomes = [
        {"case_id": "s-a", "genuine_reward": 1.0, "proxy_reward": 0.0},
        {"case_id": "s-c", "genuine_reward": 0.0, "proxy_reward": 1.0},
        {"case_id": "n-c", "genuine_reward": 1.0, "proxy_reward": 0.0},
    ]
    updated = update_acquisition_diagnostics(initial, cases, outcomes)
    summary = acquisition_diagnostics_summary(updated)
    assert summary["overall"]["sample_count"] == 3
    assert summary["cells"]["semantic"]["aligned"]["optimized_reward_mean"] == 1.0
    assert summary["cells"]["semantic"]["diagnostic_conflict"][
        "optimal_action_accuracy"
    ] == 0.0
    assert summary["cells"]["neutral"]["diagnostic_conflict"][
        "optimized_reward_mean"
    ] == 1.0
    assert initial["cells"]["semantic"]["aligned"]["count"] == 0
    assert validate_acquisition_diagnostics_state(
        updated, arm="genuine", cue_regimes=["semantic", "neutral"]
    ) == updated

    invalid = copy.deepcopy(cases)
    invalid[0]["condition"] = "easy"
    with pytest.raises(ValueError, match="condition"):
        update_acquisition_diagnostics(initial, invalid, outcomes)
    nonbinary = copy.deepcopy(outcomes)
    nonbinary[0]["genuine_reward"] = 0.5
    with pytest.raises(ValueError, match="binary rewards"):
        update_acquisition_diagnostics(initial, cases, nonbinary)


def test_acquisition_gate_window_is_exact_trailing_and_resume_safe() -> None:
    cases = [
        {"case_id": "a", "cue_regime": "semantic", "condition": "aligned"},
        {
            "case_id": "c",
            "cue_regime": "semantic",
            "condition": "diagnostic_conflict",
        },
    ]
    state = initialize_acquisition_gate_window_diagnostics(
        "genuine", ["semantic"], window_updates=2, samples_per_update=2
    )
    for completed_update, rewards in enumerate(((1.0, 0.0), (1.0, 1.0), (0.0, 0.0)), 1):
        outcomes = [
            {"case_id": case["case_id"], "genuine_reward": reward}
            for case, reward in zip(cases, rewards, strict=True)
        ]
        state = update_acquisition_gate_window_diagnostics(
            state, cases, outcomes, completed_update=completed_update
        )
    summary = acquisition_gate_window_diagnostics_summary(state)
    assert summary["diagnostic_scope"] == "trailing_optimizer_updates"
    assert summary["completed_updates"] == 3
    assert summary["covered_updates"] == {
        "first_completed_update": 2,
        "last_completed_update": 3,
        "update_count": 2,
    }
    assert summary["overall"]["sample_count"] == 4
    assert summary["overall"]["optimal_action_count"] == 2
    assert [entry["completed_update"] for entry in state["updates"]] == [2, 3]
    assert validate_acquisition_gate_window_state(
        state,
        arm="genuine",
        cue_regimes=["semantic"],
        window_updates=2,
        samples_per_update=2,
    ) == state

    tampered = copy.deepcopy(state)
    tampered["updates"][-1]["cells"]["semantic"]["aligned"]["count"] += 1
    with pytest.raises(ValueError, match="complete rollout batch"):
        validate_acquisition_gate_window_state(
            tampered,
            arm="genuine",
            cue_regimes=["semantic"],
            window_updates=2,
            samples_per_update=2,
        )


def test_checkpoint_persists_resume_safe_acquisition_diagnostics(tmp_path) -> None:
    class TinyArtifact(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def save_pretrained(self, directory, *, safe_serialization):
            assert safe_serialization is True
            (Path(directory) / "adapter_model.safetensors").write_bytes(b"weights")

    class TinyTokenizer:
        def save_pretrained(self, directory):
            (Path(directory) / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class TinyEnvironment:
        def state_dict(self):
            return {"experienced_cases": 2, "cursor": 2}

    diagnostics = initialize_acquisition_diagnostics("genuine", ["semantic"])
    diagnostics = update_acquisition_diagnostics(
        diagnostics,
        [
            {"case_id": "a", "cue_regime": "semantic", "condition": "aligned"},
            {
                "case_id": "c",
                "cue_regime": "semantic",
                "condition": "diagnostic_conflict",
            },
        ],
        [
            {"case_id": "a", "genuine_reward": 1.0},
            {"case_id": "c", "genuine_reward": 0.0},
        ],
    )
    gate_window = initialize_acquisition_gate_window_diagnostics(
        "genuine", ["semantic"], window_updates=1, samples_per_update=2
    )
    gate_window = update_acquisition_gate_window_diagnostics(
        gate_window,
        [
            {"case_id": "a", "cue_regime": "semantic", "condition": "aligned"},
            {
                "case_id": "c",
                "cue_regime": "semantic",
                "condition": "diagnostic_conflict",
            },
        ],
        [
            {"case_id": "a", "genuine_reward": 1.0},
            {"case_id": "c", "genuine_reward": 0.0},
        ],
        completed_update=1,
    )
    model = TinyArtifact()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    record = _save_checkpoint(
        run_dir=tmp_path,
        model=model,
        tokenizer=TinyTokenizer(),
        optimizer=optimizer,
        scheduler=scheduler,
        environment=TinyEnvironment(),
        completed_updates=1,
        baseline=0.5,
        baseline_initialized=True,
        arm="genuine",
        seed=11,
        config_sha256="c" * 64,
        spec_sha256="s" * 64,
        environment_provenance={"environment_id": "tiny"},
        initial_environment_state_sha256="e" * 64,
        acquisition_diagnostics_state=diagnostics,
        acquisition_gate_window_diagnostics_state=gate_window,
        model_runtime_attestation=_test_runtime_attestation(),
    )
    expected = acquisition_diagnostics_summary(diagnostics)
    expected_gate_window = acquisition_gate_window_diagnostics_summary(gate_window)
    assert record["acquisition_diagnostics"] == expected
    assert record["acquisition_gate_window_diagnostics"] == expected_gate_window
    checkpoint = tmp_path / "checkpoints" / "checkpoint-000001"
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["acquisition_diagnostics"] == expected
    assert manifest["acquisition_gate_window_diagnostics"] == expected_gate_window
    saved = torch.load(checkpoint / "bridge_state.pt", weights_only=False)
    assert saved["acquisition_diagnostics_state"] == diagnostics
    assert saved["acquisition_gate_window_diagnostics_state"] == gate_window


def test_training_spec_consumes_configured_schedule_and_optimizer_knobs() -> None:
    config = {
        "bridge": {"checkpoint_fractions": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]},
        "training": {
            "algorithm": "reinforce_exact_binary",
            "updates": 300,
            "rollout_batch_size": 64,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-5,
            "warmup_ratio": 0.05,
            "entropy_coefficient": 0.01,
            "kl_coefficient": 0.02,
            "normalize_advantages": True,
            "max_grad_norm": 1.0,
            "checkpoint_steps": 50,
        },
    }
    spec = BridgeTrainingSpec.from_config(config)
    assert spec.microbatch_size == 32
    assert {0, 30, 50, 75, 150, 225, 300} <= set(spec.checkpoint_updates)
    assert spec.gradient_accumulation_steps == 2
    assert spec.warmup_ratio == pytest.approx(0.05)
    assert spec.kl_coefficient == pytest.approx(0.02)
    assert spec.normalize_advantages is True


def test_evidence_training_spec_rejects_effective_runtime_overrides() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    configured = BridgeTrainingSpec.from_config(config)
    assert config_bound_bridge_training_spec(config) == configured
    assert config_bound_bridge_training_spec(
        config, {"learning_rate": configured.learning_rate}
    ) == configured
    with pytest.raises(ValueError, match="forbids settings that differ"):
        config_bound_bridge_training_spec(
            config, {"learning_rate": configured.learning_rate * 2.0}
        )


def test_run_evaluation_rejects_training_spec_not_bound_to_config(tmp_path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    spec = BridgeTrainingSpec.from_config(config)
    run = tmp_path / "run"
    for update in spec.checkpoint_updates:
        (run / "checkpoints" / f"checkpoint-{update:06d}").mkdir(parents=True)
    manifest_path = run / "bridge_manifest.json"
    manifest = {
        "state": "COMPLETE",
        "bridge_spec": json.loads(json.dumps(spec.__dict__)),
        "bridge_spec_sha256": configured_bridge_spec_sha256(config),
        "bridge_spec_source": "loaded_config_exact",
        "optimizer_update_semantics": {
            "checkpoint_updates": list(spec.checkpoint_updates)
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert fixed_bridge_checkpoints(run, config=config) == [
        run.resolve() / "checkpoints" / f"checkpoint-{update:06d}"
        for update in spec.checkpoint_updates
    ]

    manifest["bridge_spec"]["learning_rate"] *= 2.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly bound"):
        fixed_bridge_checkpoints(run, config=config)

    manifest["bridge_spec"] = json.loads(json.dumps(spec.__dict__))
    manifest["bridge_spec_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly bound"):
        fixed_bridge_checkpoints(run, config=config)


def test_checkpoint_fraction_must_identify_an_exact_update() -> None:
    with pytest.raises(ValueError, match="exact optimizer update"):
        BridgeTrainingSpec(updates=7, checkpoint_fractions=(0.5,))


def test_advantages_are_normalized_over_full_rollout_not_each_microbatch() -> None:
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    advantages = rollout_advantages(rewards, baseline=0.25, normalize=True)
    assert advantages.tolist() == pytest.approx([1.0, 1.0, -1.0, -1.0])
    assert advantages[:2].mean() > 0
    assert advantages[2:].mean() < 0


def test_legal_preference_does_not_hide_tiny_legal_choice_mass() -> None:
    diagnostics = legal_choice_diagnostics(-100.0, -110.0)
    assert diagnostics["probability_A"] > 0.999
    assert diagnostics["legal_choice_mass"] < 1e-40
    assert diagnostics["log_legal_choice_mass"] == pytest.approx(-99.9999546)


def test_kl_constrains_non_choice_mass_not_only_conditional_ab_preference() -> None:
    current_raw = torch.log(torch.tensor([[0.10, 0.10]]))
    reference_raw = torch.log(torch.tensor([[0.40, 0.40]]))
    current_conditional = torch.log_softmax(current_raw, dim=1)
    reference_conditional = torch.log_softmax(reference_raw, dim=1)
    assert categorical_policy_kl(current_conditional, reference_conditional).item() == pytest.approx(0.0)
    assert legal_plus_other_policy_kl(current_raw, reference_raw).item() > 0.5


def test_next_token_scorer_uses_full_prompt_budget_with_left_truncation() -> None:
    class CharacterTokenizer:
        chat_template = "present"
        pad_token_id = 0

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return messages[-1]["content"] + "|"

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(character) for character in text]}

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.last_input_ids = None

        def forward(self, *, input_ids, attention_mask, logits_to_keep):
            assert logits_to_keep == 1
            self.last_input_ids = input_ids.detach().clone()
            logits = self.anchor + torch.zeros(
                input_ids.shape[0], 1, 256, device=input_ids.device
            )
            return type("Output", (), {"logits": logits})()

    tokenizer = CharacterTokenizer()
    model = RecordingModel()
    case = {"messages": [{"role": "user", "content": "0123456789"}]}
    differentiable_choice_log_probs(model, tokenizer, [case], max_length=5)
    assert model.last_input_ids.tolist() == [[ord(character) for character in "6789|"]]


@pytest.mark.parametrize(
    ("text", "parsed", "status"),
    [
        ("A", "A", "exact"),
        ("  B\n", "B", "exact"),
        ("", None, "empty"),
        ("A because", None, "extra_text"),
        ("Choice A", None, "non_choice"),
    ],
)
def test_unconstrained_first_token_parser(text: str, parsed: str | None, status: str) -> None:
    assert parse_unconstrained_choice(text) == (parsed, status)


def _valid_extinction_cases() -> list[dict[str, object]]:
    protocol = dict(EXTINCTION_INVARIANTS)
    common = {
        "split": "dev",
        "world_id": "world-1",
        "renderer_id": "renderer-1",
        "base_template_id": "renderer-1-base-v1",
        "intervention_template_id": "renderer-1-update-v1",
        "cue_regime": "semantic",
        "pair_id": "pair-1",
        "condition": "extinction",
        "messages": [{"role": "user", "content": "Choose A or B."}],
        "pre_target_action": "A",
        "expected_actions": {"genuine": "A", "proxy": "B"},
        "extinction_protocol": protocol,
    }
    return [
        {
            **copy.deepcopy(common),
            "case_id": "control",
            "intervention": {
                "family": "genuine_value",
                "base_family": "value",
                "objective": "genuine",
                "mode": "sham",
                "active": False,
                "cue_regime": "semantic",
                "value_update_type": "devalue_preferred",
            },
        },
        {
            **copy.deepcopy(common),
            "case_id": "active",
            "paired_control_id": "control",
            "intervention": {
                "family": "genuine_value",
                "base_family": "value",
                "objective": "genuine",
                "mode": "switch",
                "active": True,
                "cue_regime": "semantic",
                "value_update_type": "devalue_preferred",
            },
        },
    ]


def test_extinction_validator_enforces_passivity_first_choice_and_pairing() -> None:
    valid = _valid_extinction_cases()
    assert len(validate_extinction_cases(valid, split="dev")) == 2

    for key in EXTINCTION_INVARIANTS:
        broken = copy.deepcopy(valid)
        expected = broken[0]["extinction_protocol"][key]
        broken[0]["extinction_protocol"][key] = 2 if expected == 1 and type(expected) is int else not expected
        with pytest.raises(ValueError, match="Extinction invariant"):
            validate_extinction_cases(broken, split="dev")

    unpaired = copy.deepcopy(valid)
    unpaired[1]["paired_control_id"] = "missing"
    with pytest.raises(ValueError, match="paired passive control"):
        validate_extinction_cases(unpaired, split="dev")


def test_extinction_generation_is_frozen_to_one_first_token() -> None:
    assert BridgeEvaluationSpec().max_new_tokens == 1
    with pytest.raises(ValueError, match="first-action"):
        BridgeEvaluationSpec(max_new_tokens=2)


def test_evaluation_spec_is_exactly_bound_to_loaded_config() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    configured = BridgeEvaluationSpec.from_config(config)
    assert config_bound_bridge_evaluation_spec(config) == configured
    assert config_bound_bridge_evaluation_spec(
        config, {"generation_subset_size": configured.generation_subset_size}
    ) == configured
    assert len(configured_bridge_evaluation_spec_sha256(config)) == 64
    with pytest.raises(ValueError, match="forbids settings"):
        config_bound_bridge_evaluation_spec(config, {"generation_subset_size": 1})


def test_fixed_checkpoint_series_rejects_missing_dynamics_point(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "bridge_manifest.json").write_text(
        json.dumps(
            {
                "state": "COMPLETE",
                "optimizer_update_semantics": {"checkpoint_updates": [0, 5, 10]},
            }
        ),
        encoding="utf-8",
    )
    for update in (0, 10):
        (run / "checkpoints" / f"checkpoint-{update:06d}").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="checkpoint-000005"):
        fixed_bridge_checkpoints(run)


def test_initial_checkpoint_reload_probe_is_explicitly_unavailable(tmp_path) -> None:
    diagnostic = adapter_reload_diagnostic(None, None, tmp_path, max_length=32)
    assert diagnostic == {
        "available": False,
        "probe_count": 0,
        "max_probability_delta": None,
    }


def test_base_control_anchors_to_checkpoint_zero_without_adapter(tmp_path, monkeypatch) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    spec = BridgeTrainingSpec.from_config(config)
    updates = list(spec.checkpoint_updates)
    run = tmp_path / "run"
    for update in updates:
        (run / "checkpoints" / f"checkpoint-{update:06d}").mkdir(parents=True)
    (run / "bridge_manifest.json").write_text(
        json.dumps(
            {
                "state": "COMPLETE",
                "bridge_spec": json.loads(json.dumps(spec.__dict__)),
                "bridge_spec_sha256": configured_bridge_spec_sha256(config),
                "bridge_spec_source": "loaded_config_exact",
                "optimizer_update_semantics": {"checkpoint_updates": updates},
            }
        ),
        encoding="utf-8",
    )
    observed = {}

    def fake_evaluate(*args, **kwargs):
        observed.update(kwargs)
        return tmp_path / "base.jsonl"

    monkeypatch.setattr(bridge_evaluation, "evaluate_bridge_checkpoint", fake_evaluate)
    result = evaluate_unchanged_base_control(
        config, object(), anchor_run_dir=run, anchor_arm="genuine", pair_seed=11
    )
    assert result == tmp_path / "base.jsonl"
    assert observed["checkpoint"].name == "checkpoint-000000"
    assert observed["base_policy"] is True
    assert observed["split"] == "dev"


def test_run_evaluation_combines_every_fixed_checkpoint_for_cli(tmp_path, monkeypatch) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    spec = BridgeTrainingSpec.from_config(config)
    updates = list(spec.checkpoint_updates)
    run = tmp_path / "run"
    for update in updates:
        (run / "checkpoints" / f"checkpoint-{update:06d}").mkdir(parents=True)
    (run / "bridge_manifest.json").write_text(
        json.dumps(
            {
                "state": "COMPLETE",
                "bridge_spec": json.loads(json.dumps(spec.__dict__)),
                "bridge_spec_sha256": configured_bridge_spec_sha256(config),
                "bridge_spec_source": "loaded_config_exact",
                "optimizer_update_semantics": {"checkpoint_updates": updates},
            }
        ),
        encoding="utf-8",
    )

    def fake_evaluate(*args, **kwargs):
        destination = kwargs["destination"]
        update = int(kwargs["checkpoint"].name.rsplit("-", 1)[1])
        destination.write_text(json.dumps({"checkpoint_update": update}) + "\n", encoding="utf-8")
        timing = {
            "schema_version": "1.0",
            "model_and_tokenizer_load_wall_seconds": 1.0,
            "adapter_load_and_attestation_wall_seconds": 0.5,
            "adapter_reload_probe_wall_seconds": 0.25,
            "forced_scoring_wall_seconds": 2.0,
            "forced_record_count": 8,
            "generation_wall_seconds": 1.0,
            "generated_record_count": 2,
            "peak_vram_bytes": 1_000 + update,
            "write_finalize_wall_seconds": 0.1,
            "total_wall_seconds": 4.85,
        }
        destination.with_suffix(".summary.json").write_text(
            json.dumps(
                {
                    "checkpoint_update": update,
                    "predictions_path": str(destination),
                    "model_runtime_attestation": _test_runtime_attestation(),
                    "model_runtime_contract": {"contract_sha256": "a" * 64},
                    "timing": timing,
                }
            ),
            encoding="utf-8",
        )
        return destination

    monkeypatch.setattr(bridge_evaluation, "evaluate_bridge_checkpoint", fake_evaluate)
    monkeypatch.setattr(
        bridge_evaluation,
        "verify_model_runtime_attestation",
        lambda config, attestation: dict(attestation),
    )
    target = tmp_path / "combined.jsonl"
    class DummyEnvironment:
        def state_dict(self):
            return {"cursor": 0}

        def load_state_dict(self, state):
            assert state == {"cursor": 0}

    result = evaluate_bridge_run(
        config,
        DummyEnvironment(),
        run_dir=run,
        arm="genuine",
        pair_seed=11,
        destination=target,
    )
    assert result == target
    updates = [json.loads(line)["checkpoint_update"] for line in target.read_text().splitlines()]
    assert updates == list(spec.checkpoint_updates)
    summary = json.loads(target.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint_updates"] == list(spec.checkpoint_updates)
    assert len(summary["timing"]["per_checkpoint"]) == len(spec.checkpoint_updates)
    assert summary["timing"]["peak_vram_bytes"] == 1_000 + max(
        spec.checkpoint_updates
    )
    assert summary["model_runtime_attestation_sha256"] == _test_runtime_attestation()[
        "attestation_sha256"
    ]


def test_final_adapter_materialization_is_resume_idempotent(tmp_path) -> None:
    run = tmp_path / "run"
    source = run / "checkpoints" / "checkpoint-000001"
    source.mkdir(parents=True)
    adapter = source / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    (source / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "completed_updates": 1,
                "file_sha256": {"adapter_model.safetensors": digest},
                "model_runtime_attestation": _test_runtime_attestation(),
                "model_runtime_attestation_sha256": _test_runtime_attestation()[
                    "attestation_sha256"
                ],
            }
        ),
        encoding="utf-8",
    )
    first = _materialize_final_adapter(run, source)
    second = _materialize_final_adapter(run, source)
    assert first == second == run / "final_adapter"

    first.rename(run / ".final_adapter.tmp")
    resumed = _materialize_final_adapter(run, source)
    assert resumed == run / "final_adapter"


def test_concrete_environment_is_paired_resumable_and_passive_at_extinction(tmp_path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    data = tmp_path / "bridge-data"
    first_manifest = build_bridge_data(config, data)
    assert build_bridge_data(config, data) == first_manifest

    genuine_environment = load_bridge_environment(config, data)
    proxy_environment = load_bridge_environment(config, data)
    initial_state = dict(genuine_environment.state_dict())
    assert proxy_environment.state_dict() == initial_state
    assert proxy_environment.provenance() == genuine_environment.provenance()
    json.dumps(initial_state, sort_keys=True)

    genuine_cases = list(
        genuine_environment.acquisition_batch(
            trajectory_seed=11, update_index=0, batch_size=32
        )
    )
    proxy_cases = list(
        proxy_environment.acquisition_batch(
            trajectory_seed=11, update_index=0, batch_size=32
        )
    )
    assert genuine_cases == proxy_cases
    validate_acquisition_batch(genuine_cases, 32)
    with pytest.raises(RuntimeError, match="unresolved action batch"):
        genuine_environment.state_dict()

    actions = ["A" if index % 2 == 0 else "B" for index in range(len(genuine_cases))]
    genuine_outcomes = list(genuine_environment.transition_batch(genuine_cases, actions))
    proxy_outcomes = list(proxy_environment.transition_batch(proxy_cases, actions))
    assert genuine_outcomes == proxy_outcomes
    assert genuine_environment.state_dict() == proxy_environment.state_dict()

    resumed_environment = load_bridge_environment(config, data)
    resumed_environment.load_state_dict(genuine_environment.state_dict())
    next_original = list(
        genuine_environment.acquisition_batch(
            trajectory_seed=11, update_index=1, batch_size=32
        )
    )
    next_resumed = list(
        resumed_environment.acquisition_batch(
            trajectory_seed=11, update_index=1, batch_size=32
        )
    )
    assert next_resumed == next_original
    next_actions = ["B"] * len(next_original)
    assert resumed_environment.transition_batch(next_resumed, next_actions) == (
        genuine_environment.transition_batch(next_original, next_actions)
    )
    assert resumed_environment.state_dict() == genuine_environment.state_dict()

    state_before_assay = dict(genuine_environment.state_dict())
    checkpoint_zero_cases = list(
        genuine_environment.extinction_cases(
            split="dev", trajectory_seed=11, checkpoint_update=0
        )
    )
    final_cases = list(
        genuine_environment.extinction_cases(
            split="dev", trajectory_seed=999, checkpoint_update=300
        )
    )
    assert final_cases == checkpoint_zero_cases
    assert genuine_environment.state_dict() == state_before_assay
    validated = validate_extinction_cases(checkpoint_zero_cases, split="dev")
    assert all(set(case["expected_actions"]) >= {"genuine", "proxy"} for case in validated)
    assert {case["intervention"]["base_family"] for case in validated} >= {
        "value",
        "transition",
    }


def test_concrete_environment_interventions_reach_paired_analysis() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    # Building in the configured output tree is unnecessary: construct the frozen
    # environment through the public loader in a test-owned temporary corpus.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        data = Path(directory) / "bridge-data"
        build_bridge_data(config, data)
        environment = load_bridge_environment(config, data)
        cases = environment.extinction_cases(
            split="dev", trajectory_seed=11, checkpoint_update=0
        )
        rows = []
        for arm in ("genuine", "proxy"):
            for case in cases:
                probability_a = 0.75 if case["pre_target_action"] == "A" else 0.25
                if case["intervention"].get("active"):
                    probability_a = 1.0 - probability_a
                rows.append(
                    {
                        **case,
                        "arm": arm,
                        "pair_seed": 11,
                        "checkpoint_update": 0,
                        "probability_A": probability_a,
                        "probability_B": 1.0 - probability_a,
                        "predicted_action": "A" if probability_a >= 0.5 else "B",
                    }
                )
        effects = paired_effects(rows)
        assert effects
        assert {effect["family"] for effect in effects} == {"value", "transition"}


def test_production_trainer_runs_paired_environment_checkpoint_loop_on_cpu(
    tmp_path, monkeypatch
) -> None:
    class TinyTokenizer:
        chat_template = "present"
        pad_token_id = 0
        eos_token_id = 127
        padding_side = "right"
        truncation_side = "right"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "\n".join(message["content"] for message in messages) + "\nassistant:"

        @staticmethod
        def _encode(text):
            return [
                1 if character == "A" else 2 if character == "B" else 3 + ord(character) % 120
                for character in text
            ]

        def __call__(
            self,
            text,
            *,
            add_special_tokens,
            padding=False,
            truncation=False,
            max_length=None,
            return_tensors=None,
        ):
            assert add_special_tokens is False
            if isinstance(text, str):
                return {"input_ids": self._encode(text)}
            rows = [self._encode(value) for value in text]
            if truncation and max_length is not None:
                rows = [
                    row[-max_length:] if self.truncation_side == "left" else row[:max_length]
                    for row in rows
                ]
            width = max(map(len, rows))
            masks = []
            padded = []
            for row in rows:
                count = width - len(row)
                if self.padding_side == "left":
                    padded.append([self.pad_token_id] * count + row)
                    masks.append([0] * count + [1] * len(row))
                else:
                    padded.append(row + [self.pad_token_id] * count)
                    masks.append([1] * len(row) + [0] * count)
            assert padding is True and return_tensors == "pt"
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

        def batch_decode(self, rows, **kwargs):
            return [
                "".join("A" if int(token) == 1 else "B" if int(token) == 2 else "?" for token in row)
                for row in rows
            ]

        def save_pretrained(self, directory):
            (Path(directory) / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter_logits = torch.nn.Parameter(torch.zeros(2))
            self.config = types.SimpleNamespace(attention_dropout=0.0, use_cache=True)
            self._adapter_disabled = False

        def forward(self, *, input_ids, attention_mask, logits_to_keep):
            assert logits_to_keep == 1
            logits = torch.zeros(input_ids.shape[0], 1, 128, device=input_ids.device)
            if not self._adapter_disabled:
                logits[:, 0, 1] = self.adapter_logits[0]
                logits[:, 0, 2] = self.adapter_logits[1]
            return types.SimpleNamespace(logits=logits)

        def get_base_model(self):
            return self

        def enable_input_require_grads(self):
            return None

        def gradient_checkpointing_enable(self, **kwargs):
            return None

        @contextlib.contextmanager
        def disable_adapter(self):
            prior = self._adapter_disabled
            self._adapter_disabled = True
            try:
                yield
            finally:
                self._adapter_disabled = prior

        def save_pretrained(self, directory, *, safe_serialization):
            assert safe_serialization is True
            payload = self.adapter_logits.detach().float().cpu().numpy().tobytes()
            (Path(directory) / "adapter_model.safetensors").write_bytes(payload)
            (Path(directory) / "adapter_config.json").write_text("{}", encoding="utf-8")

        def generate(self, *, input_ids, attention_mask, max_new_tokens, **kwargs):
            assert max_new_tokens == 1
            choice = 1 if float(self.adapter_logits[0]) >= float(self.adapter_logits[1]) else 2
            completion = torch.full(
                (input_ids.shape[0], 1), choice, dtype=torch.long, device=input_ids.device
            )
            return torch.cat([input_ids, completion], dim=1)

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = lambda **kwargs: kwargs
    fake_peft.get_peft_model = lambda model, _config: model
    fake_peft.PeftModel = types.SimpleNamespace(
        from_pretrained=lambda base, checkpoint, is_trainable: base
    )
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.set_seed = lambda seed: torch.manual_seed(seed)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(bridge_training, "load_tokenizer", lambda config: TinyTokenizer())
    monkeypatch.setattr(
        bridge_training, "load_base_model", lambda config, training: TinyPolicy()
    )
    monkeypatch.setattr(
        bridge_training, "inspect_lora_target_inventory", lambda config, model: {}
    )
    monkeypatch.setattr(
        bridge_training,
        "build_model_runtime_attestation",
        lambda config, model, tokenizer, inventory: _test_runtime_attestation(),
    )
    monkeypatch.setattr(
        bridge_training,
        "verify_model_runtime_attestation",
        lambda config, attestation: dict(attestation),
    )

    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    data = tmp_path / "data"
    build_bridge_data(config, data)

    resumable_run = tmp_path / "resumable"
    monkeypatch.setenv("UE_HARD_DEADLINE_EPOCH", "1")
    with pytest.raises(BridgeRunStopped, match="soft-stop"):
        train_bridge_arm(
            config,
            load_bridge_environment(config, data),
            arm="genuine",
            pair_seed=11,
            run_dir=resumable_run,
        )
    assert (resumable_run / "STOPPED_BUDGET").is_file()
    monkeypatch.delenv("UE_HARD_DEADLINE_EPOCH")
    resumed = train_bridge_arm(
        config,
        load_bridge_environment(config, data),
        arm="genuine",
        pair_seed=11,
        run_dir=resumable_run,
        resume=True,
    )
    assert resumed == resumable_run / "final_adapter"
    assert (resumable_run / "COMPLETE").is_file()

    runs = {}
    for arm in ("genuine", "proxy"):
        run_dir = tmp_path / arm
        result = train_bridge_arm(
            config,
            load_bridge_environment(config, data),
            arm=arm,
            pair_seed=11,
            run_dir=run_dir,
        )
        assert result == run_dir / "final_adapter"
        manifest = json.loads((run_dir / "bridge_manifest.json").read_text(encoding="utf-8"))
        assert manifest["state"] == "COMPLETE"
        assert manifest["completed_updates"] == 1
        final_checkpoint = run_dir / "checkpoints" / "checkpoint-000001"
        checkpoint_manifest = json.loads(
            (final_checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        diagnostics = checkpoint_manifest["acquisition_diagnostics"]
        assert diagnostics["optimized_arm"] == arm
        assert diagnostics["overall"]["sample_count"] == 64
        assert diagnostics["cells"]["semantic"]["diagnostic_conflict"]["sample_count"] > 0
        gate_window = checkpoint_manifest["acquisition_gate_window_diagnostics"]
        assert gate_window["covered_updates"]["update_count"] == 1
        assert gate_window["overall"]["sample_count"] == 64
        runs[arm] = {
            "manifest": manifest,
            "events": [
                json.loads(line)
                for line in (run_dir / "acquisition_events.jsonl").read_text(encoding="utf-8").splitlines()
            ],
        }

    assert runs["genuine"]["manifest"]["initial_adapter_file_sha256"] == runs["proxy"][
        "manifest"
    ]["initial_adapter_file_sha256"]
    genuine_events = runs["genuine"]["events"]
    proxy_events = runs["proxy"]["events"]
    assert len(genuine_events) == len(proxy_events) == 64
    assert [
        (event["case_id"], event["action"]) for event in genuine_events
    ] == [(event["case_id"], event["action"]) for event in proxy_events]
    assert any(
        left["optimized_reward"] != right["optimized_reward"]
        for left, right in zip(genuine_events, proxy_events, strict=True)
        if left["acquisition_condition"] == "diagnostic_conflict"
    )

    def load_tiny_adapter(_config, checkpoint):
        model = TinyPolicy()
        payload = bytearray((Path(checkpoint) / "adapter_model.safetensors").read_bytes())
        values = torch.frombuffer(payload, dtype=torch.float32).clone()
        with torch.no_grad():
            model.adapter_logits.copy_(values)
        return model

    monkeypatch.setattr(bridge_evaluation, "load_tokenizer", lambda config: TinyTokenizer())
    monkeypatch.setattr(bridge_evaluation, "load_adapter_model", load_tiny_adapter)
    monkeypatch.setattr(
        bridge_evaluation, "load_base_model", lambda config, training: TinyPolicy()
    )
    monkeypatch.setattr(
        bridge_evaluation,
        "verify_model_runtime_attestation",
        lambda config, attestation: dict(attestation),
    )
    monkeypatch.setattr(
        bridge_evaluation,
        "validate_loaded_lora_runtime",
        lambda config, model, tokenizer, inventory, attestation: {
            "contract_sha256": "f" * 64
        },
    )
    monkeypatch.setattr(
        bridge_evaluation,
        "validate_loaded_base_runtime",
        lambda config, model, tokenizer, attestation: {
            "contract_sha256": "f" * 64
        },
    )
    predictions = tmp_path / "genuine-dev.jsonl"
    evaluated = evaluate_bridge_run(
        config,
        load_bridge_environment(config, data),
        run_dir=tmp_path / "genuine",
        arm="genuine",
        pair_seed=11,
        destination=predictions,
    )
    assert evaluated == predictions
    prediction_rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert {row["checkpoint_update"] for row in prediction_rows} == {0, 1}
    assert {
        row["checkpoint_acquisition_diagnostics"]["overall"]["sample_count"]
        for row in prediction_rows
    } == {0, 64}
    assert {
        row["checkpoint_acquisition_gate_window_diagnostics"]["overall"]["sample_count"]
        for row in prediction_rows
    } == {0, 64}
    assert all(row["policy_artifact"]["adapter_loaded"] is True for row in prediction_rows)
    assert all(
        row["bridge_spec_sha256"] == configured_bridge_spec_sha256(config)
        and row["bridge_spec_source"] == "loaded_config_exact"
        for row in prediction_rows
    )
    assert all(
        row["bridge_evaluation_spec"] == json.loads(
            json.dumps(BridgeEvaluationSpec.from_config(config).__dict__)
        )
        and row["bridge_evaluation_spec_sha256"]
        == configured_bridge_evaluation_spec_sha256(config)
        and row["bridge_evaluation_spec_source"] == "loaded_config_exact"
        for row in prediction_rows
    )
    assert {
        (
            row["generation_subset_attestation"]["requested_size"],
            row["generation_subset_attestation"]["available_case_count"],
            row["generation_subset_attestation"]["selected_case_count"],
        )
        for row in prediction_rows
    } == {(256, 44, 44)}
    assert all(row["generation_subset_selected"] for row in prediction_rows)

    final_checkpoint = tmp_path / "genuine" / "checkpoints" / "checkpoint-000001"
    checkpoint_manifest_path = final_checkpoint / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest["bridge_spec_sha256"] = "0" * 64
    checkpoint_manifest_path.write_text(json.dumps(checkpoint_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="bridge_spec_sha256 differs"):
        bridge_evaluation._verify_checkpoint_for_evaluation(
            config,
            load_bridge_environment(config, data),
            final_checkpoint,
            arm="genuine",
            pair_seed=11,
        )
