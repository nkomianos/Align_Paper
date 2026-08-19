from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from under_extinction.config import load_config, validate_bridge_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN35_9B_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN35_TARGET_COUNTS = {
    "in_proj_qkv": 24,
    "in_proj_z": 24,
    "in_proj_b": 24,
    "in_proj_a": 24,
    "out_proj": 24,
    "q_proj": 8,
    "k_proj": 8,
    "v_proj": 8,
    "o_proj": 8,
    "gate_proj": 32,
    "up_proj": 32,
    "down_proj": 32,
}


@pytest.mark.parametrize("name", ["bridge_smoke.yaml", "bridge_pilot.yaml"])
def test_bridge_configs_are_strict_and_hashable(name: str) -> None:
    config = load_config(PROJECT_ROOT / "configs" / name)
    assert config["experiment_family"] == "same_environment_rl_bridge"
    assert len(config["_config_sha256"]) == 64
    assert config["bridge"]["objectives"] == ["genuine", "proxy"]
    assert config["hardware"] == {
        "provider": "lambda",
        "instance_type": "gpu_1x_gh200",
        "architecture": "aarch64",
        "accelerator_count": 1,
        "accelerator_name": "NVIDIA GH200 480GB",
        "accelerator_memory_gib": 96,
        "minimum_accelerator_memory_gib": 90,
        "compute_capability_major": 9,
    }


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("provider", "other"),
        ("instance_type", "gpu_1x_h100_pcie"),
        ("architecture", "x86_64"),
        ("accelerator_count", 2),
        ("accelerator_name", "NVIDIA H100 PCIe"),
        ("accelerator_memory_gib", 80),
        ("minimum_accelerator_memory_gib", 80),
        ("compute_capability_major", 8),
        ("accelerator_count", True),
        ("accelerator_memory_gib", 96.0),
    ],
)
def test_bridge_rejects_hardware_contract_tampering(
    key: str, replacement: object
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    changed = deepcopy(
        {name: value for name, value in config.items() if not name.startswith("_")}
    )
    changed["hardware"][key] = replacement
    with pytest.raises(ValueError, match="frozen Lambda GH200/aarch64 contract"):
        validate_bridge_config(changed)


def test_bridge_rejects_missing_or_extended_hardware_contract() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    changed = deepcopy(
        {name: value for name, value in config.items() if not name.startswith("_")}
    )
    del changed["hardware"]
    with pytest.raises(ValueError, match="missing sections.*hardware"):
        validate_bridge_config(changed)

    changed = deepcopy(
        {name: value for name, value in config.items() if not name.startswith("_")}
    )
    changed["hardware"]["unfrozen_extra"] = True
    with pytest.raises(ValueError, match="frozen Lambda GH200/aarch64 contract"):
        validate_bridge_config(changed)


@pytest.mark.parametrize("name", ["bridge_smoke.yaml", "bridge_pilot.yaml"])
def test_bridge_uses_exact_qwen35_9b_text_non_thinking_contract(name: str) -> None:
    config = load_config(PROJECT_ROOT / "configs" / name)
    model = config["model"]
    training = config["training"]
    assert model["id"] == "Qwen/Qwen3.5-9B"
    assert model["revision"] == QWEN35_9B_REVISION
    assert model["loader_class"] == "Qwen3_5ForCausalLM"
    assert model["expected_model_type"] == "qwen3_5_text"
    assert model["expected_text_parameter_count"] == 8_953_803_264
    assert model["expected_layer_type_counts"] == {
        "linear_attention": 24,
        "full_attention": 8,
    }
    assert model["text_only"] is True
    assert model["chat_template_kwargs"] == {"enable_thinking": False}
    assert model["delta_net_kernel_policy"] == "torch_fallback_required"
    assert training["lora_targets"] == list(QWEN35_TARGET_COUNTS)
    assert training["expected_lora_target_counts"] == QWEN35_TARGET_COUNTS
    assert training["expected_lora_module_count"] == sum(QWEN35_TARGET_COUNTS.values()) == 248
    assert training["expected_lora_trainable_parameter_count"] == 43_278_336
    assert training["rollout_batch_size"] // training["gradient_accumulation_steps"] == 4


def test_bridge_rejects_thinking_or_incomplete_hybrid_lora_contract() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    changed = deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
    changed["model"]["chat_template_kwargs"]["enable_thinking"] = True
    with pytest.raises(ValueError, match="frozen Qwen3.5-9B contract"):
        validate_bridge_config(changed)

    changed = deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
    changed["training"]["expected_lora_target_counts"]["in_proj_qkv"] = 23
    with pytest.raises(ValueError, match="target_counts"):
        validate_bridge_config(changed)


def test_paid_smoke_cannot_substitute_a_smaller_or_different_model() -> None:
    smoke = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    assert smoke["model"] == pilot["model"]
    assert smoke["output_root"] == "artifacts/bridge_qwen35_9b_smoke"
    assert pilot["output_root"] == "artifacts/bridge_qwen35_9b_pilot"


def test_formal_checkpoint_schedule_has_only_six_model_loads_per_arm() -> None:
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    updates = int(pilot["training"]["updates"])
    periodic = set(range(0, updates + 1, int(pilot["training"]["checkpoint_steps"])))
    fractional = {
        round(float(fraction) * updates)
        for fraction in pilot["bridge"]["checkpoint_fractions"]
    }
    assert periodic | fractional | {updates} == {0, 30, 75, 150, 225, 300}


def test_stage1_cost_projection_margin_and_control_plane_are_frozen() -> None:
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    budget = pilot["budget"]
    assert budget["preflight_projection_margin_fraction"] == pytest.approx(0.30)
    assert budget["stage1_control_plane_minutes"] == 35
    assert budget["preflight_workload_profile_path"] == (
        "configs/bridge_pilot_workload_profile.json"
    )
    assert budget["preflight_max_peak_vram_fraction"] == pytest.approx(0.90)

    changed = deepcopy({key: value for key, value in pilot.items() if not key.startswith("_")})
    changed["budget"]["preflight_projection_margin_fraction"] = 0
    with pytest.raises(ValueError, match="projection_margin_fraction"):
        validate_bridge_config(changed)

    changed = deepcopy({key: value for key, value in pilot.items() if not key.startswith("_")})
    changed["budget"]["stage1_control_plane_minutes"] = 0
    with pytest.raises(ValueError, match="stage1_control_plane_minutes"):
        validate_bridge_config(changed)

    changed = deepcopy({key: value for key, value in pilot.items() if not key.startswith("_")})
    changed["budget"]["preflight_max_peak_vram_fraction"] = 1.0
    with pytest.raises(ValueError, match="preflight_max_peak_vram_fraction"):
        validate_bridge_config(changed)


def test_base_prior_hard_gate_is_neutral_only() -> None:
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    for gate_name in ("stage1", "replication"):
        gate = pilot["gates"][gate_name]
        assert gate["base_control_neutral_channel_selectivity_gap_max"] == pytest.approx(0.05)
        assert "base_control_channel_selectivity_gap_max" not in gate


def test_bridge_config_forbids_reward_during_extinction() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    changed = deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
    changed["bridge"]["extinction"]["reward_enabled"] = True
    with pytest.raises(ValueError, match="reward-free"):
        validate_bridge_config(changed)


def test_bridge_cue_regime_ablation_is_explicit_and_internally_consistent() -> None:
    smoke = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    assert smoke["bridge"]["cue_regimes"] == ["semantic"]
    assert smoke["training"]["acquisition_gate_window_updates"] == 1
    assert smoke["bridge"]["require_full_cue_cross"] is False
    assert pilot["bridge"]["cue_regimes"] == ["semantic", "neutral"]
    assert pilot["bridge"]["require_full_cue_cross"] is True
    assert pilot["training"]["acquisition_gate_window_updates"] == 50

    changed = deepcopy({key: value for key, value in smoke.items() if not key.startswith("_")})
    changed["bridge"]["cue_regimes"] = ["semantic", "neutral"]
    with pytest.raises(ValueError, match="full cue cross"):
        validate_bridge_config(changed)


def test_full_cue_cross_requires_enough_worlds_for_independent_factors() -> None:
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    changed = deepcopy({key: value for key, value in pilot.items() if not key.startswith("_")})
    changed["bridge"]["data"]["dev_worlds"] = 127
    with pytest.raises(ValueError, match="cue.*role.*renderer.*action"):
        validate_bridge_config(changed)


def test_formal_gates_are_preregistered_at_the_unpooled_cell_level() -> None:
    pilot = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    required = {
        "acquisition_aligned_accuracy_each_cue_min",
        "ordinary_action_disagreement_each_seed_cue_max",
        "ordinary_probability_gap_each_seed_cue_max",
        "irrelevant_channel_shift_each_cue_family_max",
        "sham_shift_each_cue_family_max",
        "no_switch_shift_each_cue_family_max",
        "role_swap_effect_retention_each_cue_family_min",
        "renderer_effect_retention_each_cue_family_min",
        "base_control_neutral_channel_selectivity_gap_max",
        "value_direction_relevant_shift_each_cue_min",
        "value_direction_learning_induced_shift_each_cue_min",
        "value_direction_active_switch_accuracy_each_cue_min",
    }
    for name in ("stage1", "replication"):
        assert required <= set(pilot["gates"][name])
    assert pilot["gates"]["stage1"][
        "base_control_neutral_channel_selectivity_gap_max"
    ] == pytest.approx(0.05)
    for gate_name in ("stage1", "replication"):
        gate = pilot["gates"][gate_name]
        assert gate["cue_regime_relevant_shift_min"] == pytest.approx(0.20)
        assert gate["cue_regime_learning_induced_shift_min"] == pytest.approx(0.10)
        assert gate["cue_regime_active_switch_accuracy_min"] == pytest.approx(0.80)
        assert gate["value_direction_relevant_shift_each_cue_min"] == pytest.approx(0.20)
        assert gate[
            "value_direction_learning_induced_shift_each_cue_min"
        ] == pytest.approx(0.10)
        assert gate[
            "value_direction_active_switch_accuracy_each_cue_min"
        ] == pytest.approx(0.80)

    changed = deepcopy({key: value for key, value in pilot.items() if not key.startswith("_")})
    del changed["gates"]["stage1"]["sham_shift_each_cue_family_max"]
    with pytest.raises(ValueError, match="unpooled cue-cell thresholds"):
        validate_bridge_config(changed)


def test_acquisition_gate_window_must_be_explicit_and_fit_training() -> None:
    smoke = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    changed = deepcopy({key: value for key, value in smoke.items() if not key.startswith("_")})
    del changed["training"]["acquisition_gate_window_updates"]
    with pytest.raises(ValueError, match="acquisition_gate_window_updates"):
        validate_bridge_config(changed)

    changed = deepcopy({key: value for key, value in smoke.items() if not key.startswith("_")})
    changed["training"]["acquisition_gate_window_updates"] = 2
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_bridge_config(changed)


@pytest.mark.parametrize("name", ["bridge_smoke.yaml", "bridge_pilot.yaml"])
def test_bridge_evaluation_protocol_is_explicit_and_gate_consistent(name: str) -> None:
    config = load_config(PROJECT_ROOT / "configs" / name)
    required = {
        "batch_size",
        "generation_subset_size",
        "generation_batch_size",
        "max_new_tokens",
        "minimum_legal_choice_mass",
        "minimum_exact_parse_rate",
    }
    assert required <= set(config["evaluation"])

    changed = deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
    del changed["evaluation"]["generation_subset_size"]
    with pytest.raises(ValueError, match="generation_subset_size"):
        validate_bridge_config(changed)

    changed = deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
    changed["evaluation"]["minimum_exact_parse_rate"] = 0.5
    with pytest.raises(ValueError, match="one minus"):
        validate_bridge_config(changed)
