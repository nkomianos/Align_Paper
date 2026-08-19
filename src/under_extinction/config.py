"""Configuration loading with strict, inexpensive validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .schema import Controller, SCHEMA_VERSION


QWEN35_9B_MODEL_CONTRACT = {
    "id": "Qwen/Qwen3.5-9B",
    "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "loader_class": "Qwen3_5ForCausalLM",
    "expected_model_type": "qwen3_5_text",
    "expected_text_parameter_count": 8_953_803_264,
    "expected_layer_type_counts": {
        "linear_attention": 24,
        "full_attention": 8,
    },
    "text_only": True,
    "chat_template_kwargs": {"enable_thinking": False},
    "delta_net_kernel_policy": "torch_fallback_required",
}

QWEN35_9B_LORA_TARGETS = [
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

QWEN35_9B_LORA_TARGET_COUNTS = {
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


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    if config.get("experiment_family") == "same_environment_rl_bridge":
        validate_bridge_config(config)
    else:
        validate_config(config)
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = config_hash(config)
    return config


def config_hash(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(public, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "experiment_name", "seed", "output_root", "data", "organisms", "model", "training", "evaluation", "gates", "budget"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration missing sections: {sorted(missing)}")
    if str(config["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version {SCHEMA_VERSION}")
    controllers = config["organisms"].get("controllers", [])
    expected = {controller.value for controller in Controller}
    if set(controllers) != expected:
        raise ValueError(f"Pilot requires exactly controllers {sorted(expected)}")
    seeds = config["organisms"].get("seeds", [])
    if not seeds or len(seeds) != len(set(seeds)) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("organisms.seeds must be a non-empty unique list of integers")
    labels = config["model"].get("choice_labels")
    if labels != ["A", "B"]:
        raise ValueError("The preregistered mechanical scorer requires choice_labels: [A, B]")
    for key in (
        "train_examples",
        "dev_examples",
        "ordinary_worlds_per_renderer",
        "audit_worlds_per_renderer",
        "direct_conflict_worlds_per_renderer",
        "comprehension_worlds_per_renderer",
        "audit_comprehension_worlds_per_renderer",
    ):
        if int(config["data"].get(key, 0)) <= 0:
            raise ValueError(f"data.{key} must be positive")
    auxiliary_fraction = float(config["data"].get("auxiliary_fraction", -1))
    if not 0.0 <= auxiliary_fraction < 1.0:
        raise ValueError("data.auxiliary_fraction must lie in [0, 1)")
    if int(config["data"]["audit_comprehension_worlds_per_renderer"]) > int(config["data"]["audit_worlds_per_renderer"]):
        raise ValueError("audit comprehension worlds cannot exceed audit worlds")
    for section, keys in {
        "training": ("batch_size", "gradient_accumulation_steps", "logging_steps", "save_steps", "eval_steps", "lora_rank"),
        "evaluation": ("batch_size", "bootstrap_replicates"),
    }.items():
        for key in keys:
            if int(config[section].get(key, 0)) <= 0:
                raise ValueError(f"{section}.{key} must be positive")
    if float(config["training"].get("epochs", 0)) <= 0 or float(config["training"].get("learning_rate", 0)) <= 0:
        raise ValueError("training epochs and learning rate must be positive")
    for key, value in config["gates"].items():
        numeric = float(value)
        if key.endswith(("_min", "_max", "_margin")) and not 0.0 <= numeric <= 1.0:
            raise ValueError(f"gates.{key} must lie in [0, 1]")
    budget = config["budget"]
    if not (0 < float(budget["reserve_fraction"]) < 1):
        raise ValueError("budget.reserve_fraction must lie strictly between zero and one")
    for key in ("hourly_usd", "nominal_usd", "preflight_minutes", "soft_stop_minutes"):
        if float(budget.get(key, 0)) <= 0:
            raise ValueError(f"budget.{key} must be positive")


def validate_bridge_config(config: dict[str, Any]) -> None:
    """Validate the separate, environment-grounded bridge contract."""
    required = {
        "schema_version", "experiment_family", "experiment_name", "seed", "output_root",
        "bridge", "model", "training", "evaluation", "gates", "budget",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Bridge configuration missing sections: {sorted(missing)}")
    if config["experiment_family"] != "same_environment_rl_bridge":
        raise ValueError("Unexpected bridge experiment_family")
    bridge = config["bridge"]
    if bridge.get("environment_id") != "two_channel_choice_v1":
        raise ValueError("The frozen pilot requires bridge.environment_id=two_channel_choice_v1")
    if bridge.get("objectives") != ["genuine", "proxy"]:
        raise ValueError("Bridge objectives must be ordered [genuine, proxy]")
    if bridge.get("paired_initialization") is not True:
        raise ValueError("The paired bridge requires bridge.paired_initialization=true")
    if bridge.get("counterbalance_channel_roles") is not True:
        raise ValueError("The bridge requires channel-role counterbalancing")
    cue_regimes = bridge.get("cue_regimes")
    require_full_cue_cross = bridge.get("require_full_cue_cross")
    if cue_regimes not in (["semantic"], ["semantic", "neutral"]):
        raise ValueError(
            "bridge.cue_regimes must be frozen as [semantic] or [semantic, neutral]"
        )
    if not isinstance(require_full_cue_cross, bool):
        raise ValueError("bridge.require_full_cue_cross must be boolean")
    if require_full_cue_cross != (cue_regimes == ["semantic", "neutral"]):
        raise ValueError(
            "A full cue cross requires exactly [semantic, neutral]; semantic-only is smoke-only"
        )
    seeds = bridge.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("bridge.seeds must be a non-empty unique list")
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("Every bridge seed must be an integer")
    fractions = [float(value) for value in bridge.get("checkpoint_fractions", [])]
    if not fractions or fractions != sorted(set(fractions)) or fractions[0] != 0.0 or fractions[-1] != 1.0:
        raise ValueError("checkpoint_fractions must be unique, sorted, and include 0 and 1")
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("checkpoint_fractions must lie in [0, 1]")
    splits = bridge.get("splits", {})
    if splits != {"train": "train", "development": "dev", "locked": "test"}:
        raise ValueError("Bridge splits are frozen as train/dev/test")
    for key in ("train_worlds", "dev_worlds", "test_worlds"):
        if int(bridge.get("data", {}).get(key, 0)) <= 0:
            raise ValueError(f"bridge.data.{key} must be positive")
    if require_full_cue_cross:
        if int(bridge["data"]["train_worlds"]) < 8:
            raise ValueError("Full cue-regime acquisition requires at least eight train worlds")
        for key in ("dev_worlds", "test_worlds"):
            if int(bridge["data"][key]) < 128:
                raise ValueError(
                    f"Full cue×role×renderer×action intervention crossing requires bridge.data.{key} >= 128"
                )
    if set(bridge.get("interventions", {}).get("families", [])) != {"value", "transition"}:
        raise ValueError("Bridge intervention families must be value and transition")
    if set(bridge.get("interventions", {}).get("modes", [])) != {"switch", "no_switch", "sham"}:
        raise ValueError("Bridge intervention modes must be switch, no_switch, and sham")
    extinction = bridge.get("extinction", {})
    if extinction.get("reward_enabled") is not False or int(extinction.get("max_choices", 0)) != 1:
        raise ValueError("Bridge evaluation must be a one-choice reward-free extinction test")
    model = config["model"]
    for key, expected in QWEN35_9B_MODEL_CONTRACT.items():
        if model.get(key) != expected:
            raise ValueError(
                f"Bridge model.{key} must equal the frozen Qwen3.5-9B contract"
            )
    revision = str(model.get("revision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Bridge model revision must be an immutable 40-character lowercase SHA")
    if model.get("choice_labels") != ["A", "B"] or int(model.get("max_length", 0)) <= 0:
        raise ValueError("Bridge scorer requires positive max_length and choice_labels [A, B]")
    if model.get("dtype") != "bfloat16" or model.get("attention") != "sdpa":
        raise ValueError("The H100 bridge is frozen to bfloat16 with SDPA attention")
    training = config["training"]
    if training.get("algorithm") != "reinforce_exact_binary":
        raise ValueError("Unsupported bridge training algorithm")
    for key in (
        "updates", "rollout_batch_size", "gradient_accumulation_steps", "checkpoint_steps",
        "acquisition_gate_window_updates", "lora_rank", "lora_alpha",
    ):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if (
        int(training["lora_rank"]) != 16
        or int(training["lora_alpha"]) != 32
        or float(training.get("lora_dropout", -1.0)) != 0.0
    ):
        raise ValueError("The Qwen3.5-9B bridge freezes LoRA at rank 16, alpha 32, dropout 0")
    if int(training["acquisition_gate_window_updates"]) > int(training["updates"]):
        raise ValueError(
            "training.acquisition_gate_window_updates cannot exceed training.updates"
        )
    if int(training["rollout_batch_size"]) % int(training["gradient_accumulation_steps"]):
        raise ValueError("training.rollout_batch_size must be divisible by gradient_accumulation_steps")
    if int(bridge["data"]["train_worlds"]) < int(training["rollout_batch_size"]):
        raise ValueError("bridge.data.train_worlds must cover at least one unique rollout batch")
    for fraction in fractions:
        update = fraction * int(training["updates"])
        if abs(update - round(update)) > 1e-9:
            raise ValueError(f"checkpoint fraction {fraction} does not map to an exact optimizer update")
    for key in ("learning_rate", "max_grad_norm"):
        if float(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("warmup_ratio", "entropy_coefficient", "kl_coefficient", "lora_dropout"):
        if float(training.get(key, -1)) < 0:
            raise ValueError(f"training.{key} cannot be negative")
    if float(training["warmup_ratio"]) >= 1:
        raise ValueError("training.warmup_ratio must be less than one")
    if training.get("lora_targets") != QWEN35_9B_LORA_TARGETS:
        raise ValueError(
            "training.lora_targets must cover the frozen Qwen3.5 DeltaNet, "
            "full-attention, and MLP projections"
        )
    if training.get("expected_lora_target_counts") != QWEN35_9B_LORA_TARGET_COUNTS:
        raise ValueError("training.expected_lora_target_counts does not match Qwen3.5-9B")
    if int(training.get("expected_lora_module_count", 0)) != sum(
        QWEN35_9B_LORA_TARGET_COUNTS.values()
    ):
        raise ValueError("training.expected_lora_module_count must equal 248")
    if int(training.get("expected_lora_trainable_parameter_count", 0)) != 43_278_336:
        raise ValueError(
            "training.expected_lora_trainable_parameter_count does not match rank-16 Qwen3.5-9B"
        )
    if not 0.0 <= float(training["lora_dropout"]) < 1.0:
        raise ValueError("training.lora_dropout must lie in [0, 1)")
    if not isinstance(training.get("normalize_advantages"), bool):
        raise ValueError("training.normalize_advantages must be boolean")
    evaluation = config["evaluation"]
    for key in (
        "batch_size", "generation_subset_size", "generation_batch_size",
        "max_new_tokens", "bootstrap_replicates",
    ):
        if int(evaluation.get(key, 0)) <= 0:
            raise ValueError(f"evaluation.{key} must be positive")
    if int(evaluation["max_new_tokens"]) != 1:
        raise ValueError("evaluation.max_new_tokens must equal one for the first-action assay")
    for key in ("minimum_legal_choice_mass", "minimum_exact_parse_rate"):
        value = evaluation.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"evaluation.{key} must lie in [0, 1]")
    gates = config["gates"]
    if set(gates) != {"smoke", "stage1", "replication"}:
        raise ValueError("Bridge gates must define smoke, stage1, and replication")
    for gate_name, gate in gates.items():
        if not isinstance(gate, dict):
            raise ValueError(f"gates.{gate_name} must be a mapping")
        for key, value in gate.items():
            if key == "finite_training_loss":
                if not isinstance(value, bool):
                    raise ValueError("gates.smoke.finite_training_loss must be boolean")
            elif key == "paired_run_count":
                if int(value) <= 0:
                    raise ValueError("gates.smoke.paired_run_count must be positive")
            elif not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"gates.{gate_name}.{key} must lie in [0, 1]")
    if require_full_cue_cross:
        unpooled_gate_keys = {
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
        for gate_name in ("stage1", "replication"):
            missing_unpooled = unpooled_gate_keys - set(gates[gate_name])
            if missing_unpooled:
                raise ValueError(
                    f"gates.{gate_name} lacks unpooled cue-cell thresholds: "
                    f"{sorted(missing_unpooled)}"
                )
    stage1_gate = gates["stage1"]
    expected_legal_mass = float(stage1_gate.get("legal_choice_mass_min", 0.05))
    expected_parse_rate = 1.0 - float(stage1_gate.get("invalid_choice_rate_max", 0.05))
    if abs(float(evaluation["minimum_legal_choice_mass"]) - expected_legal_mass) > 1e-12:
        raise ValueError(
            "evaluation.minimum_legal_choice_mass must equal the Stage 1 legal-choice gate"
        )
    if abs(float(evaluation["minimum_exact_parse_rate"]) - expected_parse_rate) > 1e-12:
        raise ValueError(
            "evaluation.minimum_exact_parse_rate must equal one minus the Stage 1 invalid-choice gate"
        )
    budget = config["budget"]
    for key in (
        "hourly_usd", "nominal_usd", "retrieval_reserve_minutes", "soft_stop_minutes",
    ):
        if float(budget.get(key, 0)) <= 0:
            raise ValueError(f"budget.{key} must be positive")
    command_ceiling_keys = [key for key in budget if key.endswith("_minutes_per_objective")]
    if not command_ceiling_keys:
        raise ValueError("Bridge budget must define per-objective command ceilings")
    for key in command_ceiling_keys:
        if float(budget[key]) <= 0:
            raise ValueError(f"budget.{key} must be positive")
    if int(budget["retrieval_reserve_minutes"]) < 30:
        raise ValueError("The paid bridge requires at least 30 minutes reserved for retrieval")
    if not 0.0 < float(budget.get("reserve_fraction", 0)) < 1.0:
        raise ValueError("budget.reserve_fraction must lie strictly between zero and one")
    # The one-update smoke config is only the measurement instrument.  Any
    # multi-update scientific config must freeze the conservative conversion
    # from that measurement into a Stage-1 time/cost authorization.
    if int(training["updates"]) > 1:
        if float(budget.get("stage1_control_plane_minutes", 0)) <= 0:
            raise ValueError("budget.stage1_control_plane_minutes must be positive")
        projection_margin = float(
            budget.get("preflight_projection_margin_fraction", 0)
        )
        if not 0.0 < projection_margin < 1.0:
            raise ValueError(
                "budget.preflight_projection_margin_fraction must lie strictly "
                "between zero and one"
            )
        profile_path = budget.get("preflight_workload_profile_path")
        if not isinstance(profile_path, str) or not profile_path.strip():
            raise ValueError(
                "budget.preflight_workload_profile_path must name the frozen "
                "local token-workload attestation"
            )
        peak_fraction = float(budget.get("preflight_max_peak_vram_fraction", 0))
        if not 0.0 < peak_fraction < 1.0:
            raise ValueError(
                "budget.preflight_max_peak_vram_fraction must lie strictly "
                "between zero and one"
            )


def output_root(config: dict[str, Any]) -> Path:
    base = Path(config["_config_path"]).parent.parent
    configured = Path(config["output_root"])
    return configured if configured.is_absolute() else (base / configured).resolve()
