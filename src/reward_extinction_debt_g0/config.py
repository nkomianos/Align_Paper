"""Frozen configuration validation."""

from __future__ import annotations

from typing import Any, Mapping

import yaml

from . import SCHEMA_VERSION
from .corpus import CONTEXT_IDS, DOMAIN_IDS


FROZEN_STATUS = "frozen_cpu_audited_test_locked_awaiting_explicit_gpu_authorization"


def load_config_bytes(payload: bytes) -> dict[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("config is not valid UTF-8 YAML") from exc
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping")
    validate_config(value)
    return value


def validate_config(cfg: Mapping[str, Any]) -> None:
    required = {
        "kind", "schema_version", "status", "claim_scope", "model", "design",
        "training", "selection", "thresholds", "environment", "integrity",
    }
    if set(cfg) != required:
        raise ValueError("config top-level fields differ from frozen schema")
    if cfg["kind"] != "reward_extinction_debt_g0" or cfg["schema_version"] != SCHEMA_VERSION:
        raise ValueError("wrong config kind or schema")
    if cfg["status"] != FROZEN_STATUS:
        raise ValueError("config is not frozen and TEST locked")
    design = cfg["design"]
    if tuple(design["domains"]) != DOMAIN_IDS or tuple(design["contexts"]) != CONTEXT_IDS:
        raise ValueError("domain/context crossing changed")
    if list(design["reacquisition_doses"]) != [0, 4, 16, 64]:
        raise ValueError("reacquisition doses changed")
    if list(design["conditions"]) != ["clean", "ordinary_aligned", "reactivation_counterconditioned"]:
        raise ValueError("condition set changed")
    if not design["test_locked"] or not design["runner_blind_to_test_oracle"]:
        raise ValueError("TEST lock is disabled")
    training = cfg["training"]
    if training["objective"] != "exact_two_action_contextual_bandit_policy_gradient":
        raise ValueError("training objective changed")
    if training["choices"] != ["A", "B"]:
        raise ValueError("legal action set changed")
    if int(training["lora_rank"]) <= 0 or int(training["micro_batch_size"]) <= 0:
        raise ValueError("invalid LoRA or batch configuration")
    if list(cfg["selection"]["alignment_checkpoint_steps"]) != sorted(cfg["selection"]["alignment_checkpoint_steps"]):
        raise ValueError("alignment checkpoint steps must be sorted")
    if int(cfg["thresholds"]["bootstrap_replicates"]) < 100:
        raise ValueError("too few bootstrap replicates")
    if cfg["integrity"]["code_tree_sha256"] == "__PINNED_CODE_TREE_SHA256__":
        raise ValueError("code tree hash was not pinned")
