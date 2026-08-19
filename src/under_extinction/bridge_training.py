"""Environment-grounded paired LoRA policy-gradient acquisition.

This module deliberately does not depend on the symbolic ledger generator.  A bridge
environment supplies observations, receives sampled actions, and returns only the
outcomes realized by those actions.  The optimizer then uses either genuine (G) or
proxy (P) reward.  No controller label or supervised target action is accepted.

Expected environment protocol
-----------------------------
``acquisition_batch`` returns mappings with ``case_id`` and chat ``messages``.
``transition_batch`` consumes those same cases plus sampled A/B actions and returns
one mapping per case with ``case_id``, ``genuine_reward``, and ``proxy_reward``.
It may update internal history so later observations contain experienced outcomes.
``state_dict`` and ``load_state_dict`` must capture that history. ``provenance`` must
identify the environment implementation and frozen task specification.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .config import config_hash, output_root
from .io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl
from .manifest import environment_snapshot, project_hash
from .modeling import (
    build_model_runtime_attestation,
    chat_prompt_text,
    inspect_lora_target_inventory,
    load_base_model,
    load_tokenizer,
    verify_model_runtime_attestation,
)


CHOICES = ("A", "B")
ARM_REWARD_FIELD = {"genuine": "genuine_reward", "proxy": "proxy_reward"}
ACQUISITION_CONDITIONS = ("aligned", "diagnostic_conflict")
ACQUISITION_CUE_REGIMES = ("semantic", "neutral")
FORBIDDEN_ACQUISITION_KEYS = {
    "controller",
    "controller_label",
    "oracle_actions",
    "target_action",
    "supervised_label",
}


class BridgeRunStopped(RuntimeError):
    """A bridge run stopped at a resumable checkpoint."""


@runtime_checkable
class BridgeEnvironment(Protocol):
    """Minimal stateful environment interface used by the acquisition loop."""

    def provenance(self) -> Mapping[str, Any]: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def acquisition_batch(
        self, *, trajectory_seed: int, update_index: int, batch_size: int
    ) -> Sequence[Mapping[str, Any]]: ...

    def transition_batch(
        self, cases: Sequence[Mapping[str, Any]], actions: Sequence[str]
    ) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class BridgeTrainingSpec:
    """Frozen optimizer semantics for the compact bridge pilot.

    ``updates`` counts optimizer steps. ``batch_size`` is the total rollout batch
    per optimizer step and is divided into ``gradient_accumulation_steps`` equal
    microbatches. Checkpoint fractions are analysis checkpoints; the periodic
    interval is an additional resilience checkpoint schedule.
    """

    updates: int = 200
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    entropy_coefficient: float = 0.01
    kl_coefficient: float = 0.0
    normalize_advantages: bool = True
    baseline_momentum: float = 0.95
    max_grad_norm: float = 1.0
    checkpoint_every: int = 25
    checkpoint_fractions: tuple[float, ...] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.updates <= 0 or self.batch_size <= 0 or self.checkpoint_every <= 0:
            raise ValueError("updates, batch_size, and checkpoint_every must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.batch_size % self.gradient_accumulation_steps:
            raise ValueError("batch_size must be divisible by gradient_accumulation_steps")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.weight_decay < 0 or self.entropy_coefficient < 0 or self.kl_coefficient < 0:
            raise ValueError("weight decay, entropy, and KL coefficients cannot be negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must lie in [0, 1)")
        if not isinstance(self.normalize_advantages, bool):
            raise TypeError("normalize_advantages must be boolean")
        if not 0.0 <= self.baseline_momentum < 1.0:
            raise ValueError("baseline_momentum must lie in [0, 1)")
        if not self.checkpoint_fractions:
            raise ValueError("checkpoint_fractions cannot be empty")
        for fraction in self.checkpoint_fractions:
            if not 0.0 <= float(fraction) <= 1.0:
                raise ValueError("checkpoint fractions must lie in [0, 1]")
            exact_update = float(fraction) * self.updates
            if not math.isclose(exact_update, round(exact_update), abs_tol=1e-9):
                raise ValueError(
                    f"Checkpoint fraction {fraction} does not map to an exact optimizer update"
                )

    @property
    def microbatch_size(self) -> int:
        return self.batch_size // self.gradient_accumulation_steps

    @property
    def checkpoint_updates(self) -> tuple[int, ...]:
        periodic = range(self.checkpoint_every, self.updates + 1, self.checkpoint_every)
        analysis = (round(float(fraction) * self.updates) for fraction in self.checkpoint_fractions)
        return tuple(sorted({0, self.updates, *periodic, *analysis}))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "BridgeTrainingSpec":
        if values is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown bridge-training settings: {sorted(unknown)}")
        converted = dict(values)
        if "checkpoint_fractions" in converted:
            converted["checkpoint_fractions"] = tuple(converted["checkpoint_fractions"])
        return cls(**converted)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
    ) -> "BridgeTrainingSpec":
        """Consume every preregistered optimizer knob instead of silently ignoring it."""
        training = config.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("Bridge config requires a training mapping")
        algorithm = training.get("algorithm", "reinforce_exact_binary")
        if algorithm != "reinforce_exact_binary":
            raise ValueError(f"Unsupported bridge training algorithm: {algorithm!r}")
        required = {
            "updates",
            "rollout_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_ratio",
            "entropy_coefficient",
            "kl_coefficient",
            "normalize_advantages",
            "max_grad_norm",
            "checkpoint_steps",
        }
        missing = required - set(training)
        if missing:
            raise ValueError(f"Bridge training config lacks settings: {sorted(missing)}")
        bridge = config.get("bridge")
        if not isinstance(bridge, Mapping) or "checkpoint_fractions" not in bridge:
            raise ValueError("Bridge config requires bridge.checkpoint_fractions")
        values: dict[str, Any] = {
            "updates": int(training["updates"]),
            "batch_size": int(training["rollout_batch_size"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training.get("weight_decay", 0.0)),
            "warmup_ratio": float(training["warmup_ratio"]),
            "entropy_coefficient": float(training["entropy_coefficient"]),
            "kl_coefficient": float(training["kl_coefficient"]),
            "normalize_advantages": training["normalize_advantages"],
            "baseline_momentum": float(training.get("baseline_momentum", 0.95)),
            "max_grad_norm": float(training["max_grad_norm"]),
            "checkpoint_every": int(training["checkpoint_steps"]),
            "checkpoint_fractions": tuple(float(value) for value in bridge["checkpoint_fractions"]),
        }
        if overrides:
            aliases = {
                "rollout_batch_size": "batch_size",
                "checkpoint_steps": "checkpoint_every",
            }
            for key, value in overrides.items():
                canonical = aliases.get(key, key)
                if canonical not in cls.__dataclass_fields__:
                    raise ValueError(f"Unknown bridge-training setting: {key}")
                values[canonical] = value
        return cls.from_mapping(values)


def canonical_arm(arm: str) -> str:
    normalized = arm.strip().lower()
    aliases = {"g": "genuine", "genuine": "genuine", "p": "proxy", "proxy": "proxy"}
    if normalized not in aliases:
        raise ValueError("Bridge arm must be G/genuine or P/proxy")
    return aliases[normalized]


def _stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _messages_are_valid(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and bool(messages)
        and all(
            isinstance(message, dict)
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            for message in messages
        )
        and messages[-1].get("role") == "user"
    )


def validate_acquisition_batch(
    cases: Sequence[Mapping[str, Any]], expected_size: int
) -> list[Mapping[str, Any]]:
    """Reject label leakage before a paid forward pass."""
    collected = list(cases)
    if len(collected) != expected_size:
        raise ValueError(f"Environment returned {len(collected)} cases; expected {expected_size}")
    seen: set[str] = set()
    for case in collected:
        if not isinstance(case, Mapping):
            raise TypeError("Each acquisition case must be a mapping")
        forbidden = FORBIDDEN_ACQUISITION_KEYS & set(case)
        if forbidden:
            raise ValueError(f"Acquisition case contains supervised/controller fields: {sorted(forbidden)}")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Each acquisition case needs a non-empty string case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate acquisition case_id within a batch: {case_id}")
        seen.add(case_id)
        if not _messages_are_valid(case.get("messages")):
            raise ValueError(f"Invalid chat messages for acquisition case {case_id}")
    return collected


def validate_realized_outcomes(
    cases: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Validate selected-action outcomes; counterfactual rewards are never requested."""
    collected = list(outcomes)
    if len(collected) != len(cases):
        raise ValueError("Environment returned the wrong number of realized outcomes")
    for case, outcome in zip(cases, collected, strict=True):
        if outcome.get("case_id") != case["case_id"]:
            raise ValueError("Environment outcome order/case_id does not match the selected action batch")
        for field in ARM_REWARD_FIELD.values():
            try:
                value = float(outcome[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Outcome {case['case_id']} lacks numeric {field}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Outcome {case['case_id']} has non-finite {field}")
    return collected


def selected_arm_rewards(
    outcomes: Sequence[Mapping[str, Any]], arm: str, *, device: Any | None = None
) -> Any:
    """Build the only reward tensor allowed to enter the policy-gradient loss."""
    import torch

    field = ARM_REWARD_FIELD[canonical_arm(arm)]
    values = [float(outcome[field]) for outcome in outcomes]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite realized reward")
    return torch.tensor(values, dtype=torch.float32, device=device)


def initialize_acquisition_diagnostics(
    arm: str, cue_regimes: Sequence[str]
) -> dict[str, Any]:
    """Create the exact cumulative sufficient statistics saved at every checkpoint."""
    selected_arm = canonical_arm(arm)
    cues = tuple(str(value) for value in cue_regimes)
    if not cues or len(set(cues)) != len(cues) or not set(cues) <= set(ACQUISITION_CUE_REGIMES):
        raise ValueError("Acquisition cue regimes must be a non-empty unique semantic/neutral subset")
    return {
        "schema_version": "1.0",
        "optimized_arm": selected_arm,
        "cue_regimes": list(cues),
        "success_definition": "optimized_realized_reward_equals_1",
        "cells": {
            cue: {
                condition: {"count": 0, "reward_sum": 0.0, "success_count": 0}
                for condition in ACQUISITION_CONDITIONS
            }
            for cue in cues
        },
    }


def validate_acquisition_diagnostics_state(
    state: Mapping[str, Any], *, arm: str, cue_regimes: Sequence[str]
) -> dict[str, Any]:
    """Return a JSON copy after checking the resume-critical accumulator schema."""
    expected = initialize_acquisition_diagnostics(arm, cue_regimes)
    copied = json.loads(canonical_json(dict(state)))
    for key in ("schema_version", "optimized_arm", "cue_regimes", "success_definition"):
        if copied.get(key) != expected[key]:
            raise ValueError(f"Acquisition diagnostics identity mismatch for {key}")
    cells = copied.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(expected["cells"]):
        raise ValueError("Acquisition diagnostics have the wrong cue-regime cells")
    for cue, expected_conditions in expected["cells"].items():
        conditions = cells.get(cue)
        if not isinstance(conditions, Mapping) or set(conditions) != set(expected_conditions):
            raise ValueError(f"Acquisition diagnostics have the wrong conditions for {cue}")
        for condition, values in conditions.items():
            if not isinstance(values, Mapping) or set(values) != {
                "count", "reward_sum", "success_count",
            }:
                raise ValueError(f"Invalid acquisition diagnostic cell {cue}/{condition}")
            count = values["count"]
            success_count = values["success_count"]
            reward_sum = values["reward_sum"]
            if type(count) is not int or count < 0:
                raise ValueError(f"Invalid acquisition count for {cue}/{condition}")
            if type(success_count) is not int or not 0 <= success_count <= count:
                raise ValueError(f"Invalid acquisition success count for {cue}/{condition}")
            if not isinstance(reward_sum, (int, float)) or isinstance(reward_sum, bool):
                raise ValueError(f"Invalid acquisition reward sum for {cue}/{condition}")
            if not math.isfinite(float(reward_sum)) or not -1e-8 <= float(reward_sum) <= count + 1e-8:
                raise ValueError(f"Out-of-range acquisition reward sum for {cue}/{condition}")
            if not math.isclose(float(reward_sum), float(success_count), abs_tol=1e-8):
                raise ValueError(
                    "Binary bridge reward sum and optimal-action success count disagree for "
                    f"{cue}/{condition}"
                )
    return copied


def update_acquisition_diagnostics(
    state: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Accumulate realized selected-arm reward only after a successful optimizer step."""
    arm = str(state.get("optimized_arm", ""))
    cues = list(state.get("cue_regimes") or [])
    updated = validate_acquisition_diagnostics_state(state, arm=arm, cue_regimes=cues)
    if len(cases) != len(outcomes):
        raise ValueError("Acquisition diagnostics received mismatched cases/outcomes")
    observations: list[tuple[str, str, float]] = []
    reward_field = ARM_REWARD_FIELD[canonical_arm(arm)]
    for case, outcome in zip(cases, outcomes, strict=True):
        if outcome.get("case_id") != case.get("case_id"):
            raise ValueError("Acquisition diagnostic case/outcome identity mismatch")
        cue = str(case.get("cue_regime", ""))
        condition = str(case.get("condition", ""))
        if cue not in updated["cells"]:
            raise ValueError(f"Unregistered acquisition cue regime: {cue!r}")
        if condition not in ACQUISITION_CONDITIONS:
            raise ValueError(f"Unregistered acquisition condition: {condition!r}")
        reward = float(outcome[reward_field])
        if not math.isfinite(reward) or not (
            math.isclose(reward, 0.0, abs_tol=1e-8)
            or math.isclose(reward, 1.0, abs_tol=1e-8)
        ):
            raise ValueError("Acquisition optimal-action accuracy requires frozen binary rewards")
        observations.append((cue, condition, 1.0 if reward > 0.5 else 0.0))
    for cue, condition, reward in observations:
        cell = updated["cells"][cue][condition]
        cell["count"] += 1
        cell["reward_sum"] += reward
        cell["success_count"] += int(reward == 1.0)
    return validate_acquisition_diagnostics_state(updated, arm=arm, cue_regimes=cues)


def acquisition_diagnostics_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize cumulative reward and optimal-action rates without hiding empty cells."""
    arm = str(state.get("optimized_arm", ""))
    cues = list(state.get("cue_regimes") or [])
    validated = validate_acquisition_diagnostics_state(state, arm=arm, cue_regimes=cues)

    def summarize(raw_cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = sum(int(cell["count"]) for cell in raw_cells)
        reward_sum = sum(float(cell["reward_sum"]) for cell in raw_cells)
        successes = sum(int(cell["success_count"]) for cell in raw_cells)
        return {
            "sample_count": count,
            "optimized_reward_sum": reward_sum,
            "optimal_action_count": successes,
            "optimized_reward_mean": reward_sum / count if count else None,
            "optimal_action_accuracy": successes / count if count else None,
        }

    cells = {
        cue: {
            condition: summarize([validated["cells"][cue][condition]])
            for condition in ACQUISITION_CONDITIONS
        }
        for cue in cues
    }
    return {
        "schema_version": validated["schema_version"],
        "diagnostic_scope": "cumulative_all_optimizer_updates",
        "optimized_arm": arm,
        "cue_regimes": cues,
        "success_definition": validated["success_definition"],
        "cells": cells,
        "by_cue_regime": {
            cue: summarize(list(validated["cells"][cue].values())) for cue in cues
        },
        "by_condition": {
            condition: summarize(
                [validated["cells"][cue][condition] for cue in cues]
            )
            for condition in ACQUISITION_CONDITIONS
        },
        "overall": summarize(
            [
                validated["cells"][cue][condition]
                for cue in cues
                for condition in ACQUISITION_CONDITIONS
            ]
        ),
    }


def initialize_acquisition_gate_window_diagnostics(
    arm: str,
    cue_regimes: Sequence[str],
    *,
    window_updates: int,
    samples_per_update: int,
) -> dict[str, Any]:
    """Create the preregistered, resume-safe trailing acquisition window.

    The state retains exact sufficient statistics for each of the last
    ``window_updates`` *optimizer* updates.  Keeping per-update cells, rather than
    subtracting from a cumulative total, makes resume and checkpoint validation
    deterministic and lets the analysis prove the claimed window coverage.
    """
    template = initialize_acquisition_diagnostics(arm, cue_regimes)
    if type(window_updates) is not int or window_updates <= 0:
        raise ValueError("Acquisition gate window_updates must be a positive integer")
    if type(samples_per_update) is not int or samples_per_update <= 0:
        raise ValueError("Acquisition gate samples_per_update must be a positive integer")
    return {
        "schema_version": "1.0",
        "diagnostic_scope": "trailing_optimizer_updates",
        "optimized_arm": template["optimized_arm"],
        "cue_regimes": template["cue_regimes"],
        "success_definition": template["success_definition"],
        "window_updates": window_updates,
        "samples_per_update": samples_per_update,
        "completed_updates": 0,
        "updates": [],
    }


def validate_acquisition_gate_window_state(
    state: Mapping[str, Any],
    *,
    arm: str,
    cue_regimes: Sequence[str],
    window_updates: int,
    samples_per_update: int,
) -> dict[str, Any]:
    """Validate every retained update and its exact cue×condition coverage total."""
    expected = initialize_acquisition_gate_window_diagnostics(
        arm,
        cue_regimes,
        window_updates=window_updates,
        samples_per_update=samples_per_update,
    )
    copied = json.loads(canonical_json(dict(state)))
    identity_keys = {
        "schema_version",
        "diagnostic_scope",
        "optimized_arm",
        "cue_regimes",
        "success_definition",
        "window_updates",
        "samples_per_update",
    }
    if set(copied) != identity_keys | {"completed_updates", "updates"}:
        raise ValueError("Acquisition gate window has an invalid state schema")
    for key in identity_keys:
        if copied.get(key) != expected[key]:
            raise ValueError(f"Acquisition gate window identity mismatch for {key}")
    completed = copied.get("completed_updates")
    updates = copied.get("updates")
    if type(completed) is not int or completed < 0 or not isinstance(updates, list):
        raise ValueError("Acquisition gate window has invalid update bookkeeping")
    expected_retained = min(completed, window_updates)
    if len(updates) != expected_retained:
        raise ValueError("Acquisition gate window retains the wrong number of updates")
    first_update = completed - expected_retained + 1
    expected_indices = list(range(first_update, completed + 1)) if expected_retained else []
    if [entry.get("completed_update") for entry in updates if isinstance(entry, Mapping)] != expected_indices:
        raise ValueError("Acquisition gate window update indices are not contiguous and trailing")
    if any(not isinstance(entry, Mapping) for entry in updates):
        raise ValueError("Acquisition gate window contains a malformed update")

    cumulative_template = initialize_acquisition_diagnostics(arm, cue_regimes)
    for entry in updates:
        if set(entry) != {"completed_update", "cells"}:
            raise ValueError("Acquisition gate window update has an invalid schema")
        per_update = dict(cumulative_template)
        per_update["cells"] = entry["cells"]
        validated_update = validate_acquisition_diagnostics_state(
            per_update,
            arm=arm,
            cue_regimes=cue_regimes,
        )
        retained_count = sum(
            int(validated_update["cells"][cue][condition]["count"])
            for cue in cue_regimes
            for condition in ACQUISITION_CONDITIONS
        )
        if retained_count != samples_per_update:
            raise ValueError(
                "Acquisition gate window update does not contain one complete rollout batch"
            )
        entry["cells"] = validated_update["cells"]
    return copied


def update_acquisition_gate_window_diagnostics(
    state: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    completed_update: int,
) -> dict[str, Any]:
    """Append exactly one optimizer update, then evict observations outside the window."""
    arm = str(state.get("optimized_arm", ""))
    cues = list(state.get("cue_regimes") or [])
    window_updates = state.get("window_updates")
    samples_per_update = state.get("samples_per_update")
    if type(window_updates) is not int or type(samples_per_update) is not int:
        raise ValueError("Acquisition gate window lacks integer sizing metadata")
    updated = validate_acquisition_gate_window_state(
        state,
        arm=arm,
        cue_regimes=cues,
        window_updates=window_updates,
        samples_per_update=samples_per_update,
    )
    if type(completed_update) is not int or completed_update != updated["completed_updates"] + 1:
        raise ValueError("Acquisition gate window updates must be appended sequentially")
    if len(cases) != samples_per_update:
        raise ValueError("Acquisition gate window did not receive one complete rollout batch")
    per_update = initialize_acquisition_diagnostics(arm, cues)
    per_update = update_acquisition_diagnostics(per_update, cases, outcomes)
    updated["updates"].append({
        "completed_update": completed_update,
        "cells": per_update["cells"],
    })
    updated["updates"] = updated["updates"][-window_updates:]
    updated["completed_updates"] = completed_update
    return validate_acquisition_gate_window_state(
        updated,
        arm=arm,
        cue_regimes=cues,
        window_updates=window_updates,
        samples_per_update=samples_per_update,
    )


def acquisition_gate_window_diagnostics_summary(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the exact trailing-window rates used by continuation gates."""
    arm = str(state.get("optimized_arm", ""))
    cues = list(state.get("cue_regimes") or [])
    window_updates = state.get("window_updates")
    samples_per_update = state.get("samples_per_update")
    if type(window_updates) is not int or type(samples_per_update) is not int:
        raise ValueError("Acquisition gate window lacks integer sizing metadata")
    validated = validate_acquisition_gate_window_state(
        state,
        arm=arm,
        cue_regimes=cues,
        window_updates=window_updates,
        samples_per_update=samples_per_update,
    )
    aggregate = initialize_acquisition_diagnostics(arm, cues)
    for entry in validated["updates"]:
        for cue in cues:
            for condition in ACQUISITION_CONDITIONS:
                source = entry["cells"][cue][condition]
                target = aggregate["cells"][cue][condition]
                target["count"] += int(source["count"])
                target["reward_sum"] += float(source["reward_sum"])
                target["success_count"] += int(source["success_count"])
    rate_summary = acquisition_diagnostics_summary(aggregate)
    rate_summary["diagnostic_scope"] = validated["diagnostic_scope"]
    retained = len(validated["updates"])
    first = validated["updates"][0]["completed_update"] if retained else None
    last = validated["updates"][-1]["completed_update"] if retained else None
    return {
        "schema_version": validated["schema_version"],
        "diagnostic_scope": validated["diagnostic_scope"],
        "optimized_arm": arm,
        "cue_regimes": cues,
        "success_definition": validated["success_definition"],
        "window_updates": window_updates,
        "samples_per_update": samples_per_update,
        "completed_updates": validated["completed_updates"],
        "covered_updates": {
            "first_completed_update": first,
            "last_completed_update": last,
            "update_count": retained,
        },
        "cells": rate_summary["cells"],
        "by_cue_regime": rate_summary["by_cue_regime"],
        "by_condition": rate_summary["by_condition"],
        "overall": rate_summary["overall"],
    }


def differentiable_choice_log_probs(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    labels: Sequence[str] = CHOICES,
) -> tuple[Any, Any]:
    """Return normalized legal-choice log-probabilities and raw token log-probs.

    The frozen bridge uses single-token ``A``/``B`` choices. Both scores therefore
    come from one next-token forward pass per prompt. ``logits_to_keep=1`` prevents
    the otherwise enormous ``batch x sequence x vocabulary`` logits allocation.
    """
    import torch
    import torch.nn.functional as functional

    if tuple(labels) != CHOICES:
        raise ValueError(f"The bridge currently requires the frozen labels {CHOICES}")
    if max_length <= 0:
        raise ValueError("Bridge max_length must be positive")
    prompt_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    for case in cases:
        prompt_text = chat_prompt_text(tokenizer, list(case["messages"]))
        untruncated_prompt = list(
            tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
        if not untruncated_prompt:
            raise ValueError("Bridge prompt produced no tokens")
        # The scorer and the unconstrained one-token generation diagnostic must
        # receive the same left-truncated prompt. The candidate is not part of the
        # model input for next-token scoring, so it must not consume this budget.
        case_prompt = untruncated_prompt[-max_length:]
        case_targets: list[int] = []
        for choice in labels:
            full_ids = list(
                tokenizer(prompt_text + choice, add_special_tokens=False)["input_ids"]
            )
            if full_ids[: len(untruncated_prompt)] != untruncated_prompt:
                raise ValueError("Tokenizer changed the prompt boundary when a choice was appended")
            completion = full_ids[len(untruncated_prompt) :]
            if len(completion) != 1:
                raise ValueError(
                    f"Bridge choice {choice!r} must be exactly one token; got {len(completion)}"
                )
            case_targets.append(int(completion[0]))
        if len(set(case_targets)) != len(case_targets):
            raise ValueError("A/B choices map to the same completion token")
        prompt_rows.append(case_prompt)
        target_rows.append(case_targets)
    if not prompt_rows:
        raise ValueError("Cannot score an empty acquisition batch")
    device = next(model.parameters()).device
    longest = max(map(len, prompt_rows))
    input_ids = torch.full(
        (len(prompt_rows), longest), int(tokenizer.pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(prompt_rows):
        offset = longest - len(ids)
        input_ids[row, offset:] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, offset:] = 1
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=1,
        ).logits
    except TypeError as exc:
        raise RuntimeError(
            "The pinned causal-LM implementation must support logits_to_keep; "
            "refusing a full-sequence logits allocation"
        ) from exc
    if logits.ndim != 3 or logits.shape[0] != len(prompt_rows) or logits.shape[1] != 1:
        raise RuntimeError(f"Unexpected bridge logits shape: {tuple(logits.shape)}")
    next_token_log_probs = functional.log_softmax(logits[:, 0, :].float(), dim=-1)
    targets = torch.tensor(target_rows, dtype=torch.long, device=device)
    raw = next_token_log_probs.gather(1, targets)
    legal = functional.log_softmax(raw, dim=-1)
    if not bool(torch.isfinite(legal).all() and torch.isfinite(raw).all()):
        raise FloatingPointError("Non-finite bridge choice likelihood")
    return legal, raw


def sample_legal_actions(log_choice_probs: Any, *, seed: int) -> Any:
    """Sample with an isolated RNG so G/P arms can use paired random numbers."""
    import torch

    if log_choice_probs.ndim != 2 or log_choice_probs.shape[1] != len(CHOICES):
        raise ValueError("Expected a [batch, 2] legal-choice log-probability tensor")
    # Generate common random numbers on CPU. This makes action sampling identical
    # across device types and pairs the G/P arms without perturbing global RNG state.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    uniforms = torch.rand(log_choice_probs.shape[0], generator=generator)
    probability_a = log_choice_probs.detach().exp()[:, 0].float().cpu()
    return (uniforms >= probability_a).long().to(log_choice_probs.device)


def categorical_policy_kl(log_choice_probs: Any, reference_log_choice_probs: Any) -> Any:
    """Forward KL on the frozen legal A/B policy, averaged over prompts."""
    import torch

    if log_choice_probs.shape != reference_log_choice_probs.shape:
        raise ValueError("Policy and reference choice distributions have different shapes")
    if log_choice_probs.ndim != 2 or log_choice_probs.shape[1] != len(CHOICES):
        raise ValueError("Expected [batch, 2] legal-choice log probabilities")
    reference = reference_log_choice_probs.detach()
    kl = (log_choice_probs.exp() * (log_choice_probs - reference)).sum(dim=1).mean()
    if not bool(torch.isfinite(kl)):
        raise FloatingPointError("Non-finite bridge policy KL")
    # Numerical roundoff can make an exact-zero KL microscopically negative.
    return kl.clamp_min(0.0)


def legal_plus_other_policy_kl(raw_choice_log_probs: Any, reference_raw_choice_log_probs: Any) -> Any:
    """KL over the exhaustive next-token partition ``A``, ``B``, and ``other``.

    A conditional A/B KL alone has a loophole: it is unchanged if the model moves
    nearly all probability to non-protocol tokens. This partitioned KL preserves
    the base model's legal-answer mass while remaining cheap to compute.
    """
    import torch

    if raw_choice_log_probs.shape != reference_raw_choice_log_probs.shape:
        raise ValueError("Policy and reference raw choice scores have different shapes")
    if raw_choice_log_probs.ndim != 2 or raw_choice_log_probs.shape[1] != len(CHOICES):
        raise ValueError("Expected [batch, 2] raw A/B token log probabilities")

    def partition(raw: Any) -> Any:
        legal = raw.exp().sum(dim=1)
        if bool((legal > 1.0 + 1e-5).any()):
            raise ValueError("A/B token probability mass exceeds one")
        epsilon = torch.finfo(raw.dtype).eps
        other = (1.0 - legal).clamp_min(epsilon)
        return torch.cat([raw, other.log()[:, None]], dim=1)

    current = partition(raw_choice_log_probs)
    reference = partition(reference_raw_choice_log_probs.detach())
    kl = (current.exp() * (current - reference)).sum(dim=1).mean()
    if not bool(torch.isfinite(kl)):
        raise FloatingPointError("Non-finite A/B/other policy KL")
    return kl.clamp_min(0.0)


def policy_gradient_objective(
    log_choice_probs: Any,
    chosen_indices: Any,
    rewards: Any,
    *,
    baseline: float,
    entropy_coefficient: float,
    normalize_advantages: bool = False,
    reference_log_choice_probs: Any | None = None,
    raw_choice_log_probs: Any | None = None,
    reference_raw_choice_log_probs: Any | None = None,
    kl_coefficient: float = 0.0,
    advantages_override: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    """REINFORCE objective using only realized selected-action reward."""
    import torch

    if log_choice_probs.ndim != 2 or log_choice_probs.shape[1] != len(CHOICES):
        raise ValueError("Expected [batch, 2] log probabilities")
    if chosen_indices.shape != rewards.shape or chosen_indices.shape[0] != log_choice_probs.shape[0]:
        raise ValueError("Action/reward batch shapes do not match choice probabilities")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("Rewards must be finite")
    selected = log_choice_probs.gather(1, chosen_indices.long()[:, None]).squeeze(1)
    raw_advantages = rewards.float() - float(baseline)
    advantages = raw_advantages if advantages_override is None else advantages_override.float()
    if advantages.shape != rewards.shape:
        raise ValueError("Provided advantages do not match the reward batch")
    if normalize_advantages and advantages_override is None:
        advantages = advantages - advantages.mean()
        scale = advantages.std(unbiased=False)
        if float(scale.detach().cpu()) > 1e-8:
            advantages = advantages / scale
    entropy = -(log_choice_probs.exp() * log_choice_probs).sum(dim=1)
    policy_loss = -(advantages.detach() * selected).mean()
    if reference_log_choice_probs is not None:
        if raw_choice_log_probs is None or reference_raw_choice_log_probs is None:
            raise ValueError("Bridge KL requires raw A/B scores to constrain non-choice mass")
        kl = legal_plus_other_policy_kl(raw_choice_log_probs, reference_raw_choice_log_probs)
    else:
        kl = log_choice_probs.new_zeros(())
    if reference_log_choice_probs is None and kl_coefficient:
        raise ValueError("A positive KL coefficient requires a reference policy")
    loss = (
        policy_loss
        - float(entropy_coefficient) * entropy.mean()
        + float(kl_coefficient) * kl
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite policy-gradient loss")
    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "policy_kl_nats": float(kl.detach().cpu()),
        "reward_mean": float(rewards.detach().float().mean().cpu()),
        "reward_std": float(rewards.detach().float().std(unbiased=False).cpu()),
        "raw_advantage_mean": float(raw_advantages.detach().mean().cpu()),
        "advantage_mean": float(advantages.detach().mean().cpu()),
        "advantage_std": float(advantages.detach().std(unbiased=False).cpu()),
        "entropy_nats": float(entropy.detach().mean().cpu()),
    }
    return loss, metrics


def update_reward_baseline(
    current: float, reward_mean: float, *, momentum: float, initialized: bool
) -> tuple[float, bool]:
    value = float(reward_mean) if not initialized else momentum * current + (1.0 - momentum) * reward_mean
    if not math.isfinite(value):
        raise ValueError("Reward baseline became non-finite")
    return value, True


def rollout_advantages(rewards: Any, *, baseline: float, normalize: bool) -> Any:
    """Compute advantages once over the full rollout, independent of microbatching."""
    import torch

    if rewards.ndim != 1 or not bool(torch.isfinite(rewards).all()):
        raise ValueError("Rollout rewards must be one finite vector")
    advantages = rewards.float() - float(baseline)
    if normalize:
        advantages = advantages - advantages.mean()
        scale = advantages.std(unbiased=False)
        if float(scale.detach().cpu()) > 1e-8:
            advantages = advantages / scale
    return advantages


def _constant_with_warmup_scheduler(optimizer: Any, spec: BridgeTrainingSpec) -> Any:
    """Create a fully checkpointed constant schedule with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    warmup_updates = math.ceil(spec.warmup_ratio * spec.updates)

    def multiplier(step_index: int) -> float:
        if warmup_updates == 0:
            return 1.0
        return min(1.0, float(step_index + 1) / float(warmup_updates))

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _frozen_reference_choice_log_probs(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
) -> tuple[Any, Any]:
    """Score the unchanged base policy while the trainable LoRA adapter is disabled."""
    import torch

    disable_adapter = getattr(model, "disable_adapter", None)
    if not callable(disable_adapter):
        raise TypeError("Bridge KL requires PEFT's disable_adapter context manager")
    with torch.no_grad(), disable_adapter():
        reference, raw_reference = differentiable_choice_log_probs(
            model,
            tokenizer,
            cases,
            max_length=max_length,
        )
    return reference.detach(), raw_reference.detach()


def _make_reload_probe(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Capture same-prompt probabilities before serialization for reload QA."""
    import torch

    selected = list(cases[:limit])
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            legal, raw = differentiable_choice_log_probs(
                model,
                tokenizer,
                selected,
                max_length=max_length,
            )
    finally:
        model.train(was_training)
    rows: list[dict[str, Any]] = []
    for case, probabilities, scores in zip(
        selected, legal.exp().float().cpu(), raw.float().cpu(), strict=True
    ):
        rows.append({
            "case_id": str(case["case_id"]),
            "messages": [dict(message) for message in case["messages"]],
            "probability_A_before_save": float(probabilities[0]),
            "logp_A_before_save": float(scores[0]),
            "logp_B_before_save": float(scores[1]),
        })
    return rows


def _environment_provenance(environment: BridgeEnvironment) -> dict[str, Any]:
    provenance = dict(environment.provenance())
    if not provenance:
        raise ValueError("Bridge environment provenance cannot be empty")
    # Fail now if the environment emitted a non-serializable or unstable structure.
    json.loads(canonical_json(provenance))
    return provenance


def _environment_state_hash(environment: BridgeEnvironment) -> str:
    """Hash the fresh task state so paired arms cannot silently start differently."""
    state = dict(environment.state_dict())
    try:
        payload = canonical_json(state)
    except (TypeError, ValueError) as exc:
        raise TypeError("Bridge environment state_dict must be stable JSON data") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _spec_hash(spec: BridgeTrainingSpec) -> str:
    return hashlib.sha256(canonical_json(asdict(spec)).encode("utf-8")).hexdigest()


def config_bound_bridge_training_spec(
    config: Mapping[str, Any], requested_settings: Mapping[str, Any] | None = None
) -> BridgeTrainingSpec:
    """Return the config spec, rejecting any evidence-changing runtime override."""
    configured = BridgeTrainingSpec.from_config(config)
    if requested_settings is not None:
        requested = BridgeTrainingSpec.from_config(config, requested_settings)
        if requested != configured:
            raise ValueError(
                "Evidence-producing bridge training forbids settings that differ "
                "from the loaded config"
            )
    return configured


def configured_bridge_spec_sha256(config: Mapping[str, Any]) -> str:
    """Hash the complete training spec derived exclusively from the run config."""
    return _spec_hash(config_bound_bridge_training_spec(config))


def _paired_initialization_id(config: Mapping[str, Any], spec: BridgeTrainingSpec, seed: int) -> str:
    payload = {
        "model": config["model"],
        "lora": {
            key: config["training"][key]
            for key in ("lora_rank", "lora_alpha", "lora_dropout", "lora_targets")
        },
        "pair_seed": int(seed),
        "bridge_spec": asdict(spec),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _checkpoint_file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "checkpoint_manifest.json"
    }


def _canonicalize_adapter_config(directory: Path) -> None:
    """Make PEFT's set-derived adapter metadata byte-stable across paired runs."""
    path = directory / "adapter_config.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("target_modules")
    if targets is not None:
        if (
            not isinstance(targets, list)
            or not all(isinstance(target, str) and target for target in targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError("Saved PEFT adapter_config has invalid target_modules")
        payload["target_modules"] = sorted(targets)
    write_json(path, payload)


def _checkpoint_record(directory: Path) -> dict[str, Any]:
    checkpoint_manifest = directory / "checkpoint_manifest.json"
    data = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    hashes = data.get("file_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError(f"Checkpoint has no file-integrity inventory: {directory}")
    for name, expected in hashes.items():
        path = directory / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Checkpoint file-integrity failure: {path}")
    runtime = data.get("model_runtime_attestation")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"Checkpoint has no model runtime attestation: {directory}")
    unsigned_runtime = dict(runtime)
    runtime_hash = unsigned_runtime.pop("attestation_sha256", None)
    if (
        runtime_hash
        != hashlib.sha256(canonical_json(unsigned_runtime).encode("utf-8")).hexdigest()
        or data.get("model_runtime_attestation_sha256") != runtime_hash
    ):
        raise ValueError(f"Checkpoint model runtime attestation failed: {directory}")
    return {
        "completed_updates": int(data["completed_updates"]),
        "path": str(directory),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "adapter_file_sha256": {
            name: digest
            for name, digest in data["file_sha256"].items()
            if name.startswith("adapter_") or name.endswith(".safetensors")
        },
        "model_runtime_attestation_sha256": runtime_hash,
        "acquisition_diagnostics": data.get("acquisition_diagnostics"),
        "acquisition_gate_window_diagnostics": data.get(
            "acquisition_gate_window_diagnostics"
        ),
    }


def _save_checkpoint(
    *,
    run_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    environment: BridgeEnvironment,
    completed_updates: int,
    baseline: float,
    baseline_initialized: bool,
    arm: str,
    seed: int,
    config_sha256: str,
    spec_sha256: str,
    environment_provenance: Mapping[str, Any],
    initial_environment_state_sha256: str,
    acquisition_diagnostics_state: Mapping[str, Any],
    acquisition_gate_window_diagnostics_state: Mapping[str, Any],
    model_runtime_attestation: Mapping[str, Any],
    reload_probe: Sequence[Mapping[str, Any]] | None = None,
    optimizer_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    import torch
    import numpy as np

    checkpoints_root = run_dir / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    target = checkpoints_root / f"checkpoint-{completed_updates:06d}"
    validated_diagnostics = validate_acquisition_diagnostics_state(
        acquisition_diagnostics_state,
        arm=arm,
        cue_regimes=list(acquisition_diagnostics_state.get("cue_regimes") or []),
    )
    diagnostic_summary = acquisition_diagnostics_summary(validated_diagnostics)
    window_updates = acquisition_gate_window_diagnostics_state.get("window_updates")
    samples_per_update = acquisition_gate_window_diagnostics_state.get("samples_per_update")
    if type(window_updates) is not int or type(samples_per_update) is not int:
        raise ValueError("Acquisition gate window checkpoint state lacks sizing metadata")
    validated_gate_window = validate_acquisition_gate_window_state(
        acquisition_gate_window_diagnostics_state,
        arm=arm,
        cue_regimes=list(acquisition_gate_window_diagnostics_state.get("cue_regimes") or []),
        window_updates=window_updates,
        samples_per_update=samples_per_update,
    )
    gate_window_summary = acquisition_gate_window_diagnostics_summary(
        validated_gate_window
    )
    verified_model_runtime = json.loads(canonical_json(dict(model_runtime_attestation)))
    # The full config is validated when the attestation is created and when a
    # checkpoint is consumed.  Here, preserve its exact self-hashed JSON payload.
    claimed_runtime_hash = verified_model_runtime.get("attestation_sha256")
    unsigned_runtime = dict(verified_model_runtime)
    unsigned_runtime.pop("attestation_sha256", None)
    if claimed_runtime_hash != hashlib.sha256(
        canonical_json(unsigned_runtime).encode("utf-8")
    ).hexdigest():
        raise ValueError("Model runtime attestation self-hash mismatch at checkpoint save")
    if validated_gate_window["cue_regimes"] != validated_diagnostics["cue_regimes"]:
        raise ValueError("Cumulative and trailing acquisition diagnostics use different cues")
    if int(validated_gate_window["completed_updates"]) != int(completed_updates):
        raise ValueError("Acquisition gate window is not synchronized with the checkpoint")
    expected_cumulative_samples = int(completed_updates) * samples_per_update
    if int(diagnostic_summary["overall"]["sample_count"]) != expected_cumulative_samples:
        raise ValueError("Cumulative acquisition diagnostics have incomplete rollout coverage")
    expected_window_samples = min(int(completed_updates), window_updates) * samples_per_update
    if int(gate_window_summary["overall"]["sample_count"]) != expected_window_samples:
        raise ValueError("Acquisition gate window has incomplete trailing rollout coverage")
    if target.exists():
        existing_manifest = json.loads(
            (target / "checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        if existing_manifest.get("acquisition_diagnostics") != diagnostic_summary:
            raise ValueError("Existing checkpoint has different acquisition diagnostics")
        if (
            existing_manifest.get("acquisition_gate_window_diagnostics")
            != gate_window_summary
        ):
            raise ValueError("Existing checkpoint has a different acquisition gate window")
        if existing_manifest.get("model_runtime_attestation") != verified_model_runtime:
            raise ValueError("Existing checkpoint has a different model runtime attestation")
        return _checkpoint_record(target)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=checkpoints_root))
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        _canonicalize_adapter_config(temporary)
        tokenizer.save_pretrained(temporary)
        environment_state = dict(environment.state_dict())
        experienced_cases = environment_state.get("experienced_cases")
        if experienced_cases is not None and int(experienced_cases) != int(
            diagnostic_summary["overall"]["sample_count"]
        ):
            raise ValueError(
                "Environment experience count disagrees with cumulative acquisition diagnostics"
            )
        state = {
            "schema_version": "1.0",
            "completed_updates": int(completed_updates),
            "arm": arm,
            "pair_seed": int(seed),
            "config_sha256": config_sha256,
            "bridge_spec_sha256": spec_sha256,
            "reward_baseline": float(baseline),
            "baseline_initialized": bool(baseline_initialized),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "environment_state": environment_state,
            "acquisition_diagnostics_state": validated_diagnostics,
            "acquisition_gate_window_diagnostics_state": validated_gate_window,
        }
        torch.save(state, temporary / "bridge_state.pt")
        if reload_probe:
            write_json(temporary / "reload_probe.json", list(reload_probe))
        checkpoint_manifest = {
            "schema_version": "1.0",
            "kind": "bridge_policy_checkpoint",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_updates": int(completed_updates),
            "arm": arm,
            "pair_seed": int(seed),
            "config_sha256": config_sha256,
            "bridge_spec_sha256": spec_sha256,
            "environment_provenance": dict(environment_provenance),
            "model_runtime_attestation": verified_model_runtime,
            "model_runtime_attestation_sha256": claimed_runtime_hash,
            "initial_environment_state_sha256": initial_environment_state_sha256,
            "optimizer_metrics": dict(optimizer_metrics) if optimizer_metrics is not None else None,
            "acquisition_diagnostics": diagnostic_summary,
            "acquisition_gate_window_diagnostics": gate_window_summary,
            "file_sha256": _checkpoint_file_hashes(temporary),
        }
        write_json(temporary / "checkpoint_manifest.json", checkpoint_manifest)
        os.replace(temporary, target)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _checkpoint_record(target)


def _verify_checkpoint(
    checkpoint: Path,
    *,
    arm: str,
    seed: int,
    config_sha256: str,
    spec_sha256: str,
    environment_provenance: Mapping[str, Any],
    initial_environment_state_sha256: str | None = None,
    model_runtime_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing bridge checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    expected = {
        "arm": arm,
        "pair_seed": int(seed),
        "config_sha256": config_sha256,
        "bridge_spec_sha256": spec_sha256,
        "environment_provenance": dict(environment_provenance),
    }
    if initial_environment_state_sha256 is not None:
        expected["initial_environment_state_sha256"] = initial_environment_state_sha256
    runtime = manifest.get("model_runtime_attestation")
    if not isinstance(runtime, Mapping):
        failures.append("model runtime attestation is missing")
    else:
        runtime_copy = dict(runtime)
        runtime_hash = runtime_copy.pop("attestation_sha256", None)
        if (
            runtime_hash
            != hashlib.sha256(canonical_json(runtime_copy).encode("utf-8")).hexdigest()
            or manifest.get("model_runtime_attestation_sha256") != runtime_hash
        ):
            failures.append("model runtime attestation hash mismatch")
        if model_runtime_attestation is not None and dict(runtime) != dict(
            model_runtime_attestation
        ):
            failures.append("model runtime attestation differs")
    for key, value in expected.items():
        if manifest.get(key) != value:
            failures.append(f"{key} differs")
    for name, digest in (manifest.get("file_sha256") or {}).items():
        path = checkpoint / name
        if not path.exists() or sha256_file(path) != digest:
            failures.append(f"file hash mismatch: {name}")
    if not manifest.get("file_sha256"):
        failures.append("file hash list is empty")
    if failures:
        raise ValueError("Bridge checkpoint provenance failed: " + "; ".join(failures))
    return manifest


def _latest_checkpoint(run_dir: Path) -> Path:
    candidates = [
        path
        for path in (run_dir / "checkpoints").glob("checkpoint-*")
        if path.is_dir() and path.name.split("-")[-1].isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"No resumable bridge checkpoint under {run_dir}")
    return max(candidates, key=lambda path: int(path.name.split("-")[-1]))


def load_bridge_state(checkpoint: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load state created by this package; never use it on an untrusted archive."""
    import torch

    return dict(torch.load(Path(checkpoint) / "bridge_state.pt", map_location=map_location, weights_only=False))


def _materialize_final_adapter(run_dir: Path, source_checkpoint: Path) -> Path:
    """Atomically expose the completed, verified checkpoint at a stable path."""
    final_adapter = run_dir / "final_adapter"
    temporary = run_dir / ".final_adapter.tmp"
    _checkpoint_record(source_checkpoint)
    source_digest = sha256_file(source_checkpoint / "checkpoint_manifest.json")

    def finalize_manifest(directory: Path) -> None:
        _checkpoint_record(directory)
        manifest_path = directory / "final_adapter_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_checkpoint_manifest_sha256") != source_digest:
                raise ValueError("Existing final adapter points to a different checkpoint")
            return
        write_json(
            manifest_path,
            {
                "schema_version": "1.0",
                "kind": "bridge_final_adapter_pointer",
                "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_manifest_sha256": source_digest,
            },
        )

    if final_adapter.exists():
        finalize_manifest(final_adapter)
        return final_adapter
    if temporary.exists():
        finalize_manifest(temporary)
        os.replace(temporary, final_adapter)
        return final_adapter
    try:
        shutil.copytree(source_checkpoint, temporary)
        finalize_manifest(temporary)
        os.replace(temporary, final_adapter)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_adapter


def _append_events(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _truncate_events_for_resume(path: Path, completed_updates: int) -> None:
    if not path.exists():
        return
    retained = [
        event for event in read_jsonl(path) if int(event.get("update_index", -1)) < completed_updates
    ]
    write_jsonl(path, retained)


def _deadline_near(config: Mapping[str, Any]) -> bool:
    raw = os.environ.get("UE_HARD_DEADLINE_EPOCH")
    if not raw:
        return False
    soft_seconds = int(float(config.get("budget", {}).get("soft_stop_minutes", 30)) * 60)
    return time.time() >= float(raw) - soft_seconds


def _update_manifest(run_dir: Path, manifest: dict[str, Any], state: str) -> None:
    manifest["state"] = state
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "bridge_manifest.json", manifest)
    for marker in ("RUNNING", "COMPLETE", "FAILED", "STOPPED_BUDGET", "STOPPED_EARLY"):
        marker_path = run_dir / marker
        if marker_path.exists():
            marker_path.unlink()
    (run_dir / state).touch()


def train_bridge_arm(
    config: dict[str, Any],
    environment: BridgeEnvironment,
    *,
    arm: str,
    pair_seed: int,
    bridge_settings: Mapping[str, Any] | None = None,
    run_dir: str | Path | None = None,
    resume: bool = False,
) -> Path:
    """Train one member of a paired G/P bridge and return its final checkpoint.

    Call this function twice with fresh, identically configured environment objects,
    the same ``pair_seed``, and arms ``genuine`` and ``proxy``.  Initial adapter weights,
    acquisition batches, dropout RNG, and action-sampling random numbers are then paired;
    only the reward field entering REINFORCE differs.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed

    if torch.cuda.is_available():
        # Bound both allocator views to this complete training process.  Reserved
        # memory, rather than allocated tensors alone, is the conservative OOM
        # quantity used by the paid preflight.
        torch.cuda.reset_peak_memory_stats()

    selected_arm = canonical_arm(arm)
    spec = config_bound_bridge_training_spec(config, bridge_settings)
    if float(config["training"]["lora_dropout"]) != 0.0:
        raise ValueError("Paired bridge acquisition requires lora_dropout=0")
    provenance = _environment_provenance(environment)
    initial_environment_state_sha256 = _environment_state_hash(environment)
    spec_sha256 = _spec_hash(spec)
    cfg_sha256 = config.get("_config_sha256") or config_hash(config)
    project_root = Path(config["_config_path"]).parent.parent
    target = (
        Path(run_dir).resolve()
        if run_dir
        else output_root(config) / "bridge" / "runs" / f"{selected_arm}_seed{pair_seed}"
    )
    manifest_path = target / "bridge_manifest.json"
    prior: dict[str, Any] | None = None
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--resume requires {manifest_path}")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        failures: list[str] = []
        if prior.get("state") not in {"STOPPED_BUDGET", "STOPPED_EARLY", "FAILED"}:
            failures.append(f"state={prior.get('state')!r}")
        expected_prior = {
            "arm": selected_arm,
            "pair_seed": int(pair_seed),
            "config_sha256": cfg_sha256,
            "bridge_spec_sha256": spec_sha256,
            "environment_provenance": provenance,
            "initial_environment_state_sha256": initial_environment_state_sha256,
            "project_tree_sha256": project_hash(project_root),
        }
        for key, value in expected_prior.items():
            if prior.get(key) != value:
                failures.append(f"{key} differs")
        if not isinstance(prior.get("initial_adapter_file_sha256"), Mapping) or not prior[
            "initial_adapter_file_sha256"
        ]:
            failures.append("initial adapter hashes are missing")
        if failures:
            raise ValueError("Bridge resume provenance failed: " + "; ".join(failures))
    elif target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty bridge run directory {target}")

    target.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "paired_environment_grounded_policy_gradient",
        "state": "RUNNING",
        "created_at_utc": (
            prior.get("created_at_utc") if prior is not None else datetime.now(timezone.utc).isoformat()
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": selected_arm,
        "pair_seed": int(pair_seed),
        "paired_initialization_id": _paired_initialization_id(config, spec, pair_seed),
        "config_sha256": cfg_sha256,
        "bridge_spec": asdict(spec),
        "bridge_spec_sha256": spec_sha256,
        "bridge_spec_source": "loaded_config_exact",
        "optimizer_update_semantics": {
            "optimizer_updates": spec.updates,
            "rollout_batch_size": spec.batch_size,
            "gradient_accumulation_steps": spec.gradient_accumulation_steps,
            "microbatch_size": spec.microbatch_size,
            "checkpoint_updates": list(spec.checkpoint_updates),
            "kl_reference": "initial_base_legal_choice_policy",
            "advantage_normalization_scope": "full_rollout_batch",
        },
        "model": config["model"],
        "lora": {
            key: config["training"][key]
            for key in ("lora_rank", "lora_alpha", "lora_dropout", "lora_targets")
        },
        "model_runtime_attestation": (
            prior.get("model_runtime_attestation") if prior is not None else None
        ),
        "environment_provenance": provenance,
        "initial_environment_state_sha256": initial_environment_state_sha256,
        "initial_adapter_file_sha256": (
            prior.get("initial_adapter_file_sha256") if prior is not None else None
        ),
        "project_tree_sha256": project_hash(project_root),
        "command_line": list(sys.argv),
        "environment": environment_snapshot(),
        "resume_lineage": None if prior is None else {
            "prior_updated_at_utc": prior.get("updated_at_utc"),
            "prior_state": prior.get("state"),
            "prior_completed_updates": prior.get("completed_updates"),
        },
        "completed_updates": 0,
        "checkpoints": list(prior.get("checkpoints", [])) if prior is not None else [],
        "result": None,
    }
    write_json(manifest_path, manifest)
    _update_manifest(target, manifest, "RUNNING")

    model: Any | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None
    completed_updates = 0
    baseline = 0.0
    baseline_initialized = False
    acquisition_diagnostics_state = initialize_acquisition_diagnostics(
        selected_arm, list(config["bridge"]["cue_regimes"])
    )
    acquisition_gate_window_diagnostics_state = (
        initialize_acquisition_gate_window_diagnostics(
            selected_arm,
            list(config["bridge"]["cue_regimes"]),
            window_updates=int(config["training"]["acquisition_gate_window_updates"]),
            samples_per_update=spec.batch_size,
        )
    )
    last_optimizer_metrics: dict[str, float] | None = None
    events_path = target / "acquisition_events.jsonl"
    started = time.monotonic()
    timing: dict[str, Any] = {
        "schema_version": "1.0",
        "model_and_tokenizer_load_wall_seconds": 0.0,
        "adapter_setup_and_attestation_wall_seconds": 0.0,
        "update_compute_wall_seconds_total": 0.0,
        "rollout_wall_seconds_total": 0.0,
        "gradient_forward_backward_wall_seconds_total": 0.0,
        "optimizer_step_wall_seconds_total": 0.0,
        "diagnostics_wall_seconds_total": 0.0,
        "checkpoint_write_wall_seconds_total": 0.0,
        "maximum_checkpoint_write_wall_seconds": 0.0,
        "checkpoint_write_count": 0,
        "reload_probe_wall_seconds_total": 0.0,
        "maximum_reload_probe_wall_seconds": 0.0,
        "reload_probe_count": 0,
        "finalize_wall_seconds": 0.0,
        "completed_update_count": 0,
        "resume_count": 0,
    }
    if prior is not None and isinstance(prior.get("timing"), Mapping):
        previous_timing = dict(prior["timing"])
        if previous_timing.get("schema_version") == timing["schema_version"]:
            timing.update(previous_timing)
        timing["resume_count"] = int(timing.get("resume_count", 0)) + 1
    model_runtime_attestation: dict[str, Any] | None = None
    try:
        set_seed(int(pair_seed))
        load_started = time.monotonic()
        tokenizer = load_tokenizer(config)
        if resume:
            checkpoint = _latest_checkpoint(target)
            verified_resume = _verify_checkpoint(
                checkpoint,
                arm=selected_arm,
                seed=pair_seed,
                config_sha256=cfg_sha256,
                spec_sha256=spec_sha256,
                environment_provenance=provenance,
                initial_environment_state_sha256=initial_environment_state_sha256,
            )
            raw_metrics = verified_resume.get("optimizer_metrics")
            if raw_metrics is not None:
                last_optimizer_metrics = {
                    str(key): float(value) for key, value in dict(raw_metrics).items()
                }
            base = load_base_model(config, training=True)
            target_inventory = inspect_lora_target_inventory(config, base)
            timing["model_and_tokenizer_load_wall_seconds"] += (
                time.monotonic() - load_started
            )
            adapter_started = time.monotonic()
            model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
            model_runtime_attestation = build_model_runtime_attestation(
                config, model, tokenizer, target_inventory
            )
            verify_model_runtime_attestation(config, model_runtime_attestation)
            checkpoint_runtime = verified_resume.get("model_runtime_attestation")
            if checkpoint_runtime != model_runtime_attestation:
                raise ValueError(
                    "Reloaded model runtime/LoRA attestation differs from the checkpoint"
                )
            if prior is not None and prior.get("model_runtime_attestation") != (
                model_runtime_attestation
            ):
                raise ValueError(
                    "Reloaded model runtime/LoRA attestation differs from the run manifest"
                )
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=spec.learning_rate,
                weight_decay=spec.weight_decay,
            )
            scheduler = _constant_with_warmup_scheduler(optimizer, spec)
            timing["adapter_setup_and_attestation_wall_seconds"] += (
                time.monotonic() - adapter_started
            )
            state = load_bridge_state(checkpoint)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            completed_updates = int(state["completed_updates"])
            baseline = float(state["reward_baseline"])
            baseline_initialized = bool(state["baseline_initialized"])
            acquisition_diagnostics_state = validate_acquisition_diagnostics_state(
                state.get("acquisition_diagnostics_state") or {},
                arm=selected_arm,
                cue_regimes=list(config["bridge"]["cue_regimes"]),
            )
            if verified_resume.get("acquisition_diagnostics") != acquisition_diagnostics_summary(
                acquisition_diagnostics_state
            ):
                raise ValueError("Checkpoint state/manifest acquisition diagnostics mismatch")
            acquisition_gate_window_diagnostics_state = (
                validate_acquisition_gate_window_state(
                    state.get("acquisition_gate_window_diagnostics_state") or {},
                    arm=selected_arm,
                    cue_regimes=list(config["bridge"]["cue_regimes"]),
                    window_updates=int(
                        config["training"]["acquisition_gate_window_updates"]
                    ),
                    samples_per_update=spec.batch_size,
                )
            )
            if verified_resume.get(
                "acquisition_gate_window_diagnostics"
            ) != acquisition_gate_window_diagnostics_summary(
                acquisition_gate_window_diagnostics_state
            ):
                raise ValueError(
                    "Checkpoint state/manifest acquisition gate window mismatch"
                )
            environment.load_state_dict(state["environment_state"])
            torch.set_rng_state(state["torch_rng_state"])
            random.setstate(state["python_rng_state"])
            import numpy as np

            np.random.set_state(state["numpy_rng_state"])
            if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
                torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
            _truncate_events_for_resume(events_path, completed_updates)
        else:
            model = load_base_model(config, training=True)
            target_inventory = inspect_lora_target_inventory(config, model)
            timing["model_and_tokenizer_load_wall_seconds"] += (
                time.monotonic() - load_started
            )
            adapter_started = time.monotonic()
            lora = LoraConfig(
                r=int(config["training"]["lora_rank"]),
                lora_alpha=int(config["training"]["lora_alpha"]),
                lora_dropout=float(config["training"]["lora_dropout"]),
                target_modules=list(config["training"]["lora_targets"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora)
            model_runtime_attestation = build_model_runtime_attestation(
                config, model, tokenizer, target_inventory
            )
            verify_model_runtime_attestation(config, model_runtime_attestation)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=spec.learning_rate,
                weight_decay=spec.weight_decay,
            )
            scheduler = _constant_with_warmup_scheduler(optimizer, spec)
            timing["adapter_setup_and_attestation_wall_seconds"] += (
                time.monotonic() - adapter_started
            )
            manifest["model_runtime_attestation"] = model_runtime_attestation
            manifest["timing"] = dict(timing)
            write_json(manifest_path, manifest)
            initial_checkpoint_started = time.monotonic()
            initial = _save_checkpoint(
                run_dir=target,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                environment=environment,
                completed_updates=0,
                baseline=baseline,
                baseline_initialized=baseline_initialized,
                arm=selected_arm,
                seed=pair_seed,
                config_sha256=cfg_sha256,
                spec_sha256=spec_sha256,
                environment_provenance=provenance,
                initial_environment_state_sha256=initial_environment_state_sha256,
                acquisition_diagnostics_state=acquisition_diagnostics_state,
                acquisition_gate_window_diagnostics_state=(
                    acquisition_gate_window_diagnostics_state
                ),
                model_runtime_attestation=model_runtime_attestation,
            )
            checkpoint_wall_seconds = time.monotonic() - initial_checkpoint_started
            timing["checkpoint_write_wall_seconds_total"] += checkpoint_wall_seconds
            timing["maximum_checkpoint_write_wall_seconds"] = max(
                float(timing["maximum_checkpoint_write_wall_seconds"]),
                checkpoint_wall_seconds,
            )
            timing["checkpoint_write_count"] += 1
            timing["mean_checkpoint_write_wall_seconds"] = timing[
                "checkpoint_write_wall_seconds_total"
            ] / int(timing["checkpoint_write_count"])
            manifest["checkpoints"].append(initial)
            manifest["initial_adapter_file_sha256"] = dict(initial["adapter_file_sha256"])
            manifest["timing"] = dict(timing)
            write_json(manifest_path, manifest)

        if model_runtime_attestation is None:
            raise AssertionError("Model runtime attestation was not created")
        manifest["model_runtime_attestation"] = model_runtime_attestation
        manifest["timing"] = dict(timing)
        write_json(manifest_path, manifest)

        base_config = getattr(model.get_base_model(), "config", model.config)
        if float(getattr(base_config, "attention_dropout", 0.0)) != 0.0:
            raise ValueError("Paired bridge acquisition requires zero base-model attention dropout")
        model.train()
        model.config.use_cache = False
        if optimizer is None or scheduler is None:
            raise AssertionError("Bridge optimizer/scheduler initialization failed")
        reload_probe_cases: list[Mapping[str, Any]] = []
        for update_index in range(completed_updates, spec.updates):
            if _deadline_near(config):
                reload_probe_started = time.monotonic()
                reload_probe = (
                    _make_reload_probe(
                        model,
                        tokenizer,
                        reload_probe_cases,
                        max_length=int(config["model"]["max_length"]),
                    )
                    if reload_probe_cases
                    else None
                )
                if reload_probe is not None:
                    reload_probe_wall_seconds = time.monotonic() - reload_probe_started
                    timing["reload_probe_wall_seconds_total"] += reload_probe_wall_seconds
                    timing["maximum_reload_probe_wall_seconds"] = max(
                        float(timing["maximum_reload_probe_wall_seconds"]),
                        reload_probe_wall_seconds,
                    )
                    timing["reload_probe_count"] += 1
                checkpoint_started = time.monotonic()
                checkpoint_record = _save_checkpoint(
                    run_dir=target,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    environment=environment,
                    completed_updates=completed_updates,
                    baseline=baseline,
                    baseline_initialized=baseline_initialized,
                    arm=selected_arm,
                    seed=pair_seed,
                    config_sha256=cfg_sha256,
                    spec_sha256=spec_sha256,
                    environment_provenance=provenance,
                    initial_environment_state_sha256=initial_environment_state_sha256,
                    acquisition_diagnostics_state=acquisition_diagnostics_state,
                    acquisition_gate_window_diagnostics_state=(
                        acquisition_gate_window_diagnostics_state
                    ),
                    model_runtime_attestation=model_runtime_attestation,
                    reload_probe=reload_probe,
                    optimizer_metrics=last_optimizer_metrics,
                )
                checkpoint_wall_seconds = time.monotonic() - checkpoint_started
                timing["checkpoint_write_wall_seconds_total"] += checkpoint_wall_seconds
                timing["maximum_checkpoint_write_wall_seconds"] = max(
                    float(timing["maximum_checkpoint_write_wall_seconds"]),
                    checkpoint_wall_seconds,
                )
                timing["checkpoint_write_count"] += 1
                timing["mean_checkpoint_write_wall_seconds"] = timing[
                    "checkpoint_write_wall_seconds_total"
                ] / int(timing["checkpoint_write_count"])
                if checkpoint_record not in manifest["checkpoints"]:
                    manifest["checkpoints"].append(checkpoint_record)
                manifest["completed_updates"] = completed_updates
                manifest["timing"] = dict(timing)
                manifest["result"] = {
                    "reason": "budget_soft_stop",
                    "wall_seconds": time.monotonic() - started,
                    "timing": dict(timing),
                }
                _update_manifest(target, manifest, "STOPPED_BUDGET")
                raise BridgeRunStopped("Bridge run reached the budget soft-stop and is resumable")

            update_started = time.monotonic()
            baseline_for_loss = baseline if baseline_initialized else 0.0
            optimizer.zero_grad(set_to_none=True)
            event_rows: list[dict[str, Any]] = []
            micro_metrics: list[dict[str, float]] = []
            optimized_rewards: list[float] = []
            rollouts: list[dict[str, Any]] = []
            rollout_started = time.monotonic()
            model.eval()
            for micro_index in range(spec.gradient_accumulation_steps):
                trajectory_index = update_index * spec.gradient_accumulation_steps + micro_index
                cases = validate_acquisition_batch(
                    environment.acquisition_batch(
                        trajectory_seed=int(pair_seed),
                        update_index=trajectory_index,
                        batch_size=spec.microbatch_size,
                    ),
                    spec.microbatch_size,
                )
                reload_probe_cases = list(cases)
                with torch.no_grad():
                    behavior_log_probs, behavior_raw_scores = differentiable_choice_log_probs(
                        model,
                        tokenizer,
                        cases,
                        max_length=int(config["model"]["max_length"]),
                    )
                chosen = sample_legal_actions(
                    behavior_log_probs,
                    seed=_stable_seed("bridge-action", pair_seed, trajectory_index),
                )
                actions = [CHOICES[int(index)] for index in chosen.detach().cpu().tolist()]
                outcomes = validate_realized_outcomes(
                    cases, environment.transition_batch(cases, actions)
                )
                rewards = selected_arm_rewards(
                    outcomes, selected_arm, device=behavior_log_probs.device
                )
                rollouts.append({
                    "micro_index": micro_index,
                    "trajectory_index": trajectory_index,
                    "cases": cases,
                    "chosen": chosen,
                    "actions": actions,
                    "outcomes": outcomes,
                    "rewards": rewards,
                    "behavior_log_probs": behavior_log_probs.detach(),
                    "behavior_raw_scores": behavior_raw_scores.detach(),
                })
                optimized_rewards.extend(float(value) for value in rewards.detach().cpu().tolist())

            rollout_wall_seconds = time.monotonic() - rollout_started
            gradient_started = time.monotonic()
            model.train()
            all_rewards = torch.cat([rollout["rewards"] for rollout in rollouts])
            full_advantages = rollout_advantages(
                all_rewards,
                baseline=baseline_for_loss,
                normalize=spec.normalize_advantages,
            )
            advantage_offset = 0
            for rollout in rollouts:
                cases = rollout["cases"]
                chosen = rollout["chosen"]
                actions = rollout["actions"]
                outcomes = rollout["outcomes"]
                rewards = rollout["rewards"]
                log_choice_probs, raw_scores = differentiable_choice_log_probs(
                    model,
                    tokenizer,
                    cases,
                    max_length=int(config["model"]["max_length"]),
                )
                behavior_probability_delta = float(
                    (
                        log_choice_probs.detach().exp()
                        - rollout["behavior_log_probs"].exp()
                    ).abs().max().cpu()
                )
                if behavior_probability_delta > 1e-5:
                    raise RuntimeError(
                        "Rollout and update policy differ before the optimizer step: "
                        f"{behavior_probability_delta}"
                    )
                reference_scores = (
                    _frozen_reference_choice_log_probs(
                        model,
                        tokenizer,
                        cases,
                        max_length=int(config["model"]["max_length"]),
                    )
                    if spec.kl_coefficient > 0.0
                    else None
                )
                reference_log_probs = reference_scores[0] if reference_scores is not None else None
                reference_raw_scores = reference_scores[1] if reference_scores is not None else None
                loss, metrics = policy_gradient_objective(
                    log_choice_probs,
                    chosen,
                    rewards,
                    baseline=baseline_for_loss,
                    entropy_coefficient=spec.entropy_coefficient,
                    normalize_advantages=spec.normalize_advantages,
                    reference_log_choice_probs=reference_log_probs,
                    raw_choice_log_probs=raw_scores,
                    reference_raw_choice_log_probs=reference_raw_scores,
                    kl_coefficient=spec.kl_coefficient,
                    advantages_override=full_advantages[
                        advantage_offset : advantage_offset + len(cases)
                    ],
                )
                metrics["behavior_policy_max_probability_delta"] = behavior_probability_delta
                advantage_offset += len(cases)
                (loss / spec.gradient_accumulation_steps).backward()
                micro_metrics.append(metrics)

                probabilities = rollout["behavior_log_probs"].exp().cpu()
                raw_cpu = rollout["behavior_raw_scores"].float().cpu()
                for row_index, (case, action, outcome) in enumerate(
                    zip(cases, actions, outcomes, strict=True)
                ):
                    log_mass = float(torch.logsumexp(raw_cpu[row_index], dim=0))
                    event_rows.append({
                        "schema_version": "1.0",
                        "update_index": update_index,
                        "microbatch_index": rollout["micro_index"],
                        "trajectory_index": rollout["trajectory_index"],
                        "case_id": case["case_id"],
                        "messages_sha256": hashlib.sha256(
                            canonical_json(case["messages"]).encode("utf-8")
                        ).hexdigest(),
                        "trajectory_id": case.get("trajectory_id"),
                        "world_id": case.get("world_id"),
                        "cue_regime": case.get("cue_regime"),
                        "acquisition_condition": case.get("condition"),
                        "arm": selected_arm,
                        "pair_seed": int(pair_seed),
                        "action": action,
                        "probability_A": float(probabilities[row_index, 0]),
                        "probability_B": float(probabilities[row_index, 1]),
                        "legal_choice_mass": math.exp(log_mass) if log_mass > -745 else 0.0,
                        "genuine_reward": float(outcome["genuine_reward"]),
                        "proxy_reward": float(outcome["proxy_reward"]),
                        "optimized_reward": float(outcome[ARM_REWARD_FIELD[selected_arm]]),
                        "optimal_action_success": math.isclose(
                            float(outcome[ARM_REWARD_FIELD[selected_arm]]),
                            1.0,
                            abs_tol=1e-8,
                        ),
                        "visible_feedback": outcome.get("visible_feedback"),
                    })

            gradient_wall_seconds = time.monotonic() - gradient_started
            optimizer_started = time.monotonic()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                spec.max_grad_norm,
            )
            if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
                raise FloatingPointError("Non-finite bridge gradient norm")
            optimizer.step()
            scheduler.step()
            optimizer_wall_seconds = time.monotonic() - optimizer_started
            diagnostics_started = time.monotonic()
            for rollout in rollouts:
                acquisition_diagnostics_state = update_acquisition_diagnostics(
                    acquisition_diagnostics_state,
                    rollout["cases"],
                    rollout["outcomes"],
                )
            acquisition_gate_window_diagnostics_state = (
                update_acquisition_gate_window_diagnostics(
                    acquisition_gate_window_diagnostics_state,
                    [case for rollout in rollouts for case in rollout["cases"]],
                    [outcome for rollout in rollouts for outcome in rollout["outcomes"]],
                    completed_update=update_index + 1,
                )
            )
            update_reward_mean = sum(optimized_rewards) / len(optimized_rewards)
            baseline, baseline_initialized = update_reward_baseline(
                baseline,
                update_reward_mean,
                momentum=spec.baseline_momentum,
                initialized=baseline_initialized,
            )
            diagnostics_wall_seconds = time.monotonic() - diagnostics_started
            completed_updates = update_index + 1
            update_compute_wall_seconds = time.monotonic() - update_started
            timing["update_compute_wall_seconds_total"] += update_compute_wall_seconds
            timing["rollout_wall_seconds_total"] += rollout_wall_seconds
            timing["gradient_forward_backward_wall_seconds_total"] += (
                gradient_wall_seconds
            )
            timing["optimizer_step_wall_seconds_total"] += optimizer_wall_seconds
            timing["diagnostics_wall_seconds_total"] += diagnostics_wall_seconds
            timing["completed_update_count"] = completed_updates
            timing["mean_update_compute_wall_seconds"] = timing[
                "update_compute_wall_seconds_total"
            ] / max(1, completed_updates)
            aggregate_metrics = {
                key: sum(metric[key] for metric in micro_metrics) / len(micro_metrics)
                for key in micro_metrics[0]
            }
            learning_rate = float(scheduler.get_last_lr()[0])
            last_optimizer_metrics = {
                **aggregate_metrics,
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                "learning_rate": learning_rate,
                "rollout_reward_mean": update_reward_mean,
                "rollout_wall_seconds": rollout_wall_seconds,
                "gradient_forward_backward_wall_seconds": gradient_wall_seconds,
                "optimizer_step_wall_seconds": optimizer_wall_seconds,
                "diagnostics_wall_seconds": diagnostics_wall_seconds,
                "update_compute_wall_seconds": update_compute_wall_seconds,
            }
            for event in event_rows:
                event["optimizer_metrics"] = dict(last_optimizer_metrics)
            _append_events(events_path, event_rows)

            if completed_updates in spec.checkpoint_updates:
                reload_probe_started = time.monotonic()
                reload_probe = _make_reload_probe(
                    model,
                    tokenizer,
                    reload_probe_cases,
                    max_length=int(config["model"]["max_length"]),
                )
                reload_probe_wall_seconds = time.monotonic() - reload_probe_started
                timing["reload_probe_wall_seconds_total"] += reload_probe_wall_seconds
                timing["maximum_reload_probe_wall_seconds"] = max(
                    float(timing["maximum_reload_probe_wall_seconds"]),
                    reload_probe_wall_seconds,
                )
                timing["reload_probe_count"] += 1
                checkpoint_started = time.monotonic()
                checkpoint_record = _save_checkpoint(
                    run_dir=target,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    environment=environment,
                    completed_updates=completed_updates,
                    baseline=baseline,
                    baseline_initialized=baseline_initialized,
                    arm=selected_arm,
                    seed=pair_seed,
                    config_sha256=cfg_sha256,
                    spec_sha256=spec_sha256,
                    environment_provenance=provenance,
                    initial_environment_state_sha256=initial_environment_state_sha256,
                    acquisition_diagnostics_state=acquisition_diagnostics_state,
                    acquisition_gate_window_diagnostics_state=(
                        acquisition_gate_window_diagnostics_state
                    ),
                    model_runtime_attestation=model_runtime_attestation,
                    reload_probe=reload_probe,
                    optimizer_metrics=last_optimizer_metrics,
                )
                checkpoint_wall_seconds = time.monotonic() - checkpoint_started
                timing["checkpoint_write_wall_seconds_total"] += checkpoint_wall_seconds
                timing["maximum_checkpoint_write_wall_seconds"] = max(
                    float(timing["maximum_checkpoint_write_wall_seconds"]),
                    checkpoint_wall_seconds,
                )
                timing["checkpoint_write_count"] += 1
                timing["mean_checkpoint_write_wall_seconds"] = timing[
                    "checkpoint_write_wall_seconds_total"
                ] / int(timing["checkpoint_write_count"])
                manifest["checkpoints"].append(checkpoint_record)
                manifest["completed_updates"] = completed_updates
                manifest["timing"] = dict(timing)
                write_json(manifest_path, manifest)

        finalize_started = time.monotonic()
        final_checkpoint = _latest_checkpoint(target)
        final_adapter = _materialize_final_adapter(target, final_checkpoint)
        timing["finalize_wall_seconds"] += time.monotonic() - finalize_started
        manifest["completed_updates"] = completed_updates
        manifest["timing"] = dict(timing)
        manifest["result"] = {
            "final_checkpoint": str(final_checkpoint),
            "final_adapter": str(final_adapter),
            "final_adapter_source_manifest_sha256": sha256_file(
                final_checkpoint / "checkpoint_manifest.json"
            ),
            "events_path": str(events_path),
            "events_sha256": sha256_file(events_path),
            "reward_baseline": baseline,
            "acquisition_diagnostics": acquisition_diagnostics_summary(
                acquisition_diagnostics_state
            ),
            "acquisition_gate_window_diagnostics": (
                acquisition_gate_window_diagnostics_summary(
                    acquisition_gate_window_diagnostics_state
                )
            ),
            "wall_seconds": time.monotonic() - started,
            "timing": dict(timing),
            "model_runtime_attestation_sha256": model_runtime_attestation[
                "attestation_sha256"
            ],
            "peak_vram_allocated_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            "peak_vram_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            ),
            "peak_vram_bytes": (
                max(
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                )
                if torch.cuda.is_available()
                else 0
            ),
        }
        _update_manifest(target, manifest, "COMPLETE")
        return final_adapter
    except BridgeRunStopped:
        raise
    except KeyboardInterrupt as exc:
        manifest["completed_updates"] = completed_updates
        manifest["result"] = {
            "reason": "keyboard_interrupt",
            "latest_checkpoint": str(_latest_checkpoint(target)),
            "wall_seconds": time.monotonic() - started,
        }
        _update_manifest(target, manifest, "STOPPED_EARLY")
        raise BridgeRunStopped("Bridge run interrupted; resume from the last fixed checkpoint") from exc
    except BaseException as exc:
        manifest["completed_updates"] = completed_updates
        manifest["result"] = {
            "error": repr(exc),
            "wall_seconds": time.monotonic() - started,
        }
        _update_manifest(target, manifest, "FAILED")
        raise
