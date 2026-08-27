from __future__ import annotations

import numpy as np
import pytest

from effect_only_tool_world import (
    FeasibilityOutcomes, ForkBranch, assess_feasibility_gate, centered_effects,
    collect_reset_replay_fork, validate_forks,
)


def _branch(action: str, post: tuple[float, ...], *, fork: str = "f0", state: str = "s0") -> ForkBranch:
    return ForkBranch(
        fork_id=fork,
        environment_state_id=state,
        pre_state_sha256="a" * 64,
        action_id=action,
        post_embedding=post,
    )


def test_centered_targets_remove_exact_common_mode() -> None:
    # The common [100, -5] component must disappear after centering.
    targets = centered_effects([
        _branch("left", (101.0, -4.0)),
        _branch("right", (103.0, -7.0)),
        _branch("stay", (100.0, -5.0)),
    ])
    np.testing.assert_allclose(targets[("f0", "left")], (-1 / 3, 4 / 3))
    np.testing.assert_allclose(targets[("f0", "right")], (5 / 3, -5 / 3))
    np.testing.assert_allclose(targets[("f0", "stay")], (-4 / 3, 1 / 3))
    np.testing.assert_allclose(sum(targets.values()), (0.0, 0.0), atol=1e-12, rtol=0.0)


def test_rejects_fork_with_mismatched_hidden_state() -> None:
    with pytest.raises(ValueError, match="resettable pre-state"):
        validate_forks([
            _branch("a", (0.0,)),
            _branch("b", (1.0,), state="different-state"),
        ])


def test_rejects_duplicate_action() -> None:
    with pytest.raises(ValueError, match="repeats an action"):
        validate_forks([_branch("a", (0.0,)), _branch("a", (1.0,))])


class _CounterEnvironment:
    def __init__(self, *, drift_on_replay: bool = False) -> None:
        self._value = 0
        self._drift_on_replay = drift_on_replay

    def reset(self, seed: int) -> dict[str, int]:
        self._value = seed
        return {"value": self._value}

    def step(self, action: str) -> dict[str, int]:
        self._value += {"inc": 1, "dec": -1, "wait": 0}[action]
        if self._drift_on_replay and action == "wait":
            self._value += 1
        return {"value": self._value}


def test_collector_replays_each_branch_from_the_same_prefix() -> None:
    branches = collect_reset_replay_fork(
        _CounterEnvironment,
        fork_id="counter-7",
        seed=7,
        prefix=("inc",),
        candidate_actions=("dec", "wait"),
        state_identity=lambda observation: f"counter:{observation['value']}",
        serialize_state=lambda observation: str(observation["value"]).encode(),
        embed_post_state=lambda observation: (float(observation["value"]),),
    )
    assert [branch.action_id for branch in branches] == ["dec", "wait"]
    assert {branch.environment_state_id for branch in branches} == {"counter:8"}
    assert {branch.pre_state_sha256 for branch in branches}


def test_collector_rejects_nondeterministic_prefix_state() -> None:
    calls = 0

    def factory() -> _CounterEnvironment:
        nonlocal calls
        calls += 1
        return _CounterEnvironment(drift_on_replay=calls == 2)

    with pytest.raises(ValueError, match="resettable pre-state"):
        collect_reset_replay_fork(
            factory,
            fork_id="unstable",
            seed=3,
            prefix=("wait",),
            candidate_actions=("inc", "dec"),
            state_identity=lambda observation: f"counter:{observation['value']}",
            serialize_state=lambda observation: str(observation["value"]).encode(),
            embed_post_state=lambda observation: (float(observation["value"]),),
        )


def test_frozen_gate_passes_only_with_all_required_margins() -> None:
    # One hundred matched forks: effect-only wins 10 clean ranks and 10 executed
    # tasks, while the delta target alone loses 10 extra ranks under the nuisance.
    ids = [str(index) for index in range(100)]
    effect_clean = {item: True for item in ids}
    delta_clean = {item: index < 90 for index, item in enumerate(ids)}
    effect_noisy = {item: index < 98 for index, item in enumerate(ids)}
    delta_noisy = {item: index < 73 for index, item in enumerate(ids)}
    effect_execution = {item: True for item in ids}
    baseline_execution = {item: index < 90 for index, item in enumerate(ids)}
    report = assess_feasibility_gate(FeasibilityOutcomes(
        effect_ranking=effect_clean,
        delta_ranking=delta_clean,
        effect_perturbed_ranking=effect_noisy,
        delta_perturbed_ranking=delta_noisy,
        effect_execution=effect_execution,
        strongest_baseline_execution=baseline_execution,
    ))
    assert report["pass"] is True
    assert report["decision"] == "EXPAND_TO_INDEPENDENT_TOOL_ENVIRONMENT"


def test_gate_rejects_an_unperturbed_ranking_gain_without_execution_gain() -> None:
    ids = [str(index) for index in range(100)]
    effect = {item: True for item in ids}
    delta = {item: index < 90 for index, item in enumerate(ids)}
    report = assess_feasibility_gate(FeasibilityOutcomes(
        effect_ranking=effect,
        delta_ranking=delta,
        effect_perturbed_ranking={item: index < 98 for index, item in enumerate(ids)},
        delta_perturbed_ranking={item: index < 73 for index, item in enumerate(ids)},
        effect_execution={item: index < 90 for index, item in enumerate(ids)},
        strongest_baseline_execution={item: index < 90 for index, item in enumerate(ids)},
    ))
    assert report["pass"] is False
    assert report["checks"]["execution_effect_size"] is False
