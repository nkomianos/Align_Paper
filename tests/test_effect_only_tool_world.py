from __future__ import annotations

import numpy as np
import pytest

from effect_only_tool_world import ForkBranch, centered_effects, validate_forks


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
