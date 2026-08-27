"""Validate resettable action forks and construct quotient supervision targets.

The module intentionally contains no browser or model dependency.  A collector
adapts an environment to :class:`ForkBranch`; this layer rejects invalid forks
before they can create false counterfactual supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Callable, Sequence
from typing import Iterable, Mapping, Protocol, TypeVar

import numpy as np


Observation = TypeVar("Observation")


class ResettableActionEnvironment(Protocol[Observation]):
    """Minimal adapter required to collect a reproducible fork.

    ``reset`` must recreate the same hidden task state for the same seed, and
    ``step`` must return the next model-visible observation.  BrowserGym adapters
    should restart their browser/task process in ``reset`` rather than reuse an
    already-mutated page.
    """

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: str) -> Observation: ...


@dataclass(frozen=True)
class ForkBranch:
    """One executable action replayed from a resettable, fixed pre-state.

    ``pre_state_sha256`` must hash the serialized state presented to every
    branch model (including task instruction and normalized observation).  The
    collector records a separate ``environment_state_id`` so an accidentally
    reused textual observation from different hidden states is not accepted as
    a counterfactual fork.
    """

    fork_id: str
    environment_state_id: str
    pre_state_sha256: str
    action_id: str
    post_embedding: tuple[float, ...]


def normalized_state_sha256(observation: Observation, serialize: Callable[[Observation], bytes]) -> str:
    """Hash exactly the normalized model-visible state used by the collector."""

    payload = serialize(observation)
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("state serializer must return non-empty bytes")
    return hashlib.sha256(payload).hexdigest()


def collect_reset_replay_fork(
    environment_factory: Callable[[], ResettableActionEnvironment[Observation]],
    *,
    fork_id: str,
    seed: int,
    prefix: Sequence[str],
    candidate_actions: Sequence[str],
    state_identity: Callable[[Observation], str],
    serialize_state: Callable[[Observation], bytes],
    embed_post_state: Callable[[Observation], Sequence[float]],
) -> tuple[ForkBranch, ...]:
    """Replay one prefix per candidate and return a validated action fork.

    Recreating the environment independently for *every* branch is intentional:
    it prevents branch order or residual browser state from defining the training
    target.  If any replay yields a different normalized pre-state or hidden
    state identity, collection aborts rather than silently mixing trajectories.
    """

    if not fork_id:
        raise ValueError("fork_id must be non-empty")
    if len(candidate_actions) < 2 or len(set(candidate_actions)) != len(candidate_actions):
        raise ValueError("candidate_actions must contain at least two unique actions")

    collected: list[ForkBranch] = []
    for action in candidate_actions:
        environment = environment_factory()
        observation = environment.reset(seed)
        for prefix_action in prefix:
            observation = environment.step(prefix_action)
        state_id = state_identity(observation)
        if not state_id:
            raise ValueError("state_identity must return a non-empty value")
        pre_hash = normalized_state_sha256(observation, serialize_state)
        post_observation = environment.step(action)
        embedding = tuple(float(value) for value in embed_post_state(post_observation))
        collected.append(ForkBranch(
            fork_id=fork_id,
            environment_state_id=state_id,
            pre_state_sha256=pre_hash,
            action_id=action,
            post_embedding=embedding,
        ))
    return validate_forks(collected)[fork_id]


def validate_forks(branches: Iterable[ForkBranch], *, min_actions: int = 2) -> Mapping[str, tuple[ForkBranch, ...]]:
    """Return canonical forks after rejecting malformed counterfactual groups."""

    if min_actions < 2:
        raise ValueError("min_actions must be at least two")

    grouped: dict[str, list[ForkBranch]] = {}
    for branch in branches:
        if not branch.fork_id:
            raise ValueError("fork_id must be non-empty")
        if not branch.environment_state_id or not branch.pre_state_sha256:
            raise ValueError(f"fork {branch.fork_id!r} lacks a state identity")
        if not branch.action_id:
            raise ValueError(f"fork {branch.fork_id!r} contains an empty action id")
        vector = np.asarray(branch.post_embedding, dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"fork {branch.fork_id!r} has an invalid post embedding")
        grouped.setdefault(branch.fork_id, []).append(branch)

    canonical: dict[str, tuple[ForkBranch, ...]] = {}
    for fork_id, members in grouped.items():
        state_ids = {member.environment_state_id for member in members}
        pre_hashes = {member.pre_state_sha256 for member in members}
        actions = [member.action_id for member in members]
        dimensions = {len(member.post_embedding) for member in members}
        if len(state_ids) != 1 or len(pre_hashes) != 1:
            raise ValueError(f"fork {fork_id!r} does not share one resettable pre-state")
        if len(members) < min_actions:
            raise ValueError(f"fork {fork_id!r} has fewer than {min_actions} actions")
        if len(actions) != len(set(actions)):
            raise ValueError(f"fork {fork_id!r} repeats an action")
        if len(dimensions) != 1:
            raise ValueError(f"fork {fork_id!r} mixes post-embedding dimensions")
        canonical[fork_id] = tuple(sorted(members, key=lambda member: member.action_id))
    if not canonical:
        raise ValueError("no branches supplied")
    return canonical


def centered_effects(branches: Iterable[ForkBranch]) -> Mapping[tuple[str, str], np.ndarray]:
    """Construct ``z_i - mean_j(z_j)`` targets for each validated fork.

    The output keys are ``(fork_id, action_id)``.  Summing targets over the
    actions of any fork is exactly zero up to floating-point roundoff, which is
    checked explicitly so collector serialization errors cannot silently change
    the quotient target.
    """

    targets: dict[tuple[str, str], np.ndarray] = {}
    for fork_id, members in validate_forks(branches).items():
        matrix = np.asarray([member.post_embedding for member in members], dtype=np.float64)
        effects = matrix - matrix.mean(axis=0, keepdims=True)
        if not np.allclose(effects.sum(axis=0), 0.0, atol=1e-10, rtol=0.0):
            raise RuntimeError(f"fork {fork_id!r} failed quotient-centering invariant")
        for member, effect in zip(members, effects, strict=True):
            targets[(fork_id, member.action_id)] = effect
    return targets
