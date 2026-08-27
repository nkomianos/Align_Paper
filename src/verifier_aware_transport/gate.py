"""A preregistered CPU feasibility gate for verifier-aware exact transports.

The sampler only changes the order in which categorical probability bins occupy
the unit interval.  Bin widths never change, so a uniform input still induces
the original categorical marginal.  A randomized lattice supplies correlated
but marginally uniform inputs; a frozen proxy orders bins before decoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np


OrderKind = Literal["probability", "proxy", "random"]


@dataclass(frozen=True)
class Scenario:
    """A finite completion distribution with a frozen imperfect verifier proxy."""

    name: str
    probabilities: tuple[float, ...]
    utilities: tuple[float, ...]
    proxy_scores: tuple[float, ...]
    batch_size: int
    selection_noise: float

    def __post_init__(self) -> None:
        size = len(self.probabilities)
        if size < 2 or len(self.utilities) != size or len(self.proxy_scores) != size:
            raise ValueError("probabilities, utilities, and proxy_scores must have equal length >= 2")
        if not np.isclose(sum(self.probabilities), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("probabilities must sum to one")
        if any(probability <= 0.0 for probability in self.probabilities):
            raise ValueError("probabilities must be strictly positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least two")
        if self.selection_noise < 0.0:
            raise ValueError("selection_noise must be non-negative")


@dataclass(frozen=True)
class TransportGateReport:
    """Result of the predeclared mathematical feasibility gate."""

    scenarios: tuple[dict[str, object], ...]
    marginal_max_error: float
    minimum_utility_gain: float
    minimum_success_gain: float
    pass_gate: bool
    decision: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def ordering(scenario: Scenario, kind: OrderKind, rng: np.random.Generator) -> np.ndarray:
    """Return a predictable bin permutation, never conditioned on a draw."""

    probabilities = np.asarray(scenario.probabilities)
    proxy_scores = np.asarray(scenario.proxy_scores)
    if kind == "probability":
        return np.argsort(-probabilities, kind="stable")
    if kind == "proxy":
        # High proxy values are contiguous: this is the coverage-maximising
        # arrangement for a high-value set under a translated one-dimensional lattice.
        return np.argsort(proxy_scores, kind="stable")
    if kind == "random":
        return rng.permutation(len(probabilities))
    raise ValueError(f"unknown order kind: {kind}")


def lattice_uniforms(batch_size: int, offset: float) -> np.ndarray:
    """One randomly translated regular lattice; each coordinate is uniform."""

    if not 0.0 <= offset < 1.0:
        raise ValueError("offset must be in [0, 1)")
    return (offset + np.arange(batch_size, dtype=float) / batch_size) % 1.0


def decode_categorical(
    probabilities: Sequence[float], order: Sequence[int], uniforms: Sequence[float]
) -> np.ndarray:
    """Inverse-CDF decode with an arbitrary measure-preserving bin order."""

    probabilities_array = np.asarray(probabilities, dtype=float)
    order_array = np.asarray(order, dtype=int)
    uniforms_array = np.asarray(uniforms, dtype=float)
    if sorted(order_array.tolist()) != list(range(len(probabilities_array))):
        raise ValueError("order must be a permutation of category indices")
    if np.any(uniforms_array < 0.0) or np.any(uniforms_array >= 1.0):
        raise ValueError("uniforms must lie in [0, 1)")
    cdf = np.cumsum(probabilities_array[order_array])
    positions = np.searchsorted(cdf, uniforms_array, side="right")
    return order_array[positions]


def _simulate(
    scenario: Scenario,
    kind: OrderKind,
    *,
    trials: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    fixed_order = ordering(scenario, kind, rng)
    probabilities = np.asarray(scenario.probabilities)
    utilities = np.asarray(scenario.utilities)
    proxy_scores = np.asarray(scenario.proxy_scores)
    selected_utility = np.empty(trials, dtype=float)
    selected_success = np.empty(trials, dtype=float)
    marginal_counts = np.zeros(len(probabilities), dtype=float)

    for trial in range(trials):
        current_order = fixed_order if kind != "random" else ordering(scenario, kind, rng)
        samples = decode_categorical(
            probabilities,
            current_order,
            lattice_uniforms(scenario.batch_size, float(rng.random())),
        )
        marginal_counts += np.bincount(samples, minlength=len(probabilities))
        noisy_scores = proxy_scores[samples] + rng.normal(
            0.0, scenario.selection_noise, size=scenario.batch_size
        )
        choice = int(samples[int(np.argmax(noisy_scores))])
        selected_utility[trial] = utilities[choice]
        selected_success[trial] = float(utilities[choice] >= 0.5)

    marginal_observed = marginal_counts / (trials * scenario.batch_size)
    return {
        "mean_selected_utility": float(np.mean(selected_utility)),
        "selected_success": float(np.mean(selected_success)),
        "marginal_max_error": float(np.max(np.abs(marginal_observed - probabilities))),
    }


def default_scenarios() -> tuple[Scenario, ...]:
    """Two non-monotone probability/value landscapes; frozen before evaluation."""

    return (
        Scenario(
            name="alternating-eight",
            probabilities=(0.14, 0.13, 0.14, 0.13, 0.12, 0.11, 0.12, 0.11),
            utilities=(1.0, 0.0, 0.9, 0.0, 0.8, 0.0, 0.7, 0.0),
            proxy_scores=(0.85, 0.15, 0.80, 0.20, 0.75, 0.10, 0.70, 0.25),
            batch_size=4,
            selection_noise=0.35,
        ),
        Scenario(
            name="heterogeneous-twelve",
            probabilities=(0.17, 0.08, 0.15, 0.07, 0.13, 0.06, 0.11, 0.05, 0.08, 0.04, 0.04, 0.02),
            utilities=(0.0, 1.0, 0.0, 0.9, 0.0, 0.8, 0.0, 0.7, 0.0, 0.6, 0.0, 0.5),
            proxy_scores=(0.20, 0.85, 0.10, 0.80, 0.25, 0.75, 0.15, 0.70, 0.30, 0.65, 0.05, 0.60),
            batch_size=8,
            selection_noise=0.40,
        ),
    )


def run_feasibility_gate(*, trials: int = 20_000, seed: int = 20260826) -> TransportGateReport:
    """Test exactness and whether proxy ordering beats probability ordering.

    Passing only establishes a synthetic mathematical mechanism.  It does not
    green-light a paper or a GPU study; the empirical gate must reproduce the
    effect with a frozen learned proxy on real model completions.
    """

    if trials < 1_000:
        raise ValueError("at least 1,000 trials are required for the feasibility gate")
    reports: list[dict[str, object]] = []
    utility_gains: list[float] = []
    success_gains: list[float] = []
    marginal_errors: list[float] = []
    for index, scenario in enumerate(default_scenarios()):
        probability = _simulate(scenario, "probability", trials=trials, seed=seed + 10 * index)
        proxy = _simulate(scenario, "proxy", trials=trials, seed=seed + 10 * index)
        random = _simulate(scenario, "random", trials=trials, seed=seed + 10 * index)
        utility_gain = proxy["mean_selected_utility"] - probability["mean_selected_utility"]
        success_gain = proxy["selected_success"] - probability["selected_success"]
        utility_gains.append(utility_gain)
        success_gains.append(success_gain)
        marginal_errors.extend(result["marginal_max_error"] for result in (probability, proxy, random))
        reports.append({
            "name": scenario.name,
            "probability_order": probability,
            "proxy_order": proxy,
            "random_order": random,
            "utility_gain_over_probability": utility_gain,
            "success_gain_over_probability": success_gain,
        })

    minimum_utility_gain = min(utility_gains)
    minimum_success_gain = min(success_gains)
    marginal_max_error = max(marginal_errors)
    passes = (
        marginal_max_error <= 0.01
        and minimum_utility_gain >= 0.05
        and minimum_success_gain >= 0.05
    )
    return TransportGateReport(
        scenarios=tuple(reports),
        marginal_max_error=marginal_max_error,
        minimum_utility_gain=minimum_utility_gain,
        minimum_success_gain=minimum_success_gain,
        pass_gate=passes,
        decision=(
            "PROCEED_TO_PREFIX_LOCAL_PROXY_GATE"
            if passes
            else "KILL_TRANSPORT_HYPOTHESIS_BEFORE_GPU"
        ),
    )
