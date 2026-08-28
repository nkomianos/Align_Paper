"""Frozen, fail-closed statistics for shadow-student transfer forecasting.

This module intentionally consumes already-computed, signed behavioral effects.
It does not fit a detector, inspect sealed rows, or choose a threshold from the
held-out set.  A caller is responsible for binding raw training/evaluation
artifacts to each :class:`Scenario` before invoking this decision rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


REQUIRED_BASELINES = frozenset({"initial_update", "token_divergence", "one_shadow", "random_retention"})
REQUIRED_CHANNELS = frozenset({"vocabulary", "body"})


@dataclass(frozen=True)
class Scenario:
    """A data-batch outcome bound to a predeclared G0 split and channel."""

    scenario_id: str
    split: str  # ``calibration`` or ``sealed``
    channel: str  # ``vocabulary`` or ``body``
    full_seed_effects: tuple[float, float]
    shadow_effects: tuple[float, ...]
    baseline_scores: Mapping[str, float]
    full_gpu_hours: float
    shadow_gpu_hours: float

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id is required")
        if self.split not in {"calibration", "sealed"}:
            raise ValueError("split must be calibration or sealed")
        if self.channel not in REQUIRED_CHANNELS:
            raise ValueError("channel must be vocabulary or body")
        if len(self.full_seed_effects) != 2 or len(self.shadow_effects) < 2:
            raise ValueError("exactly two full seeds and at least two shadows are required")
        if not REQUIRED_BASELINES.issubset(self.baseline_scores):
            raise ValueError("every required baseline score is required")
        if any(not np.isfinite(value) for value in (*self.full_seed_effects, *self.shadow_effects)):
            raise ValueError("effects must be finite")
        if any(not np.isfinite(float(value)) for value in self.baseline_scores.values()):
            raise ValueError("baseline scores must be finite")
        if self.full_gpu_hours <= 0 or self.shadow_gpu_hours <= 0:
            raise ValueError("GPU hours must be positive")

    @property
    def full_effect(self) -> float:
        return float(np.mean(self.full_seed_effects))

    @property
    def shadow_score(self) -> float:
        return float(np.mean(self.shadow_effects))


@dataclass(frozen=True)
class Thresholds:
    """Frozen G0 conditions from the candidate protocol."""

    minimum_full_effect: float = 0.10
    minimum_spearman: float = 0.70
    minimum_spearman_lcb: float = 0.50
    minimum_recall: float = 0.80
    maximum_fpr: float = 0.20
    minimum_baseline_margin: float = 0.10
    maximum_shadow_compute_fraction: float = 0.20
    bootstrap_samples: int = 4_000
    bootstrap_seed: int = 20260828


@dataclass(frozen=True)
class GateDecision:
    """Immutable G0 report; ``pass_gate`` is true only when every condition holds."""

    pass_gate: bool
    decision: str
    calibrated_threshold: float | None
    sealed_spearman: float | None
    sealed_spearman_lcb: float | None
    sealed_recall: float | None
    sealed_fpr: float | None
    sentry_auc: float | None
    baseline_aucs: Mapping[str, float]
    compute_fraction: float | None
    channel_reproducible: Mapping[str, bool]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _labels(records: Sequence[Scenario], threshold: float) -> np.ndarray:
    """A positive must be signed and replicated by both full training seeds."""

    return np.asarray(
        [all(effect >= threshold for effect in record.full_seed_effects) for record in records], dtype=int
    )


def _calibrate_threshold(records: Sequence[Scenario], labels: np.ndarray, maximum_fpr: float) -> float | None:
    """Choose once on calibration rows: maximal recall, then stricter threshold."""

    scores = np.asarray([record.shadow_score for record in records])
    candidates = np.unique(scores)
    eligible: list[tuple[float, float]] = []
    for threshold in candidates:
        predicted = scores >= threshold
        positives, negatives = labels == 1, labels == 0
        recall = float(np.mean(predicted[positives])) if np.any(positives) else 0.0
        fpr = float(np.mean(predicted[negatives])) if np.any(negatives) else 0.0
        if fpr <= maximum_fpr:
            eligible.append((recall, float(threshold)))
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item[0], item[1]))[1]


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return None if len(np.unique(labels)) < 2 else float(roc_auc_score(labels, scores))


def _spearman_lcb(records: Sequence[Scenario], thresholds: Thresholds) -> tuple[float | None, float | None]:
    effects = np.asarray([record.full_effect for record in records])
    scores = np.asarray([record.shadow_score for record in records])
    if len(np.unique(scores)) < 2 or len(np.unique(effects)) < 2:
        return None, None
    rho = float(spearmanr(scores, effects).statistic)
    if len(records) < 4 or not np.isfinite(rho):
        return None, None
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    bootstrap = []
    for _ in range(thresholds.bootstrap_samples):
        indices = rng.integers(0, len(records), len(records))
        if len(np.unique(scores[indices])) < 2 or len(np.unique(effects[indices])) < 2:
            continue
        value = float(spearmanr(scores[indices], effects[indices]).statistic)
        if np.isfinite(value):
            bootstrap.append(value)
    return rho, (float(np.quantile(bootstrap, 0.05)) if bootstrap else None)


def evaluate_gate(records: Sequence[Scenario], thresholds: Thresholds = Thresholds()) -> GateDecision:
    """Evaluate the preregistered SENTRY G0 without looking inside artifacts.

    The rule deliberately fails closed on too few rows, missing positive or
    negative controls, degenerate rankings, or a missing channel.
    """

    if len({record.scenario_id for record in records}) != len(records):
        raise ValueError("scenario identifiers must be unique")
    calibration = tuple(record for record in records if record.split == "calibration")
    sealed = tuple(record for record in records if record.split == "sealed")
    failures: list[str] = []
    if len(calibration) < 4 or len(sealed) < 4:
        failures.append("need at least four calibration and four sealed scenarios")
    if {record.channel for record in sealed} != REQUIRED_CHANNELS:
        failures.append("sealed scenarios must include vocabulary and body channels")
    calibration_labels = _labels(calibration, thresholds.minimum_full_effect)
    if len(np.unique(calibration_labels)) < 2:
        failures.append("calibration needs both positive and neutral full outcomes")
    threshold = _calibrate_threshold(calibration, calibration_labels, thresholds.maximum_fpr) if not failures else None
    if threshold is None:
        failures.append("no calibration threshold meets the frozen FPR condition")

    sealed_labels = _labels(sealed, thresholds.minimum_full_effect)
    scores = np.asarray([record.shadow_score for record in sealed])
    predicted = scores >= threshold if threshold is not None else np.zeros(len(sealed), dtype=bool)
    positive, negative = sealed_labels == 1, sealed_labels == 0
    recall = float(np.mean(predicted[positive])) if np.any(positive) else None
    fpr = float(np.mean(predicted[negative])) if np.any(negative) else None
    if recall is None or fpr is None:
        failures.append("sealed set needs both positive and neutral full outcomes")
    elif recall < thresholds.minimum_recall or fpr > thresholds.maximum_fpr:
        failures.append("sealed recall/FPR condition failed")

    rho, rho_lcb = _spearman_lcb(sealed, thresholds)
    if rho is None or rho_lcb is None or rho < thresholds.minimum_spearman or rho_lcb < thresholds.minimum_spearman_lcb:
        failures.append("sealed rank-forecast condition failed")

    sentry_auc = _safe_auc(sealed_labels, scores)
    baseline_aucs = {
        name: _safe_auc(sealed_labels, np.asarray([record.baseline_scores[name] for record in sealed]))
        for name in sorted(REQUIRED_BASELINES)
    }
    if sentry_auc is None or any(value is None for value in baseline_aucs.values()):
        failures.append("AUC comparison is degenerate")
    elif any(sentry_auc < float(value) + thresholds.minimum_baseline_margin for value in baseline_aucs.values()):
        failures.append("SENTRY does not beat every baseline by the frozen margin")

    reproducible = {
        channel: any(
            record.channel == channel and all(effect >= thresholds.minimum_full_effect for effect in record.full_seed_effects)
            for record in sealed
        )
        for channel in REQUIRED_CHANNELS
    }
    if not all(reproducible.values()):
        failures.append("no reproducible full transfer in every required sealed channel")

    compute_fraction = (
        float(sum(record.shadow_gpu_hours for record in records) / sum(record.full_gpu_hours for record in records))
        if records else None
    )
    if compute_fraction is None or compute_fraction > thresholds.maximum_shadow_compute_fraction:
        failures.append("shadow compute fraction exceeds frozen budget")

    return GateDecision(
        pass_gate=not failures,
        decision="PROCEED_TO_EXTERNAL_REPLICATION" if not failures else "KILL_SHADOW_STUDENT_CANDIDATE",
        calibrated_threshold=threshold,
        sealed_spearman=rho,
        sealed_spearman_lcb=rho_lcb,
        sealed_recall=recall,
        sealed_fpr=fpr,
        sentry_auc=sentry_auc,
        baseline_aucs={name: float(value) if value is not None else float("nan") for name, value in baseline_aucs.items()},
        compute_fraction=compute_fraction,
        channel_reproducible=reproducible,
        failures=tuple(failures),
    )
