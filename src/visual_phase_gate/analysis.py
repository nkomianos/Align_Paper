"""Frozen phase-instability and compute-matched ensemble statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .corpus import PhaseCase


@dataclass(frozen=True)
class PhaseThresholds:
    thin_flip_rate: float = .15
    thin_minus_thick_flip: float = .05
    periodic_excess: float = .03
    ensemble_margin: float = .05
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 20260829


def normalize_answer(text: str, task: str) -> str:
    value = text.strip().lower()
    if task == "count":
        match = re.search(r"(?<!\d)(\d+)(?!\d)", value)
        return match.group(1) if match else "INVALID"
    match = re.search(r"\b(yes|no)\b", value)
    return match.group(1) if match else "INVALID"


def _mode(values: Sequence[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def _bootstrap_gap(values: Sequence[float], *, replicates: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(replicates, len(array)))
    distribution = array[draws].mean(axis=1)
    return {"point": float(array.mean()), "ci95_low": float(np.quantile(distribution, .025)), "ci95_high": float(np.quantile(distribution, .975))}


def score_family(cases: Sequence[PhaseCase], raw: Iterable[Mapping[str, Any]], *, expected_periods: Sequence[int], thresholds: PhaseThresholds = PhaseThresholds()) -> dict[str, Any]:
    case_by_id = {case.image_id: case for case in cases}
    records: dict[tuple[str, int], str] = {}
    for row in raw:
        image_id, sample_id = str(row["image_id"]), int(row["sample_id"])
        if image_id not in case_by_id or (image_id, sample_id) in records:
            raise ValueError("unknown or duplicate visual-phase result")
        records[(image_id, sample_id)] = normalize_answer(str(row["completion"]), case_by_id[image_id].task)
    if any((case.image_id, 0) not in records for case in cases):
        raise ValueError("every phase image requires a deterministic sample")
    grouped: dict[tuple[str, str], list[PhaseCase]] = {}
    for case in cases:
        grouped.setdefault((case.base_id, case.thickness), []).append(case)
    flips: dict[str, list[float]] = {"thin": [], "thick": []}
    lag_scores: dict[int, list[float]] = {period: [] for period in expected_periods}
    ensemble_differences: list[float] = []
    rows: list[dict[str, Any]] = []
    for (base_id, thickness), members in grouped.items():
        members.sort(key=lambda case: case.phase_x)
        answers = [records[(case.image_id, 0)] for case in members]
        flips[thickness].append(float(len(set(answers)) > 1))
        if thickness == "thin":
            for period in expected_periods:
                if period + 1 >= len(answers):
                    continue
                same = np.mean([answers[index] == answers[index + period] for index in range(len(answers) - period)])
                neighbors = []
                for lag in (period - 1, period + 1):
                    neighbors.append(np.mean([answers[index] == answers[index + lag] for index in range(len(answers) - lag)]))
                lag_scores[period].append(float(same - np.mean(neighbors)))
        first = members[0]
        if first.split == "TEST" and thickness == "thin":
            period = expected_periods[0]
            offsets = sorted({0, period // 4, period // 2, 3 * period // 4})
            phase_answers = [answers[offset] for offset in offsets if offset < len(answers)]
            phase_vote = _mode(phase_answers)
            same_image = [records.get((first.image_id, sample_id), records[(first.image_id, 0)]) for sample_id in range(4)]
            same_vote = _mode(same_image)
            correct = first.answer
            difference = float(phase_vote == correct) - float(same_vote == correct)
            ensemble_differences.append(difference)
            rows.append({"base_id": base_id, "phase_vote": phase_vote, "same_image_vote": same_vote, "answer": correct, "difference": difference})
    thin_rate, thick_rate = float(np.mean(flips["thin"])), float(np.mean(flips["thick"]))
    periodic = {str(period): float(np.mean(values)) if values else float("nan") for period, values in lag_scores.items()}
    best_period = max(periodic, key=lambda key: periodic[key])
    ensemble = _bootstrap_gap(ensemble_differences, replicates=thresholds.bootstrap_replicates, seed=thresholds.bootstrap_seed)
    return {
        "thin_flip_rate": thin_rate,
        "thick_flip_rate": thick_rate,
        "thin_minus_thick_flip": thin_rate - thick_rate,
        "periodic_excess": periodic,
        "best_expected_period": int(best_period),
        "best_periodic_excess": periodic[best_period],
        "ensemble_gap": ensemble,
        "test_rows": rows,
    }


def evaluate_gate(families: Mapping[str, Mapping[str, Any]], *, thresholds: PhaseThresholds = PhaseThresholds()) -> dict[str, Any]:
    if len(families) < 2:
        raise ValueError("visual-phase G0 requires two independent VLM families")
    checks: dict[str, bool] = {}
    for family, report in families.items():
        checks[f"{family}/thin_flip"] = report["thin_flip_rate"] >= thresholds.thin_flip_rate
        checks[f"{family}/frequency_control"] = report["thin_minus_thick_flip"] >= thresholds.thin_minus_thick_flip
        checks[f"{family}/phase_lock"] = report["best_periodic_excess"] >= thresholds.periodic_excess
        gap = report["ensemble_gap"]
        checks[f"{family}/ensemble"] = gap["point"] >= thresholds.ensemble_margin and gap["ci95_low"] > 0
    passed = all(checks.values())
    return {"pass": passed, "decision": "EXPAND_PATCH_PHASE_STUDY" if passed else "KILL_PATCH_PHASE_STUDY", "checks": checks, "families": dict(families)}
