"""Frozen metrics for the environment-effect uncertainty gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .corpus import EffectCase
from .environment import action_signature, execute_plan, parse_plan


BASELINES = ("raw_consistency", "action_consistency", "tool_consistency", "token_confidence")


@dataclass(frozen=True)
class GateThresholds:
    auc_margin: float = 0.05
    vote_margin: float = 0.03
    alias_auc_margin: float = 0.08
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 20260829


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(set(labels)) != 2:
        raise ValueError("AUROC requires both successful and failed reference plans")
    return float(roc_auc_score(labels, scores))


def _mode(values: Sequence[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def _agreement(reference: str, values: Sequence[str]) -> float:
    return sum(value == reference for value in values) / len(values)


def _tool_signature(plan: Sequence[Any]) -> str:
    return "|".join(action.tool for action in plan)


def materialize_rows(cases: Sequence[EffectCase], raw: Iterable[Mapping[str, Any]], answer_key: Mapping[str, str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in raw:
        grouped.setdefault(str(record["task_id"]), []).append(record)
    rows: list[dict[str, Any]] = []
    for case in cases:
        records = sorted(grouped.get(case.task_id, []), key=lambda row: int(row["sample_id"]))
        if len(records) < 4 or int(records[0]["sample_id"]) != 0:
            raise ValueError(f"{case.task_id} needs reference sample 0 and at least three stochastic samples")
        plans, effects, raw_texts, action_sigs, tool_sigs = [], [], [], [], []
        for record in records:
            text = str(record["completion"])
            try:
                plan = parse_plan(text)
            except ValueError as exc:
                plan = ()
                effect = f"INVALID:PARSE:{exc}"
            else:
                effect = execute_plan(case.domain, case.initial_state, plan).effect
            plans.append(plan)
            effects.append(effect)
            raw_texts.append(" ".join(text.split()))
            action_sigs.append(action_signature(plan))
            tool_sigs.append(_tool_signature(plan))
        oracle = answer_key[case.task_id]
        sampled = slice(1, None)
        effect_vote = _mode(effects)
        action_vote = _mode(action_sigs)
        selected_action_index = action_sigs.index(action_vote)
        rows.append({
            "task_id": case.task_id,
            "domain": case.domain,
            "split": case.split,
            "interface": case.interface,
            "stratum": case.stratum,
            "label": int(effects[0] == oracle),
            "effect_consistency": _agreement(effects[0], effects[sampled]),
            "raw_consistency": _agreement(raw_texts[0], raw_texts[sampled]),
            "action_consistency": _agreement(action_sigs[0], action_sigs[sampled]),
            "tool_consistency": _agreement(tool_sigs[0], tool_sigs[sampled]),
            "token_confidence": float(records[0].get("token_confidence", 0.0)),
            "effect_vote_correct": int(effect_vote == oracle),
            "action_vote_correct": int(effects[selected_action_index] == oracle),
        })
    return rows


def _bootstrap_auc_gap(rows: Sequence[Mapping[str, Any]], left: str, right: str, *, replicates: int, seed: int) -> dict[str, float]:
    labels = [int(row["label"]) for row in rows]
    point = _auc(labels, [float(row[left]) for row in rows]) - _auc(labels, [float(row[right]) for row in rows])
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, len(rows), len(rows))
        draw = [rows[int(index)] for index in indices]
        draw_labels = [int(row["label"]) for row in draw]
        if len(set(draw_labels)) != 2:
            continue
        gaps.append(_auc(draw_labels, [float(row[left]) for row in draw]) - _auc(draw_labels, [float(row[right]) for row in draw]))
    if len(gaps) < replicates // 2:
        raise ValueError("too few valid bootstrap AUROC draws")
    return {"point": point, "ci95_low": float(np.quantile(gaps, .025)), "ci95_high": float(np.quantile(gaps, .975))}


def _bootstrap_binary_gap(rows: Sequence[Mapping[str, Any]], left: str, right: str, *, replicates: int, seed: int) -> dict[str, float]:
    differences = np.asarray([float(row[left]) - float(row[right]) for row in rows])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(replicates, len(rows)))
    distribution = differences[draws].mean(axis=1)
    return {"point": float(differences.mean()), "ci95_low": float(np.quantile(distribution, .025)), "ci95_high": float(np.quantile(distribution, .975))}


def score_family(cases: Sequence[EffectCase], raw: Iterable[Mapping[str, Any]], answer_key: Mapping[str, str], *, thresholds: GateThresholds = GateThresholds()) -> dict[str, Any]:
    rows = materialize_rows(cases, raw, answer_key)
    test = [row for row in rows if row["split"] == "TEST"]
    domains: dict[str, Any] = {}
    for domain in sorted({row["domain"] for row in test}):
        subset = [row for row in test if row["domain"] == domain]
        baseline_aucs = {name: _auc([row["label"] for row in subset], [row[name] for row in subset]) for name in BASELINES}
        strongest = max(baseline_aucs, key=lambda name: (baseline_aucs[name], name))
        domains[domain] = {
            "count": len(subset),
            "reference_accuracy": float(np.mean([row["label"] for row in subset])),
            "effect_auc": _auc([row["label"] for row in subset], [row["effect_consistency"] for row in subset]),
            "baseline_aucs": baseline_aucs,
            "strongest_baseline": strongest,
            "auc_gap": _bootstrap_auc_gap(subset, "effect_consistency", strongest, replicates=thresholds.bootstrap_replicates, seed=thresholds.bootstrap_seed + len(domains)),
            "vote_gap": _bootstrap_binary_gap(subset, "effect_vote_correct", "action_vote_correct", replicates=thresholds.bootstrap_replicates, seed=thresholds.bootstrap_seed + 100 + len(domains)),
        }
    alias = [row for row in test if row["interface"] == "alias_rich"]
    alias_baselines = {name: _auc([row["label"] for row in alias], [row[name] for row in alias]) for name in BASELINES}
    strongest_alias = max(alias_baselines, key=lambda name: (alias_baselines[name], name))
    alias_gap = _bootstrap_auc_gap(alias, "effect_consistency", strongest_alias, replicates=thresholds.bootstrap_replicates, seed=thresholds.bootstrap_seed + 200)
    return {"rows": rows, "domains": domains, "alias_gap": alias_gap, "strongest_alias_baseline": strongest_alias}


def evaluate_gate(families: Mapping[str, Mapping[str, Any]], *, thresholds: GateThresholds = GateThresholds()) -> dict[str, Any]:
    if len(families) < 2:
        raise ValueError("the gate requires two independent model families")
    checks: dict[str, bool] = {}
    for family, report in families.items():
        for domain, cell in report["domains"].items():
            gap, vote = cell["auc_gap"], cell["vote_gap"]
            checks[f"{family}/{domain}/auc"] = gap["point"] >= thresholds.auc_margin and gap["ci95_low"] > 0
            checks[f"{family}/{domain}/vote"] = vote["point"] >= thresholds.vote_margin and vote["ci95_low"] > 0
        alias = report["alias_gap"]
        checks[f"{family}/alias"] = alias["point"] >= thresholds.alias_auc_margin and alias["ci95_low"] > 0
    passed = all(checks.values())
    return {
        "pass": passed,
        "decision": "EXPAND_EFFECT_CONSISTENCY" if passed else "KILL_EFFECT_CONSISTENCY",
        "checks": checks,
        "families": dict(families),
    }
