"""Exploratory certificate/data audit; no LLM inference or paper go/no-go gate.

CP is applied only as an explicitly labelled comparator. The clustered
counterexample is an exact probability calculation, not empirical agent data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom


def cp_lower(k: int, n: int, alpha: float = .05) -> float:
    if not 0 < alpha < 1 or not 0 <= k <= n or n < 1:
        raise ValueError("invalid binomial counts/level")
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n-k+1))


def exact_cluster_case(tasks: int, repeats: int, survival: float,
                       target: float, alpha: float = .05) -> dict:
    """Every baseline succeeds. A fixed gate accepts all/none by task type.

    Task types are iid Bernoulli(survival); repeats within task are identical.
    This is also conditionally iid given task (degenerate Bernoulli trials).
    Thus new-task retained-success recall equals survival exactly.
    """
    if tasks < 1 or repeats < 1 or not 0 < survival < target < 1:
        raise ValueError("requires positive counts and survival < target")
    mass = binom.pmf(np.arange(tasks+1), tasks, survival)
    pooled = np.array([cp_lower(int(k*repeats), tasks*repeats, alpha)
                       for k in range(tasks+1)])
    taskwise = np.array([cp_lower(int(k), tasks, alpha)
                         for k in range(tasks+1)])
    # Positive control: really iid episodes, same total count, same recall.
    n = tasks*repeats
    iid_bounds = np.array([cp_lower(k, n, alpha) for k in range(n+1)])
    iid_mass = binom.pmf(np.arange(n+1), n, survival)
    return {
        "tasks": tasks, "repeats": repeats, "true_recall": survival,
        "target": target, "alpha": alpha,
        "pooled_false_certificate_probability": float(mass[pooled >= target].sum()),
        "taskwise_false_certificate_probability": float(mass[taskwise >= target].sum()),
        "iid_episode_false_certificate_probability": float(iid_mass[iid_bounds >= target].sum()),
        "all_survive_probability": survival**tasks,
        "pooled_all_survive_lower": float(pooled[-1]),
        "taskwise_all_survive_lower": float(taskwise[-1]),
    }


def cluster_hoeffding_certificate(success: np.ndarray, retained: np.ndarray,
                                  target: float, alpha: float = .05) -> dict:
    """Known baseline, NOT a new method. Fixed m iid-task cluster design.

    Target R = E_task E_rollout[Y*A|task] / E_task E_rollout[Y|task].
    Let D_i = mean_j(Y_ij*A_ij - target*Y_ij), in [-target, 1-target].
    Independent tasks + a gate frozen before certification imply one-sided
    Hoeffding P(E[D] < mean(D)-sqrt(log(1/alpha)/(2*n))) <= alpha.
    E[Y]>0 is required to interpret a ratio; positive lower bound implies it.
    Arbitrary within-task dependence is allowed; unequal/adaptive task weights
    are deliberately not implemented.
    """
    y, a = np.asarray(success), np.asarray(retained)
    if y.shape != a.shape or y.ndim != 2 or min(y.shape) < 1:
        raise ValueError("matching nonempty task x repeat arrays required")
    if not 0 < target < 1 or not 0 < alpha < 1:
        raise ValueError("invalid target/level")
    if not np.isin(y, [0, 1]).all() or not np.isin(a, [0, 1]).all():
        raise ValueError("binary observations required")
    d = (y*a - target*y).mean(axis=1)
    lower = float(d.mean() - np.sqrt(np.log(1/alpha)/(2*len(d))))
    return {"tasks": len(d), "repeats": y.shape[1], "mean_contrast": float(d.mean()),
            "lower_contrast": lower, "certified": lower >= 0,
            "observed_recall": float((y*a).sum()/y.sum()) if y.sum() else None}


def audit_data(root: Path) -> dict:
    with gzip.open(root / "replay_index.jsonl.gz", "rt", encoding="utf8") as f:
        replay = [json.loads(line) for line in f if line.strip()]
    hazard = json.loads((root / "hazard_train.json").read_text(encoding="utf8"))
    def label(x):
        value = x.get("resolved")
        if value is None:
            return "unscored"
        if not isinstance(value, bool):
            raise ValueError(f"unexpected resolved label {type(value)}")
        return "resolved" if value else "not_resolved"
    by_run, groups = defaultdict(Counter), defaultdict(list)
    for x in replay:
        by_run[str(x.get("run"))][label(x)] += 1
        groups[str(x.get("instance_id"))].append(x)
    return {
        "replay": {
            "rows": len(replay), "fields": sorted(set().union(*(x.keys() for x in replay))),
            "outcomes": dict(Counter(map(label, replay))),
            "by_run": dict(by_run), "distinct_tasks": len(groups),
            "tasks_with_any_success": sum(any(label(x)=="resolved" for x in g) for g in groups.values()),
            "rows_per_task_histogram": dict(Counter(map(len, groups.values()))),
            "same_task_is_not_iid_repeats": "Rows include different models, forks and prompt regimes; do not pool them as repeated independent runs of one fixed policy.",
        },
        "hazard_train": {
            "rows": len(hazard), "outcomes": dict(Counter(map(label, hazard))),
            "fields": sorted(set().union(*(x.keys() for x in hazard))),
            "task_ids_present": any("instance_id" in x or "task_id" in x for x in hazard),
            "split_scope": "train only; no validation/test/hidden labels opened",
            "limitation": "Trajectory IDs do not identify underlying tasks; cannot reconstruct independent task clusters from these fields.",
        },
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "kind": "EXPLORATORY_STATISTICAL_AND_DATA_AVAILABILITY_AUDIT",
        "agent_experiment": False,
        "source_sha256": sha(Path(__file__)),
        "counterexamples": [exact_cluster_case(n, m, .95, .97)
                            for n in [20, 50, 100] for m in [1, 4, 8, 32]],
        "data_audit": audit_data(data_root),
        "source_data_sha256": {p.name: sha(p) for p in sorted(data_root.iterdir()) if p.is_file()},
        "data_revisions": {
            "Anonymousblind/agent-failure-dynamics": "4e448b80d8fd7dec19dacd43312811ee0013f629",
            "ashritha0907/replay-gap-trajectories": "3f3e9f544819afc7fe7faf4a4f5955554e4a15db",
        },
        "decision": "KNOWN_CLUSTERING_FAILURE_CONFIRMED_NO_PAPER_GREENLIGHT",
    }
    (output / "RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    manifest = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data_root, args.output), indent=2))
