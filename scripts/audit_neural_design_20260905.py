"""Deterministic code/data audit, without model loading or training."""
from collections import Counter
from pathlib import Path
import hashlib
import itertools
import json
import subprocess

from interaction_sprint.hindsight_neural_anchor import build_records, SEED
from interaction_sprint.hindsight_neural_gradient_v2 import build_nested_anchor_panels
from interaction_sprint.hindsight_neural_policy_g1 import (
    build_disjoint_policy_panels, build_policy_schedules, POLICY_SEED,
    summarize_evaluation_rows,
)
from interaction_sprint.hindsight_neural_policy_power import target_means

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/independent_audit_20260905/neural"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, evaluation, _ = build_records()
    nested = build_nested_anchor_panels(rows)
    policy = build_disjoint_policy_panels(rows)
    schedules = build_policy_schedules(rows, policy)
    # A mathematical counterexample to aggregate label-position cancellation.
    cancellation = [
        {"id": f"{domain}-{swap}", "swap": swap,
         "semantic_probabilities": [1. - value, value], "ab_mass": 1.}
        for domain, probs in (("d0", (1., 0.)), ("d1", (0., 1.)))
        for swap, value in enumerate(probs)
    ]
    tracked = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    report = {
        "kind": "deterministic audit of existing generated data and decision functions; no neural experiment",
        "commit": tracked,
        "rows": len(rows), "distinct_training_prompts": len({r["prompt"] for r in rows}),
        "training_domains": len({r["domain"] for r in rows}), "evaluation_rows": len(evaluation),
        "evaluation_domains": len({r["domain"] for r in evaluation}),
        "evaluation_domains_seen_in_training": len({r["domain"] for r in evaluation} & {r["domain"] for r in rows}),
        "exact_prompt_overlap": len({r["prompt"] for r in evaluation} & {r["prompt"] for r in rows}),
        "overlap_after_removing_study_prefix": len({r["prompt"].split("Choose the response format", 1)[1] for r in evaluation} & {r["prompt"].split("Choose the response format", 1)[1] for r in rows}),
        "raw_label_mean": sum(r["immediate_semantic"] for r in rows)/len(rows),
        "delayed_expression_label_mean": sum(r["delayed_expression_semantic"] for r in rows)/len(rows),
        "action1_residual_nonzero": sum(r["immediate_feedback"] != r["delayed_expression_feedback"] for r in rows if r["logged_action"] == 1),
        "gradient_panel_overlap_counts": {str(b): [len(set(a) & set(c)) for a,c in itertools.combinations(ps,2)] for b,ps in nested.items()},
        "gradient_panels_unique_rows": {str(b): len(set(itertools.chain.from_iterable(ps))) for b,ps in nested.items()},
        "policy_panels_unique_rows": len(set(itertools.chain.from_iterable(policy))),
        "global_schedule_row_exposures": dict(Counter(Counter(itertools.chain.from_iterable(schedules["global"])).values())),
        "g0_adapter_seed": SEED, "g1_adapter_seed": POLICY_SEED,
        "policy_label_targets": target_means(rows,policy),
        "position_cancellation_counterexample": {"rows": cancellation, "saved_aggregate_metric": summarize_evaluation_rows(cancellation), "paired_absolute_position_gap": 1.},
        "utility_counterexample": {
            "declared_expression_P_z1": .75,
            "oracle_endpoint_p_action1": .8,
            "candidate_p_action1": .99,
            "oracle_correctness": .25 + .5*.8,
            "candidate_correctness": .25 + .5*.99,
            "candidate_distance_from_trained_oracle": .19,
            "interpretation": "Oracle-endpoint distance is imitation fidelity, not monotone in persistent-state correctness."
        },
    }
    (OUT / "design_recomputation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
