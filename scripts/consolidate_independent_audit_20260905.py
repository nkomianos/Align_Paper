"""Build a deduplicated claim index from additive audit ledgers, not raw TEST data."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/independent_audit_20260905"
LABELS = ("Valid positive", "Valid negative", "Invalid assay/capability", "Developmental/apparatus only", "Not run")
ABORTED = {"hindsight_parameter_probe_cpu_v1", "hindsight_revelation_cpu_20260904_v1", "early_smoke_failed", "sdpo_format_failed_launch", "sdpo_single_profile_failed"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads((AUDIT / "data/claim_ledger.json").read_text(encoding="utf-8"))
    portfolio = json.loads((AUDIT / "portfolio/portfolio_ledger.json").read_text(encoding="utf-8"))
    early = json.loads((AUDIT / "portfolio/supplemental_early_ledger.json").read_text(encoding="utf-8"))
    remaining = json.loads((AUDIT / "data/remaining_agent_monitor_ledger.json").read_text(encoding="utf-8"))
    records = []
    for index, original in enumerate(data["rows"], 1):
        row = dict(original)
        row["id"] = f"H{index:03d}"
        row["source_ledger"] = "artifacts/independent_audit_20260905/data/claim_ledger.json"
        row["record_type"] = {
            "empirical_run_or_attempt": "aborted_launch" if row["claim"] in ABORTED else "measured_assay",
            "preparation_or_computation": "preparation_or_computation",
            "never_run_stage": "never_run_protocol_group",
        }[row.pop("unit_type")]
        records.append(row)
    for original in portfolio["records"]:
        row = dict(original)
        row["source_ledger"] = "artifacts/independent_audit_20260905/portfolio/portfolio_ledger.json"
        row["record_type"] = {"executed_assay": "measured_assay", "posthoc_reanalysis": "posthoc_reanalysis", "construction_audit": "preparation_or_computation", "never_run_protocol": "never_run_protocol_group"}[row["record_type"]]
        records.append(row)

    for original in early["records"]:
        row = dict(original)
        row["source_ledger"] = "artifacts/independent_audit_20260905/portfolio/supplemental_early_ledger.json"
        row["record_type"] = {"executed_assay": "measured_assay", "construction_audit": "preparation_or_computation"}[row["record_type"]]
        records.append(row)
    offline_reanalyses = {"remaining_agentuq_inventory", "remaining_agentuq_matched", "remaining_agentuq_calibrated_v2", "remaining_agentuq_prefix", "remaining_agentuq_events", "remaining_scopejudge"}
    for original in remaining["claims"]:
        row = dict(original)
        row["id"] = row.pop("claim_id")
        row["source_ledger"] = "artifacts/independent_audit_20260905/data/remaining_agent_monitor_ledger.json"
        if row["classification"] == "Not run":
            row["record_type"] = "never_run_protocol_group"
        elif row.get("duplicate_of"):
            row["record_type"] = "superseded_duplicate_artifact"
        elif row["id"] in offline_reanalyses:
            row["record_type"] = "posthoc_reanalysis"
        else:
            row["record_type"] = "preparation_or_computation"
        records.append(row)

    theory = {
        "hindsight_factorized_nonidentification_20260904_v1": ("2be2af5", "Valid positive", "formal_claim", "Exact finite law: historical Z0 observed, deployment individual Z0 unavailable for regret witness", "Exact immediate TV0, delayed TV.3 and policy reversal survive. State-visible deployment A=Z0 eliminates this witness regret."),
        "hindsight_minimax_exact_20260904_v1": ("0d8f2c3", "Valid positive", "formal_claim", "Two-action minimax decision problem, corollary of same witness", "Retain2/15 universal expected-risk bound. .2 only for a fixed deterministic action; data-dependent deterministic rule with worst expected regret.16 contradicts original broad wording."),
        "hindsight_multistate_exact_20260904_v1": ("768348f", "Valid positive", "formal_claim", "Exact3-state2-action law with known full-rank measurement channel", "Standard linear identification under known emission, support and state semantics; no new general identification mathematics."),
        "hindsight_longitudinal_mechanism_cpu_20260904_v1": ("17ccaa8", "Valid negative", "finite_model_mechanism_test", "28 configurations,16 seed blocks of1024 simulated users,64 rounds; exact mean equations", "Symmetric adaptation does not necessarily increase baseline drift; exact mean mechanism and paired saved-state replay oppose automatic harm."),
        "hindsight_delayed_anchor_dev_20260904_v1": ("05ffeca", "Developmental/apparatus only", "preparation_or_computation", "150 designed cells per budget,2000 coupled draws per cell", "Clipped augmentation regret.00934317 vs ordinary anchors.01377017 at16 anchors/action survives as synthetic method screen."),
        "hindsight_anchor_robustness_dev_20260904_v1": ("6455e05", "Developmental/apparatus only", "preparation_or_computation", "320 designed cells,1000 coupled draws per cell", "Bias correction survives; clean ordinary anchor mean.00380385 beats augmented-IPW.00417692. Unweighted augmentation.00222308 beats ordinary mean at random selection."),
    }
    for row in records:
        if row["claim"] in theory:
            commit, label, kind, unit, conclusion = theory[row["claim"]]
            row.update(protocol_commit=commit, classification=label, record_type=kind, unit_of_analysis=unit, verifier_result="Committed read-only verifier PASS; independent theory recomputation; see THEORY_RECOMPUTATION.json", error_or_confound="Scope/novelty/measurement limitations independently detailed in AUDIT_THEORY_LITERATURE_20260905.md", corrected_conclusion=conclusion, next_action="Apply corrected theorem/estimator scope; no paper green light")
        if row["claim"] == "hindsight_gradient_power_audit_20260904_v1":
            row["verifier_result"] = "PASS: root reran committed verifier,28 model-free cells; no neural power claim"
        if row["claim"] == "hindsight_neural_policy_power_audit_20260904_v2":
            row["verifier_result"] = "PASS: root reran committed verifier,9/9 ideal surrogate cells and0/9 exact nulls; not neural statistical power"
    extras = [
        ("TEX1", "Bounded-copying partial-identification calculation", "hindsight_partial_identification_20260904_v1.json", "201facd", "Exact designed grid", "External valid copying bound is an assumption; compatibility cannot validate it."),
        ("TEX2", "Anchor confidence-set screen Hoeffding version", "hindsight_anchor_value_20260904_v1.json", "516ed82 current replay source", "108 design cells,2000 draws/cell", "Rows/aggregates replay; oldest artifact lacks subsequently added interval_kind metadata."),
        ("TEX3", "Anchor confidence-set screen CP version", "hindsight_anchor_value_cp_20260904_v1.json", "516ed82", "108 design cells,2000 draws/cell", "Exact replay; pre-treatment objective differs from persistence-target claim."),
        ("TEX4", "Anchor misspecification/cost calculation", "hindsight_anchor_stress_20260904_v1.json", "516ed82", "216 stress/324 hypothetical cost cells", "Exact replay; assumed valid slack and hypothetical costs; anchor-only can win total-cost comparison."),
    ]
    for identifier, claim, filename, commit, unit, conclusion in extras:
        records.append(dict(id=identifier, claim=claim, evidence_root=f"artifacts/{filename}", protocol_commit=commit, unit_of_analysis=unit, record_type="preparation_or_computation", verifier_result="Independent direct deterministic payload replay; see theory/THEORY_RECOMPUTATION.json", classification="Developmental/apparatus only", error_or_confound=conclusion, corrected_conclusion=conclusion, next_action="Retain component check, not paper evidence", source_ledger="docs/AUDIT_THEORY_LITERATURE_20260905.md"))

    required = {"claim", "evidence_root", "protocol_commit", "unit_of_analysis", "verifier_result", "classification", "error_or_confound", "corrected_conclusion", "next_action", "record_type", "id"}
    for row in records:
        if not required <= row.keys() or row["classification"] not in LABELS:
            raise ValueError(f"invalid ledger row: {row.get('id')}")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate record ID")
    counts = defaultdict(Counter)
    for row in records:
        counts[row["record_type"]][row["classification"]] += 1
    counts = {kind: {label: count.get(label,0) for label in LABELS} for kind,count in counts.items()}
    sources = [AUDIT / "data/claim_ledger.json", AUDIT / "portfolio/portfolio_ledger.json", AUDIT / "portfolio/supplemental_early_ledger.json", AUDIT / "portfolio/excluded_attempts.json", AUDIT / "data/remaining_agent_monitor_ledger.json", AUDIT / "theory/THEORY_RECOMPUTATION.json", AUDIT / "inventory.json", AUDIT / "inventory_receipt.json", AUDIT / "runtime_receipts.json", AUDIT / "neural/design_recomputation.json"]
    report = {
        "as_of": "2026-09-05 UTC / 2026-09-04 Pacific",
        "audited_starting_commit": "1cfe22124955dee56c8683cbf0ea0f51137e146a",
        "kind": "independent claim-level audit with explicit coverage/provenance limits, not universal neural reproduction",
        "count_convention": "One logical measured assay version; five explicitly identified Hindsight pre-inference attempts are separate aborted launches. Additional historical partial/asset failures and duplicates are in the ancillary attempt register, not silently counted as pre-inference. Copied roots, models/cells, seeds, rotations, panels and checkpoints do not increase study count. Formal claims are separate and partly dependent. Never-run entries are16 grouped known protocol stages, not every unexecuted script. Six supplemental released-data reanalyses and one superseded duplicate are distinct from measured model assays. New proposed6-arm study is excluded from historical counts.",
        "counts": counts,
        "records": records,
        "source_fingerprints": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "ancillary_attempt_register": "artifacts/independent_audit_20260905/portfolio/excluded_attempts.json",
        "coverage_limits": portfolio["coverage_limits"] + early["coverage_limits"] + ["The main portfolio's earlier structural-only caveat is superseded for AgentUQ, ScopeJudge, BeyondMasks, cluster certification, byte/native qualification and pre-bridge oracle roots by the supplemental ledgers and reports. CPU SQuAD P29 now has a dedicated passing full replay. Generic source/download/preview fixtures are still not scientific endpoints.", "Supplemental exact finite coupling E03 occurs once in this index; it is not duplicated among the Hindsight formal claims.", "Supplemental offline reanalyses recompute raw rewards/statistics and source bindings but do not independently refit all logistic coefficients or rebuild terminal NLL from absent sidecars; retired-candidate literature was not universally refreshed.", "Some PAHF/PUPPET verifiers parse reserved content; these were not run. Hash checks and unlocked data analysis are not full confirmation reconstruction.", "The numeric PUPPET verifier passes invalid targets; original source-to-repaired-DEV target binding remains partially unaudited to avoid reading mixed-file reserved outcomes.", "Future source/model/dependency/runtime provenance and verifier hardening are needed before execution. No current GPU host status or complete historical billable-hour total can be inferred."],
        "supporting_reports": ["docs/ICLR_2027_INDEPENDENT_AUDIT_REPORT_20260905.md", "docs/AUDIT_THEORY_LITERATURE_20260905.md", "docs/AUDIT_NEURAL_QUEUE_20260905.md", "docs/AUDIT_HINDSIGHT_DATA_20260905.md", "docs/AUDIT_PORTFOLIO_20260905.md", "docs/AUDIT_PORTFOLIO_EARLY_SUPPLEMENT_20260905.md", "docs/AUDIT_OFFLINE_MONITORS_SUPPLEMENT_20260905.md", "docs/ICLR_2027_PAPER_COMPLETION_PLAN_20260905.md"],
    }
    output = ROOT / "docs/RESEARCH_EVIDENCE_INDEX_20260905.json"
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"records": len(records), "counts": counts},indent=2))


if __name__ == "__main__":
    main()
