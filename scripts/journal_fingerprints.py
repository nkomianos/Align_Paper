"""Fingerprint only the allowlisted aggregate reports behind the journal.

Does not open original locked TEST, raw trajectories, private keys or credentials.
This is NOT a rerun of experiment verifiers: it identifies the report bytes read
when consolidating the journal. Requires the owner's ignored local archives.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPORTS = {
    "UE-1": "results/stage1_20260819/stage1_dev_report.json",
    "UE-DID": "retrieved/did_v1.1.1_20260825/analysis/did_v1.1.1_report.json",
    "PA-0": "retrieved/provenance_authority_g0_20260825/analysis/report_local.json",
    "RI-0": "retrieved/interface_invariance_g0_20260825/analysis/report_local.json",
    "HM-0": "retrieved/hybrid_memory_g0_v1_1_20260825/report_local.json",
    "REC-0": "retrieved/recency_g0_20260828T0432Z/run/gate_report.json",
    "J-0": "retrieved/recipe_invariant_j0_20260828T0710Z_verified_20260828T0815Z/run/gate_report.json",
    "SA-DEV": "retrieved/semantic_ancestry_rag_g0_20260828T0901Z_qwen_family_snapshot_20260828T142825Z/family_qwen3_5/gate_report.json",
    "SA-0b": "retrieved/semantic_ancestry_rag_g0b_20260829T0128Z_complete_20260829T1054Z/recovered_aggregate_20260829T1054Z/gate_report.json",
    "EC-0": "retrieved/effect_consistency_uq_g0_20260829T2225Z_retrieved_20260830T0134Z/effect_consistency_uq_g0_20260829T2225Z/GATE_REPORT.json",
    "VP-0": "retrieved/visual_patch_phase_g0_gemma4_20260829T2300Z_retrieved_20260830T0210Z/visual_patch_phase_g0_gemma4_20260829T2300Z/GATE_REPORT.json",
    "FL-0": "retrieved/feedback_leakage_g0_20260830T212423Z_retrieved_20260830T225635Z/verification_report.json",
    "EW-DEV": "artifacts/reward_hack_early_warning_dev_20260830/REPORT.json",
    "PR-0": "retrieved/phantom_rollback_g0_20260901T0930Z/VERIFIED_REPORT.json",
    "RED-0": "retrieved/reward_extinction_debt_g0_20260901T1100Z_VERIFIED_REPORT.json/VERIFIED_GATE_REPORT.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    records = []
    for experiment, relative in REPORTS.items():
        raw = (repo / relative).read_bytes()
        report = json.loads(raw)
        records.append({"id": experiment, "source": relative,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "reported_status": {k: report[k] for k in ("decision", "status", "pass", "passed", "pass_gate", "all_gates_pass") if k in report}})
    result = {"as_of": "2026-09-02", "kind": "aggregate_report_fingerprints_not_reverification",
              "records": records}
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Fingerprinted {len(records)} allowlisted aggregate reports")


if __name__ == "__main__":
    main()
