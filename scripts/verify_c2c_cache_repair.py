"""Read-only repair verification, including independent closed-form map replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from scripts.run_c2c_cache_repair import SPEC, SOURCE_FILES, validate_parent
from scripts.run_c2c_sender_update import SPEC as UPDATE_SPEC, sha256
from scripts.verify_c2c_paired_update import verify as verify_paired, combine as combine_paired
from latent_contract.cache_calibration import calibration_split
from latent_contract.cache_repair import sample_positions
from latent_contract.repair_analysis import analyze


def verify_maps(root):
    import torch
    from latent_contract.cache_repair import verify_head_map, METHODS
    cases = json.loads((root / "calibration.json").read_text())
    fit_ids = calibration_split(cases, SPEC["fit_cases_per_dataset"])
    bank = json.loads((root / "cache_bank/bank.json").read_text())
    if bank["mapping"] != [[i, i+8] for i in range(28)] or bank["fit_cases"] != 96 or bank["check_cases"] != 32:
        raise ValueError("calibration mapping/split differs")
    if bank["generation_calls"] != 0 or bank["backbone_forwards"] != 384 or bank["maximum_positions_per_case"] != 32:
        raise ValueError("calibration execution metadata differs")
    if [r["case_id"] for r in bank["cases"]] != [c["case_id"] for c in cases]:
        raise ValueError("calibration case grid differs")
    counts = {"fit": 0, "check": 0}
    for row in bank["cases"]:
        split = "fit" if row["case_id"] in fit_ids else "check"
        positions = sample_positions(len(row["prompt_ids"])-1, 32)
        if row["split"] != split or row["selected_prefix_positions"] != positions:
            raise ValueError("calibration sampling differs")
        counts[split] += len(positions)
    fitting_costs = dict.fromkeys([*METHODS, "retuned"], 0.)
    for index in range(28):
        data = torch.load(root / "cache_bank" / f"layer_{index:02d}.pt", weights_only=True, map_location="cpu")
        layer = root / "repairs" / f"layer_{index:02d}"
        for split in ("fit", "check"):
            expected_keys = {r+"_"+k for r in ("old", "new", "receiver") for k in ("key", "value")}
            if set(data[split]) != expected_keys:
                raise ValueError("cache model/key coverage differs")
            for tensor in data[split].values():
                if tensor.shape != (1, 8, counts[split], 128) or not torch.isfinite(tensor).all():
                    raise ValueError("cache shape/value differs")
        for method in METHODS:
            actual = torch.load(layer / (method+".pt"), weights_only=True, map_location="cpu")
            if set(actual) != {"key", "value"}:
                raise ValueError("repair map key coverage differs")
            for kind in ("key", "value"):
                verify_head_map(data["fit"]["new_"+kind], data["fit"]["old_"+kind], actual[kind], method, ridge=SPEC["ridge"])
        report = json.loads((layer / "fit.json").read_text())
        if [row["step"] for row in report["curve"]] != list(range(1, 101)):
            raise ValueError("retune steps differ")
        for row in report["curve"]:
            if len(row["sample_indices"]) != 64 or any(i < 0 or i >= counts["fit"] for i in row["sample_indices"]):
                raise ValueError("retune sample indices differ")
            if any(not math.isfinite(row[k]) for k in ("normalized_output_mse", "gradient_norm")):
                raise ValueError("nonfinite retune telemetry")
        for filename in ("retuned_float32.pt", "retuned_bfloat16.pt"):
            if not (layer / filename).is_file():
                raise ValueError("missing retuned checkpoint")
        for method, elapsed in report["fit_check_save_seconds"].items():
            if method not in fitting_costs or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid fitting time")
            fitting_costs[method] += elapsed
    return {"closed_form_map_replay": "passed", "retune_optimizer_replayed": False,
            "fit_check_save_seconds": fitting_costs, "fit_tokens": counts["fit"], "check_tokens": counts["check"]}


def verify(root, key, ticket, paired_report, expected_commit, parent_rows):
    files = json.loads((root / "MANIFEST.json").read_text())["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "MANIFEST.json"}
    if actual != set(files):
        raise ValueError("repair manifest coverage differs")
    for name, digest in files.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or sha256(target) != digest:
            raise ValueError("repair manifest digest/path differs")
    if (root / "FAILED.json").exists():
        raise ValueError("repair run failed")
    spec = json.loads((root / "spec.json").read_text())
    seed = spec.pop("seed")
    if spec != SPEC or seed not in UPDATE_SPEC["seeds"]:
        raise ValueError("repair spec differs")
    if json.loads((root / "COMPLETE.json").read_text()) != {"calls": 1280, "seed": seed, "calibration_cases": 128, "retuned_layers": 28}:
        raise ValueError("repair incomplete")
    if (root / "release_ticket.json").read_bytes() != ticket.read_bytes() or (root / "paired_report.json").read_bytes() != paired_report.read_bytes():
        raise ValueError("repair release differs")
    parent = validate_parent(paired_report, sha256(paired_report), sha256(ticket))
    if sha256(root / "cases.json") != SPEC["cases_sha256"] or sha256(root / "calibration.json") != SPEC["calibration_sha256"]:
        raise ValueError("repair data differs")
    runtime = json.loads((root / "runtime.json").read_text())
    if any(runtime[k] != SPEC[k] for k in ("torch", "transformers")):
        raise ValueError("repair dependencies differ")
    if runtime["models"]["old"] != SPEC["sender_revision"] or runtime["models"]["receiver"] != SPEC["receiver_revision"]:
        raise ValueError("repair base revisions differ")
    ticket_data = json.loads(ticket.read_text())
    if runtime["updated_model_files"] != ticket_data["updates"][str(seed)]["merged_files"] or runtime["ticket_sha256"] != sha256(ticket) or runtime["paired_report_sha256"] != sha256(paired_report):
        raise ValueError("repair model/release attestation differs")
    if runtime["strict_projectors_loaded"] != 28 or runtime["mapping"] != {"0": {"1": {str(i): [[i+8, i]] for i in range(28)}}}:
        raise ValueError("repair projector mapping differs")
    sources = json.loads((root / "source.json").read_text())
    if sources["commit"] != expected_commit or set(sources["files"]) != set(SOURCE_FILES):
        raise ValueError("repair source commit differs")
    repo = Path(__file__).resolve().parents[1]
    for relative in SOURCE_FILES:
        expected = subprocess.check_output(["git", "-C", str(repo), "show", expected_commit+":"+relative])
        if (root / "source" / relative).read_bytes() != expected or hashlib.sha256(expected).hexdigest() != sources["files"][relative]:
            raise ValueError("repair source bytes differ")
    if sha256(key) != "1aec66fe9cdaff557a6d834be98ad3fadde195ba10248a8ce3637b62b55c7c41":
        raise ValueError("repair answer key differs")
    cases = json.loads((root / "cases.json").read_text())
    answer_key = json.loads(key.read_text())["final_eval"]
    rows = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    result = analyze(cases, answer_key, rows, parent_rows[seed])
    result["calibration_verification"] = verify_maps(root)
    cost = json.loads((root / "repair_cost.json").read_text())
    if cost["backbone_forwards"] != 384 or cost["requires_original_sender_for_calibration"] is not True:
        raise ValueError("repair collection costs omitted")
    if any(not math.isfinite(cost[k]) or cost[k] < 0 for k in ("cache_collection_seconds", "fitting_and_loading_seconds")):
        raise ValueError("invalid preparation timing")
    result.update({"seed": seed, "scope": SPEC["scope"], "preparation_cost": cost})
    old_report = next(r for r in parent["per_seed"] if r["seed"] == seed)
    result["paired_reference_costs"] = {k: old_report["metrics"][k] for k in ("new_sender", "new_text", "new_c2c")}
    saving = (old_report["metrics"]["new_text"]["serial_generation_seconds"]-result["metrics"]["ridge"]["generation_seconds"])/len(cases)
    prep = cost["cache_collection_seconds"]+cost["fitting_and_loading_seconds"]
    result["conservative_all_baselines_break_even_questions_vs_text"] = prep/saving if saving > 0 else None
    result["cost_caveat"] = "Includes all baseline fitting for a conservative amortization bound; serial separate-run timings are not an optimized deployment benchmark. Original-sender storage and sender-alone alternatives remain relevant."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, nargs=2, required=True)
    parser.add_argument("--paired-root", type=Path, nargs=2, required=True)
    for name in ("key", "ticket", "paired-report", "upstream", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("expected-commit", "paired-commit"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in [*args.root, *args.paired_root, args.upstream]):
        parser.error("report must be outside evidence")
    parents = [verify_paired(root, args.key, args.ticket, args.paired_commit, args.upstream) for root in args.paired_root]
    rebuilt = combine_paired([p[0] for p in parents], [p[1] for p in parents])
    if rebuilt != json.loads(args.paired_report.read_text()):
        raise ValueError("paired report does not reproduce from its verified evidence")
    parent_rows = {report["seed"]: rows for report, rows in parents}
    results = [verify(root, args.key, args.ticket, args.paired_report, args.expected_commit, parent_rows) for root in args.root]
    if len({r["seed"] for r in results}) != 2:
        raise ValueError("duplicate repair seed")
    repeated = all(r["decision"] == "EXISTING_RIDGE_BASELINE_RECOVERS_DAMAGE_NOT_NEW_METHOD" for r in results)
    report = {"decision": "EXISTING_REPAIR_RECOVERY_REPEATED_NOVELTY_UNRESOLVED" if repeated else "PRIMARY_REPAIR_NOT_ESTABLISHED_ACROSS_BOTH_UPDATES",
              "per_seed": results, "paper_green_light": False,
              "interpretation": "No new method or independent task/model-family replication has been established; a PI novelty/utility review remains necessary."}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report))
