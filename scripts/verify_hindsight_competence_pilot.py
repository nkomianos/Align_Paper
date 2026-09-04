"""Read-only manifest and outcome/coefficient replay; does not rerun training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from interaction_sprint.competence_pilot import SPEC, dataset, schedule, qualifies, summarize
from interaction_sprint.factorial_probe import feedback_channel
from interaction_sprint.feedback_gradient_audit import gradients
from scripts.verify_hindsight_factorial_probe import same

SOURCE_SHA = "16b2a92c88b8871ccd9d6b25a0c5efdf03f32eb800c20fe3864b4d9f5add0f4d"


def validate(rows, wanted):
    expected = {r["id"]: r for r in wanted}
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("case coverage mismatch")
    for r in rows:
        if any(r[k] != expected[r["id"]][k] for k in ("answer", "domain")):
            raise ValueError("case label/domain mismatch")
        p = np.asarray(r["probabilities"])
        if p.shape != (2,) or not np.isfinite(p).all() or not (p > 0).all() or abs(p.sum()-1) > 1e-8:
            raise ValueError("invalid probabilities")
        if not np.isfinite(r["AB_mass"]) or not 0 <= r["AB_mass"] <= 1.00001:
            raise ValueError("invalid vocabulary mass")


def verify(root):
    torch.set_num_threads(4)
    load = lambda name: json.loads((root/name).read_text())
    manifest = load("MANIFEST.json")
    if {p.name for p in root.iterdir()} != set(manifest) | {"MANIFEST.json"}:
        raise ValueError("manifest coverage mismatch")
    for name, digest in manifest.items():
        if Path(name).name != name or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("checksum mismatch")
    if manifest.get("runner_source.py") != SOURCE_SHA or load("spec.json") != SPEC or load("cases.json") != dataset():
        raise ValueError("frozen source or specification mismatch")
    for path, digest in load("sources.json").items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("analysis dependency changed")
    data = dataset()
    splits = {s: [r for r in data if r["split"] == s] for s in ("warm", "qualify", "adapt", "eval")}
    expected_schedule = dict(warm=[[r["id"] for r in b] for b in schedule(splits["warm"], 16)],
          adapt=[[r["id"] for r in b] for b in schedule(splits["adapt"], 8)], anchor_positions=[0, 1])
    if load("schedule.json") != expected_schedule:
        raise ValueError("batch/anchor schedule mismatch")
    runtime = load("runtime.json")
    if not runtime["zero_adapter_exact"] or runtime["threads"] != 4 or runtime["torch"] != "2.11.0+cpu":
        raise ValueError("runtime/control mismatch")
    for name in ("base_qualification.json", "qualification.json"):
        validate(load(name), splits["qualify"])
    report = load("RESULT.json")
    qualified = qualifies(load("qualification.json"))
    if not same(summarize(load("qualification.json")), report["qualification"]) or report["paper_green_light"]:
        raise ValueError("qualification/report mismatch")
    expected_counts = (dict(forwards=1250, backwards=544, warm_updates=16, feedback_updates=64) if qualified else
                       dict(forwards=194, backwards=128, warm_updates=16, feedback_updates=0))
    if report["counts"] != expected_counts or load("COMPLETE.json") != expected_counts:
        raise ValueError("compute mismatch")
    if report["decision"] != ("DEVELOPMENT_LEARNING_COMPLETE_REQUIRES_ANALYSIS" if qualified else "INVALID_COMPETENCE_NO_FEEDBACK_TRAINING"):
        raise ValueError("decision mismatch")
    warm = load("warm_steps.json")
    if [r["step"] for r in warm] != list(range(1, 17)) or any(not np.isfinite(r["loss"]) or r["loss"] < 0 for r in warm):
        raise ValueError("warmup log invalid")
    initial = torch.load(root/"initial_adapter.pt", weights_only=True, map_location="cpu")
    checkpoint_deltas = {}
    for path in root.glob("*_adapter.pt"):
        state = torch.load(path, weights_only=True, map_location="cpu")
        if state.keys() != initial.keys() or any(v.shape != initial[k].shape or not torch.isfinite(v).all() for k, v in state.items()):
            raise ValueError("invalid adapter")
        checkpoint_deltas[path.name] = sum(float((v.double()-initial[k].double()).square().sum()) for k, v in state.items())**.5
    if checkpoint_deltas["competent_adapter.pt"] <= 0:
        raise ValueError("warmup did not change adapter")
    result = dict(scope="VERIFIED_CAPABILITY_CONTROLLED_PILOT_NO_MODEL_REPLAY", paper_green_light=False,
        decision=report["decision"], counts=expected_counts, elapsed_seconds=report["elapsed_seconds"],
        manifest_sha256=hashlib.sha256((root/"MANIFEST.json").read_bytes()).hexdigest(),
        base_qualification=summarize(load("base_qualification.json")), qualification=report["qualification"],
        checkpoint_delta_from_initial=checkpoint_deltas,
        limitations=["No independent model-forward, backward or optimizer replay.",
                     "Synthetic four-domain benchmark, one initialization, fixed feedback teacher and eight adaptation updates.",
                     "Known simulator channel integrated exactly, not an estimated feedback channel.",
                     "Anchor projection is an existing first-order baseline, not a novel method."])
    if not qualified:
        if any("_eval.json" in name for name in manifest):
            raise ValueError("evaluation opened after qualification failure")
        return result
    baseline = load("competent_eval.json"); validate(baseline, splits["eval"])
    metrics = {"competent": summarize(baseline)}
    teachers = load("teachers.json")
    if set(teachers) != {r["id"] for r in splits["adapt"]}:
        raise ValueError("teacher coverage mismatch")
    for c in splits["adapt"]:
        if len(teachers[c["id"]]) != 2:
            raise ValueError("teacher observations missing")
        for row in teachers[c["id"]]:
            validate([row], [c])
    byid = {r["id"]: r for r in data}
    for rho in SPEC["copy_strengths"]:
        for method in SPEC["methods"]:
            arm = f"rho{rho}_{method}"
            rows = load(f"{arm}_eval.json"); validate(rows, splits["eval"])
            metrics[arm] = summarize(rows)
            logs = load(f"{arm}_steps.json")
            if len(logs) != 8 or [r["step"] for r in logs] != list(range(1, 9)):
                raise ValueError("adaptation step coverage mismatch")
            for step, log in enumerate(logs):
                batch = expected_schedule["adapt"][step]
                exposed = batch[:2] if method == "anchor_only" else batch
                validate(log["rows"], [byid[k] for k in exposed])
                if not np.isfinite(log["applied_gradient_norm"]) or log["applied_gradient_norm"] < 0:
                    raise ValueError("invalid applied gradient norm")
                expected_scale = .01*min(1., 1./max(log["applied_gradient_norm"], 1e-30))
                if not np.isclose(expected_scale, log["scale"]):
                    raise ValueError("step scaling mismatch")
                if method == "projected_full" and log["anchor_dot_after"] < -1e-5:
                    raise ValueError("projection constraint violated")
                for r in log["rows"]:
                    if r["anchor_available"] != (r["id"] in batch[:2]):
                        raise ValueError("anchor allocation mismatch")
                    if method != "anchor_only":
                        c = byid[r["id"]]
                        q = np.asarray([x["probabilities"] for x in teachers[r["id"]]])
                        g = gradients(r["probabilities"][1], feedback_channel(c["answer"], rho, "truth90"), q)
                        key = "sampled_response_ascent" if method == "own" else "full_reverse_kl_ascent"
                        if not np.isclose(g[key], r["ascent_coefficient"], atol=1e-9, rtol=1e-7):
                            raise ValueError("coefficient replay mismatch")
    if not same(metrics, report["metrics"]):
        raise ValueError("metric replay mismatch")
    # With no action copying, the exact own/full updates should agree, not just final accuracy.
    null_own = torch.load(root/"rho0.0_own_adapter.pt", weights_only=True, map_location="cpu")
    null_full = torch.load(root/"rho0.0_full_adapter.pt", weights_only=True, map_location="cpu")
    null_delta = sum(float((v.double()-null_full[k].double()).square().sum()) for k, v in null_own.items())**.5
    if null_delta > 1e-5:
        raise ValueError("independent-feedback null trajectories differ")
    anchors = [k for b in expected_schedule["adapt"] for k in b[:2]]
    result.update(metrics=metrics, anchor_exposures=len(anchors), unique_anchor_contexts=len(set(anchors)),
                  independent_feedback_adapter_difference_L2=null_delta,
                  min_teacher_AB_mass=min(x["AB_mass"] for rows in teachers.values() for x in rows),
                  copying_minus_independent={m: {
                      "accuracy_pp": 100*(metrics[f"rho0.9_{m}"]["accuracy"]-metrics[f"rho0.0_{m}"]["accuracy"]),
                      "nll": metrics[f"rho0.9_{m}"]["nll"]-metrics[f"rho0.0_{m}"]["nll"]} for m in SPEC["methods"]})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))
