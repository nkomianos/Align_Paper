"""Read-only evidence and arithmetic replay; does not rerun model forwards."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from interaction_sprint.parameter_probe import SPEC, cases

SOURCE_SHA = "24b34c8f3276403e5e5a255873df470f2da16080e5f2bb88527b65020fb520f5"


def metrics(rows, split):
    selected = [r for r in rows if r["split"] == split]
    return {"n": len(selected),
            "accuracy": float(np.mean([np.argmax(r["probabilities"]) == r["answer"] for r in selected])),
            "original_request_NLL": float(np.mean([-np.log(r["probabilities"][r["answer"]]) for r in selected])),
            "minimum_AB_mass": min(r["AB_mass"] for r in selected)}


def validate_rows(rows):
    wanted = {c["id"]: c for c in cases()}
    if len(rows) != len(wanted) or {r["id"] for r in rows} != set(wanted):
        raise ValueError("missing/duplicate cases")
    for r in rows:
        p = np.asarray(r["probabilities"])
        expected = wanted[r["id"]]
        if r["answer"] != expected["answer"] or r["split"] != expected["split"]:
            raise ValueError("label/split mismatch")
        if p.shape != (2,) or not np.isfinite(p).all() or not (p > 0).all() or abs(p.sum()-1) > 1e-6:
            raise ValueError("invalid probabilities")
        if not 0 <= r["AB_mass"] <= 1.00001:
            raise ValueError("invalid vocabulary mass")


def verify(root):
    manifest = json.loads((root/"MANIFEST.json").read_text())
    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != "MANIFEST.json"}
    if actual != set(manifest):
        raise ValueError("manifest coverage mismatch")
    for name, digest in manifest.items():
        if Path(name).name != name or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("checksum mismatch")
    if hashlib.sha256((root/"probe_source.py").read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("unreviewed runner")
    load = lambda name: json.loads((root/name).read_text())
    if load("spec.json") != SPEC or load("cases.json") != cases():
        raise ValueError("specification/input mismatch")
    if (root/"FAILED.json").exists() or load("COMPLETE.json") != {"forwards": 66, "backwards": 8}:
        raise ValueError("incomplete probe")
    runtime = load("runtime.json")
    if runtime["torch"] != "2.11.0+cpu" or runtime["transformers"] != "5.6.2" or not runtime["zero_adapter_exact"]:
        raise ValueError("unexpected runtime/control")
    result = load("RESULT.json")
    if result["counts"] != load("COMPLETE.json") or result["paper_green_light"] is not False:
        raise ValueError("inconsistent result metadata")
    rows_by_arm = {"baseline": load("baseline.json"), **{name: load(f"{name}_one_step.json")["rows"] for name in ("own", "full")}}
    recomputed = {}
    for name, rows in rows_by_arm.items():
        validate_rows(rows)
        recomputed[name] = {}
        for split in ("train", "heldout"):
            recomputed[name][split] = metrics(rows, split)
            if any(not np.isclose(v, result["metrics"][name][split][k], atol=1e-8) for k, v in recomputed[name][split].items()):
                raise ValueError("reported metric mismatch")
    directions = torch.load(root/"directions.pt", weights_only=True, map_location="cpu")
    geometry = []
    for rho in SPEC["copy_strengths"]:
        a, b = [directions[f"rho{rho}_{name}"].double() for name in ("own", "full")]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("invalid directions")
        cos = float(torch.dot(a, b)/(a.norm()*b.norm()))
        geometry.append({"rho": rho, "cosine_float64": cos, "own_norm_float64": float(a.norm()),
                         "full_norm_float64": float(b.norm()), "difference_norm": float((a-b).norm())})
    if geometry[0]["difference_norm"] > 1e-6:
        raise ValueError("independent feedback null failed")
    initial = torch.load(root/"initial_adapter.pt", weights_only=True, map_location="cpu")
    steps = {}
    for name in ("own", "full"):
        updated = torch.load(root/f"{name}_one_step_adapter.pt", weights_only=True, map_location="cpu")
        if list(initial) != list(updated):
            raise ValueError("adapter ordering/coverage mismatch")
        delta = torch.cat([(updated[k]-initial[k]).reshape(-1) for k in initial]).double()
        direction = directions[f"rho{SPEC['one_step_copy_strength']}_{name}"].double()
        expected = direction/direction.norm()*SPEC["one_step_parameter_L2"]
        if delta.shape != expected.shape or not torch.allclose(delta, expected, atol=1e-7, rtol=1e-4):
            raise ValueError("stored update does not match declared direction")
        steps[name] = {"actual_delta_norm": float(delta.norm()), "direction_error_norm": float((delta-expected).norm())}
    return {"scope": "VERIFIED_ONE_STEP_DIAGNOSTIC_NO_MODEL_REPLAY", "paper_green_light": False,
            "source_sha256": SOURCE_SHA, "manifest_sha256": hashlib.sha256((root/"MANIFEST.json").read_bytes()).hexdigest(),
            "metrics": recomputed, "geometry_float64": geometry, "step_checks": steps,
            "limitations": ["Gradients and model probabilities are not independently re-executed.",
                            "Recomputed float64 cosines supersede imprecise float32 diagnostic values only; original evidence unchanged.",
                            "Both updates are normalized to equal parameter distance; this does not compare raw step magnitudes.",
                            "Tiny handwritten split, 50% training baseline accuracy; not a robust utility benchmark.",
                            "Teacher A/B probabilities are saved but teacher full-vocabulary mass was not retained.",
                            "No conclusion about full SDPO training, changing human preferences, or acceptance viability."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("output must be outside immutable evidence")
    result = verify(args.root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))
