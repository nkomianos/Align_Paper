"""Read-only checksum, tokenization and arithmetic replay; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from interaction_sprint.factorial_probe import SPEC, analyze, design, teacher_prompt
from scripts.verify_hindsight_parameter_probe import verify as verify_parent

SOURCE_SHA = "0aef884ffec06c3bd0a514dccbf0762b9b612d4fc888fbe2c62d5736067f6a23"


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return np.isclose(a, b, atol=1e-8, rtol=1e-7)
    return a == b


def verify(root, parent):
    torch.set_num_threads(4)
    load = lambda name: json.loads((root/name).read_text())
    manifest = load("MANIFEST.json")
    if {p.name for p in root.iterdir()} != set(manifest) | {"MANIFEST.json"}:
        raise ValueError("manifest coverage mismatch")
    for name, digest in manifest.items():
        if Path(name).name != name or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("checksum mismatch")
    if manifest.get("runner_source.py") != SOURCE_SHA:
        raise ValueError("unreviewed runner")
    if load("spec.json") != SPEC or load("cases.json") != design():
        raise ValueError("frozen design mismatch")
    if load("COMPLETE.json") != {"forwards": 82, "backwards": 16, "updates": 0}:
        raise ValueError("incomplete or unexpected compute")
    parent_report = verify_parent(parent)
    if load("parent.json")["manifest_sha256"] != parent_report["manifest_sha256"]:
        raise ValueError("parent mismatch")
    for name, digest in load("source_digests.json").items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("analysis dependency changed")
    records = load("records.json")
    expected = {c["id"]: c for c in design()}
    masses = {"base": [], "released": [], "plain": []}
    for r in records:
        c = expected[r["id"]]
        if any(r[k] != c[k] for k in ("source_id", "intent", "task", "answer")):
            raise ValueError("metadata mismatch")
        if set(r["teachers"]) != {"released", "plain"}:
            raise ValueError("teacher grid mismatch")
        groups = {"base": [r["base"]], **r["teachers"]}
        for key, rows in groups.items():
            if len(rows) != (1 if key == "base" else 2):
                raise ValueError("teacher observation coverage mismatch")
            for row in rows:
                p = np.asarray(row["probabilities"])
                mass = row["AB_mass"]
                if p.shape != (2,) or not np.isfinite(p).all() or not (p > 0).all() or abs(p.sum()-1) > 1e-6:
                    raise ValueError("invalid probabilities")
                if not np.isfinite(mass) or not 0 <= mass <= 1.00001:
                    raise ValueError("invalid vocabulary mass")
                masses[key].append(mass)
    initial = torch.load(root/"initial_adapter.pt", weights_only=True, map_location="cpu")
    old = torch.load(parent/"initial_adapter.pt", weights_only=True, map_location="cpu")
    if list(initial) != list(old) or any(not torch.equal(initial[k], old[k]) for k in old):
        raise ValueError("initial adapters differ")
    nparams = sum(v.numel() for v in initial.values())
    j = torch.load(root/"jacobians.pt", weights_only=True, map_location="cpu")
    if any(v.shape != (nparams,) or not torch.isfinite(v).all() for v in j.values()):
        raise ValueError("invalid Jacobian")
    prompts = [design()[0]["prompt"]]*2
    for c in design():
        prompts.append(c["prompt"])
        prompts.extend(teacher_prompt(c, t, o) for t in SPEC["template"] for o in (0, 1))
    saved = load("prompts.json")
    if [p["text"] for p in saved] != prompts:
        raise ValueError("prompt sequence mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SPEC["model"], revision=SPEC["revision"], local_files_only=True)
    for row in saved:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": row["text"]}], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False)
        if ids != row["token_ids"] or len(ids) > SPEC["max_tokens"]:
            raise ValueError("tokenization mismatch")
    replay = analyze(records, j)
    result = load("RESULT.json")
    if not same(replay, {k: result[k] for k in replay}) or not result["parent_primary_reproduced"]:
        raise ValueError("arithmetic replay mismatch")
    earlier = {r["id"]: r for r in json.loads((parent/"baseline.json").read_text())}
    for r in records:
        if r["intent"] == "stated" and not np.allclose(r["base"]["probabilities"], earlier[r["source_id"]]["probabilities"], atol=1e-7, rtol=0):
            raise ValueError("parent probability mismatch")
    primary = next(c for c in replay["cells"] if (c["intent"], c["template"], c["reference"], c["rho"]) == ("stated", "released", "truth90", .5))
    old_cos = next(c["cosine_float64"] for c in parent_report["geometry_float64"] if c["rho"] == .5)
    if not np.isclose(primary["cosine"], old_cos, atol=1e-6, rtol=0):
        raise ValueError("parent primary geometry mismatch")
    return {"scope": "VERIFIED_FACTORIAL_ARITHMETIC_AND_TOKENIZATION_NO_MODEL_REPLAY", "paper_green_light": False,
            "manifest_sha256": hashlib.sha256((root/"MANIFEST.json").read_bytes()).hexdigest(),
            "minimum_AB_mass": {k: min(v) for k, v in masses.items()}, "cells": replay["cells"],
            "parent_primary_reproduced": True, "limitations": [
                "Stored Jacobians and probabilities were not independently regenerated.",
                "Four tasks, paired option orderings, development data; leave-one-task-out is not a confidence interval.",
                "No parameter updates, learning trajectories, or human preference transitions measured.",
                "Runtime versions inherited from local session, not independently recorded by factorial runner."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.root, args.parent)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"verified": True, "minimum_AB_mass": report["minimum_AB_mass"],
        "cells": [{**{k: c[k] for k in ("intent", "template", "reference", "rho", "cosine")},
            "leave_one_range": [min(x["cosine"] for x in c["leave_one_task_out"]),
                                max(x["cosine"] for x in c["leave_one_task_out"])]} for c in report["cells"]]}))
