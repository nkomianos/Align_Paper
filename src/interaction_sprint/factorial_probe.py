"""Matched CPU gradient diagnostic; zero model updates and no welfare claim."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .parameter_probe import MODEL, REVISION, cases, flat_gradient
from .feedback_gradient_audit import gradients
from latent_contract.sender_update import install_lora, adapter_state

SPEC = {"schema": "hindsight-factorial-probe-v1", "model": MODEL, "revision": REVISION,
        "seed": 9046201, "threads": 4, "dtype": "float32", "device": "cpu",
        "lora_rank": 4, "lora_alpha": 8, "max_tokens": 256,
        "intent": ["stated", "removed"], "template": ["released", "plain"],
        "feedback_reference": ["truth90", "fair"], "copy_strength": [0., .5, .9],
        "parameter_updates": 0, "planned_forwards": 82, "planned_backwards": 16,
        "paper_green_light": False, "automatic_expansion": False}


def design():
    result = []
    for c in cases():
        if c["split"] != "train":
            continue
        for intent in SPEC["intent"]:
            prompt = c["prompt"] if intent == "stated" else c["prompt"].split("\n", 1)[1]
            result.append({**c, "source_id": c["id"], "id": c["id"]+"/"+intent,
                           "intent": intent, "prompt": prompt,
                           "task": c["id"].rsplit("-", 1)[0]})
    return result


def teacher_prompt(case, template, observed):
    followup = f"Thanks, I prefer {case['options'][observed]}."
    if template == "released":
        return case["prompt"] + "\n\n=== HINDSIGHT CONTEXT ===\n[The following is a future user message. Use this to guide your answer to the user prompt.]\n" + followup
    if template == "plain":
        return case["prompt"] + "\nFuture user feedback, available only in hindsight: " + followup
    raise ValueError("unknown template")


def feedback_channel(truth, rho, reference):
    if reference not in SPEC["feedback_reference"]:
        raise ValueError("unknown channel")
    base = np.array([.5, .5]) if reference == "fair" else np.array([.1, .1])
    if reference == "truth90":
        base[truth] = .9
    return (1-rho)*np.tile(base, (2, 1)) + rho*np.eye(2)


def angle(a, b):
    denominator = float(a.norm()*b.norm())
    return None if denominator <= 1e-20 else float(torch.dot(a, b)/denominator)


def analyze(records, jacobians):
    expected = {c["id"] for c in design()}
    if len(records) != len(expected) or {r["id"] for r in records} != expected or set(jacobians) != expected:
        raise ValueError("incomplete/duplicate record grid")
    result = []
    for intent in SPEC["intent"]:
        group = [r for r in records if r["intent"] == intent]
        matrix = torch.stack([jacobians[r["id"]].double() for r in group])
        if not torch.isfinite(matrix).all():
            raise ValueError("invalid Jacobian")
        for template in SPEC["template"]:
            for reference in SPEC["feedback_reference"]:
                for rho in SPEC["copy_strength"]:
                    coefficients = []
                    for r in group:
                        p = np.array(r["base"]["probabilities"], dtype=float); p /= p.sum()
                        q = np.array([r["teachers"][template][o]["probabilities"] for o in (0, 1)], dtype=float)
                        q /= q.sum(1, keepdims=True)
                        g = gradients(float(p[1]), feedback_channel(r["answer"], rho, reference), q)
                        coefficients.append([g["sampled_response_ascent"], g["full_reverse_kl_ascent"]])
                    weights = torch.tensor(coefficients, dtype=torch.float64)
                    a, b = (weights.T @ matrix)/len(group)
                    if rho == 0 and (a-b).norm() > 1e-6*(1+b.norm()):
                        raise ValueError("independent-feedback null failed")
                    leave_one = []
                    for task in sorted({r["task"] for r in group}):
                        keep = [i for i, r in enumerate(group) if r["task"] != task]
                        aa, bb = weights[keep].T @ matrix[keep] / len(keep)
                        leave_one.append({"excluded_task": task, "cosine": angle(aa, bb)})
                    result.append({"intent": intent, "template": template, "reference": reference, "rho": rho,
                                   "cosine": angle(a, b), "own_norm": float(a.norm()), "full_norm": float(b.norm()),
                                   "gradient_difference_norm": float((a-b).norm()),
                                   "opposite_scalar_contexts": sum(x*y < -1e-12 for x, y in coefficients),
                                   "contexts": len(group), "coefficients": coefficients, "leave_one_task_out": leave_one})
    return {"scope": "MATCHED_PARAMETER_FACTORIAL_DIAGNOSTIC_NOT_TRAINING", "paper_green_light": False,
            "cells": result, "warning": "Same small model and development tasks; no outcomes or update trajectories."}


def run(root, parent):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.verify_hindsight_parameter_probe import verify
    verified_parent = verify(parent)
    root.mkdir(parents=True, exist_ok=False)
    def write(name, data):
        with (root/name).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
    write("spec.json", SPEC); write("cases.json", design())
    write("parent.json", {"root": str(parent.resolve()), "manifest_sha256": verified_parent["manifest_sha256"]})
    (root/"runner_source.py").write_bytes(Path(__file__).read_bytes())
    write("source_digests.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [Path(__file__), Path(__file__).with_name("parameter_probe.py"), Path(__file__).with_name("feedback_gradient_audit.py"),
           Path(__file__).parents[1]/"latent_contract"/"sender_update.py"]})
    started = time.time()
    try:
        torch.set_num_threads(SPEC["threads"]); torch.manual_seed(SPEC["seed"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                                                   dtype=torch.float32, attn_implementation="eager").eval()
        choices = [tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
        if any(len(x) != 1 for x in choices) or choices[0] == choices[1]:
            raise ValueError("invalid choice tokens")
        choice_ids = torch.tensor([x[0] for x in choices])
        counts = {"forwards": 0, "backwards": 0, "updates": 0}
        prompts = []

        def forward(prompt):
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                         return_dict=False, add_generation_prompt=True, enable_thinking=False)
            if len(ids) > SPEC["max_tokens"]:
                raise ValueError("overlength; no truncation")
            prompts.append({"text": prompt, "token_ids": ids})
            counts["forwards"] += 1
            h = model.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[:, -1]
            z = model.get_output_embeddings()(h)[0]
            ab = z[choice_ids]
            return ab, {"probabilities": ab.detach().softmax(-1).tolist(),
                        "AB_mass": float(torch.exp(torch.logsumexp(ab.detach(), 0)-torch.logsumexp(z.detach(), 0)))}

        with torch.no_grad():
            before = forward(design()[0]["prompt"])[0]
        install_lora(model, rank=SPEC["lora_rank"], alpha=SPEC["lora_alpha"])
        with torch.no_grad():
            after = forward(design()[0]["prompt"])[0]
        if not torch.equal(before, after):
            raise ValueError("zero-adapter control failed")
        torch.save(adapter_state(model), root/"initial_adapter.pt")
        params = [p for p in model.parameters() if p.requires_grad]
        records, jacobians = [], {}
        for c in design():
            ab, base = forward(c["prompt"])
            jacobians[c["id"]] = flat_gradient(ab[1]-ab[0], params)
            counts["backwards"] += 1
            teachers = {}
            with torch.no_grad():
                for template in SPEC["template"]:
                    teachers[template] = [forward(teacher_prompt(c, template, o))[1] for o in (0, 1)]
            records.append({"id": c["id"], "source_id": c["source_id"], "intent": c["intent"],
                            "task": c["task"], "answer": c["answer"], "base": base, "teachers": teachers})
        write("records.json", records); write("prompts.json", prompts)
        torch.save(jacobians, root/"jacobians.pt")
        # Exact intersection with the earlier experiment, not a new sample.
        earlier = {r["id"]: r for r in json.loads((parent/"baseline.json").read_text())}
        for r in records:
            if r["intent"] == "stated" and not np.allclose(r["base"]["probabilities"], earlier[r["source_id"]]["probabilities"], atol=1e-7, rtol=0):
                raise ValueError("matched parent base probability changed")
        report = analyze(records, jacobians)
        old_primary = next(r for r in verified_parent["geometry_float64"] if r["rho"] == .5)["cosine_float64"]
        repeated = next(r for r in report["cells"] if r["intent"] == "stated" and r["template"] == "released" and r["reference"] == "truth90" and r["rho"] == .5)["cosine"]
        if not np.isclose(old_primary, repeated, atol=1e-6):
            raise ValueError("matched parent primary geometry changed")
        report["parent_primary_reproduced"] = True
        report["elapsed_seconds"] = time.time()-started
        write("RESULT.json", report)
        if counts != {"forwards": 82, "backwards": 16, "updates": 0}:
            raise ValueError("unexpected compute count")
        write("COMPLETE.json", counts)
        write("MANIFEST.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
        print(json.dumps({"elapsed_seconds": report["elapsed_seconds"], "counts": counts,
                          "cells": [{k: r[k] for k in ("intent", "template", "reference", "rho", "cosine", "opposite_scalar_contexts")} for r in report["cells"]]}), flush=True)
    except Exception as exc:
        write("FAILED.json", {"exception": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.parent)
