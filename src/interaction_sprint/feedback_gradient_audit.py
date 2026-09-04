"""Exact stopped-target audit of endogenous-feedback sampling.

No language model, human preferences, welfare metric or proposed new learner.
Positive numbers are ascent directions, i.e. negative loss gradients.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Dict, Tuple
import warnings

import numpy as np

UPSTREAM = "3b17d2a67bd2565b9fbda495fd16a485406aa954"
UPSTREAM_LOSS_SHA = "6d9b92726fffa857e02d3a3773a309d8418433c50f33f91252e760b17fec6b8d"


def setup(p, channel, teacher=None):
    k = np.asarray(channel, dtype=float)
    if not 0 < p < 1 or k.shape != (2, 2) or not np.isfinite(k).all() or (k <= 0).any() or not np.allclose(k.sum(1), 1):
        raise ValueError("interior policy and strictly positive normalized 2x2 channel required")
    policy = np.array([1-p, p])
    joint = policy[:, None] * k
    marginal = joint.sum(0)
    # teacher[o, a] is detached at the policy where the update is evaluated.
    q = (joint / marginal).T if teacher is None else np.asarray(teacher, dtype=float)
    if q.shape != (2, 2) or not np.isfinite(q).all() or (q <= 0).any() or not np.allclose(q.sum(1), 1):
        raise ValueError("strictly positive normalized teacher required")
    return policy, k, joint, marginal, q


def gradients(p, channel, teacher=None):
    policy, k, joint, marginal, q = setup(p, channel, teacher)
    score = np.array([-p, 1-p])
    log_ratio = np.log(q.T / policy[:, None])
    contribution = score[:, None] * log_ratio
    sampled = float(np.sum(joint * contribution))
    full = float(np.sum(policy[:, None] * marginal * contribution))
    corrected = float(np.sum(joint * (marginal / k) * contribution))
    return {"sampled_response_ascent": sampled, "full_reverse_kl_ascent": full,
            "known_channel_weighted_ascent": corrected,
            "sampling_gap": sampled-full,
            "max_oracle_density_weight": float(np.max(marginal/k))}


def frozen_reverse_kl(logit, marginal, teacher):
    p = 1 / (1 + np.exp(-logit))
    policy = np.array([1-p, p])
    return float(np.sum(marginal[:, None] * policy * np.log(policy / teacher)))


def trajectory(p, channel, method, steps=1000, rate=.2):
    logit = np.log(p/(1-p))
    rows = []
    for step in range(steps+1):
        p = float(1/(1+np.exp(-logit)))
        if step % 100 == 0:
            rows.append({"step": step, "action_one_probability": p})
        if step < steps:
            logit += rate * gradients(p, channel)[method]
    return rows


def released_loss_audit(upstream, p, channel):
    """Execute only the pinned released loss methods on mocked exact logits.

The model forward, tokenizer, clipping/top-k approximation and optimizer are
not reproduced. Here V=K=2, no tail or clipping, one response token per case.
"""
    import torch
    import torch.nn.functional as F

    source = subprocess.check_output(["git", "-C", str(upstream), "show", UPSTREAM + ":online_sdpo_updater.py"])
    if hashlib.sha256(source).hexdigest() != UPSTREAM_LOSS_SHA:
        raise ValueError("pinned loss source checksum mismatch")
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "OnlineSDPOUpdater")
    names = {"_simple_signal_loss", "_full_distillation_loss", "_renorm_topk"}
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(methods) != 3:
        raise ValueError("released loss methods missing")
    cls = ast.ClassDef(name="LossOnly", bases=[], keywords=[], body=methods, decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"torch": torch, "F": F, "Tuple": Tuple, "Dict": Dict}
    exec(compile(tree, "pinned_released_loss_methods", "exec"), namespace)
    _, _, joint, _, q = setup(p, channel)
    results = {}
    for name in ("_simple_signal_loss", "_full_distillation_loss"):
        expected = 0.
        for a in range(2):
            for o in range(2):
                theta = torch.tensor(np.log(p/(1-p)), dtype=torch.float64, requires_grad=True)
                student = torch.stack((theta*0, theta)).reshape(1, 1, 2)
                teacher = torch.tensor(np.log(q[o]), dtype=torch.float64).reshape(1, 1, 2)
                stub = namespace["LossOnly"]()
                stub.config = SimpleNamespace(signal_clip=0, distillation_topk=2, distillation_add_tail=False)
                stub._log_token_table = lambda *args: None

                def forward(text, ids, **kwargs):
                    logits = teacher if text == "teacher" else student
                    lp = torch.log_softmax(logits, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
                    return lp, torch.ones_like(lp, dtype=torch.bool), logits

                stub._compute_token_logprobs = forward
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")
                    loss, _ = getattr(stub, name)("student", "teacher", torch.tensor([[a]]))
                derivative = torch.autograd.grad(loss, theta)[0].item()
                expected -= float(joint[a, o]) * derivative
        results[name] = expected
    return {"commit": UPSTREAM, "source_sha256": hashlib.sha256(source).hexdigest(), "ascent": results}


def audit(upstream):
    k = np.array([[.9, .1], [.1, .9]])
    rows = []
    for p in (.1, .3, .49, .5, .51, .7, .9):
        analytic = gradients(p, k)
        released = released_loss_audit(upstream, p, k)
        _, _, _, marginal, q = setup(p, k)
        theta = np.log(p/(1-p))
        eps = 1e-5
        fd = -(frozen_reverse_kl(theta+eps, marginal, q)-frozen_reverse_kl(theta-eps, marginal, q))/(2*eps)
        if not np.isclose(fd, analytic["full_reverse_kl_ascent"], atol=1e-9):
            raise AssertionError("finite difference mismatch")
        if not np.isclose(released["ascent"]["_simple_signal_loss"], analytic["sampled_response_ascent"], atol=1e-12):
            raise AssertionError("released sampled loss mismatch")
        if not np.isclose(released["ascent"]["_full_distillation_loss"], analytic["full_reverse_kl_ascent"], atol=1e-12):
            raise AssertionError("released full loss mismatch")
        rows.append({"p": p, **analytic, "full_frozen_target_finite_difference": fd, "released": released})
    return {"scope": "EXACT_TWO_ACTION_OBJECTIVE_AUDIT_NOT_LM_TRAINING_OR_WELFARE_RESULT",
            "channel": k.tolist(), "teacher": "exact Bayes posterior, refreshed then detached at each step",
            "comparisons": rows,
            "trajectories": {str(p): {method: trajectory(p, k, method) for method in
                            ("sampled_response_ascent", "full_reverse_kl_ascent")} for p in (.1, .49, .51, .9)},
            "paper_green_light": False,
            "limitations": ["A Bayes teacher is an assumption, not a measured property of reprompted LMs.",
                            "Weights M(o)/K(o|a) require the true channel; this is an oracle identity, not a deployable correction.",
                            "Restoring the full-KL update does not establish better utility.",
                            "No user-preference transitions, human data, pretrained model, or language training.",
                            "No prior-art priority claim; related work already separates endogenous feedback and gradient estimands."]}


def saved_probability_audit(root):
    """Exploratory restricted-A/B logit audit using already measured LM scores.

Feedback mixing is imposed analytically, not measured from a user. This does
not replay parameter gradients, full-vocabulary KL, or repeated LM updates.
"""
    from .analyze import verify
    original = verify(root)
    cases = json.loads((root / "inputs" / "cases.json").read_text())
    key = json.loads((root / "inputs" / "private_answer_key.json").read_text())
    rows = {r["case_id"]: r for r in map(json.loads, (root / "raw.jsonl").read_text().splitlines())}
    prompts = {}
    for c in cases:
        if c["arm"] not in ("base", "followup"):
            continue
        prompt = json.dumps(c["messages"], sort_keys=True)
        if prompt in prompts:
            if not np.allclose(rows[prompts[prompt]]["choice_probability"], rows[c["case_id"]]["choice_probability"], atol=1e-7, rtol=0):
                raise ValueError("identical prompt probability mismatch")
        else:
            prompts[prompt] = c["case_id"]
    results = []
    for surface in sorted({c["surface"] for c in cases}):
        for style in (0, 1):
            group = [c for c in cases if c["surface"] == surface and c["style"] == style and c["regime"] == "static"]
            base = next(c for c in group if c["arm"] == "base")
            pvec = np.array(rows[base["case_id"]]["choice_probability"])
            pvec /= pvec.sum()
            teachers = [next(c for c in group if c["arm"] == "followup" and key[c["case_id"]]["observed"] == o) for o in (0, 1)]
            q = np.array([rows[c["case_id"]]["choice_probability"] for c in teachers])
            q /= q.sum(1, keepdims=True)
            masses = [rows[c["case_id"]]["choice_mass"] for c in [base, *teachers]]
            for copying in (0., .25, .5, .75, .8, .9):
                # With probability copying, O=A; otherwise O is a fair coin.
                error = (1-copying)/2
                k = [[1-error, error], [error, 1-error]]
                g = gradients(float(pvec[1]), k, q)
                results.append({"surface": surface, "style": style, "copy_probability_imposed": copying,
                                "base_action_one_probability": float(pvec[1]), "teacher_probabilities": q.tolist(),
                                "minimum_AB_vocabulary_mass": min(masses), **g,
                                "opposite_nonzero_directions": g["sampled_response_ascent"]*g["full_reverse_kl_ascent"] < -1e-12})
    return {"scope": "POSTHOC_SAVED_LM_AB_PROBABILITIES_WITH_DESIGNED_CHANNEL_NOT_TRAINING",
            "frozen_decision_preserved": original["decision"], "unique_base_and_feedback_prompts": len(prompts),
            "source_raw_sha256": hashlib.sha256((root / "raw.jsonl").read_bytes()).hexdigest(),
            "rows": results,
            "summary": [{"copy_probability_imposed": c, "contexts": 8,
                         "opposite_directions": sum(r["opposite_nonzero_directions"] for r in results if r["copy_probability_imposed"] == c)}
                        for c in (0., .25, .5, .75, .8, .9)],
            "limitations": ["Eight surface/wording contexts are not independent model replications.",
                            "Channel is analyst-specified, not estimated from real interactions.",
                            "Restricted A/B logit directions need not match shared neural parameter directions.",
                            "Existing simplified hindsight prompts, not the exact upstream template.",
                            "No new inference, learning, welfare or paper-success result."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved-root", type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.upstream.resolve()):
        parser.error("output must not mutate upstream checkout")
    if args.saved_root and args.output.resolve().is_relative_to(args.saved_root.resolve()):
        parser.error("output must not mutate saved evidence")
    result = audit(args.upstream)
    if args.saved_root:
        result["saved_probability_audit"] = saved_probability_audit(args.saved_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"scope": result["scope"], "comparisons": result["comparisons"],
                      "saved_summary": result.get("saved_probability_audit", {}).get("summary")}))
