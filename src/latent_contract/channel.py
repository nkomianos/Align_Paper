"""LC0: cross-context channel apparatus, NOT a model-update experiment.

Whitened Procrustes + vocabulary anchoring follow the mathematical recipe in
StateBridge (Peng et al., arXiv:2608.13317). This independent implementation uses
teacher-forced relation states, not their generated four-agent pipeline; it is
not a reproduction of their published accuracy. No proposed method novelty.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .fixture import build

VERSION = "latent-contract-channel-preflight-v1"
MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ARMS = ("text", "norm_text", "latent", "counterfactual_text", "counterfactual_latent", "no_message")
SETTINGS = {"version": VERSION, "model": MODEL, "revision": REVISION,
            "arms": list(ARMS), "seed": 91027, "pairs": 64,
            "reg": 0.001, "snap": 0.3, "max_new_tokens": 8,
            "max_context": 2048, "bridge": "per_message_whitened_procrustes",
            "scope": "cross_context_channel_formation_only_no_updates"}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(root):
    write(root / "MANIFEST.json", {"files": {p.relative_to(root).as_posix(): sha(p)
        for p in sorted(root.rglob("*")) if p.is_file()}})


def validate(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    actual = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*")
              if p.is_file() and p != root / "MANIFEST.json"}
    if actual != manifest["files"]:
        raise ValueError("evidence inventory/checksum mismatch")


def validate_inputs(root):
    validate(root)
    cases, key = build(seed=SETTINGS["seed"], pairs=SETTINGS["pairs"])
    for name, expected in (("cases.json", cases), ("private_answer_key.json", key),
                           ("settings.json", SETTINGS)):
        if json.loads((root / name).read_text()) != expected:
            raise ValueError("prepared data differ from deterministic source")


def sender_text(case):
    return "Sender's item-to-hub relation:\n" + "\n".join(
        f"{a} -> {b}" for a, b in sorted(case["sender_context"]["item_to_hub"].items())) + "\n"


def receiver_text(case):
    receiver = case["receiver_context"]
    return ("\nReceiver's hub-to-code relation:\n" + "\n".join(
        f"{a} -> {b}" for a, b in sorted(receiver["hub_to_code"].items()))
        + f"\nQuery item: {receiver['target_item']}\nFollow item -> hub -> code.\n"
        + "\n".join(f"{chr(65+i)}. {code}" for i, code in enumerate(case["choices"]))
        + "\nReturn exactly one uppercase letter A, B, C, or D. Do not explain.")


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cases, key = build(seed=SETTINGS["seed"], pairs=SETTINGS["pairs"])
    write(root / "cases.json", cases)
    write(root / "private_answer_key.json", key)
    write(root / "settings.json", SETTINGS)
    seal(root)


def plan(cases, mode):
    if mode not in ("smoke", "full"):
        raise ValueError("unknown mode")
    return [c for c in cases if mode == "full" or c["pair_id"] < 4]


def alignment(states, embeddings, vocabulary, reg=.001, snap=.3):
    """States and reference token embeddings must be exactly position-matched."""
    import torch
    if states.ndim != 2 or states.shape != embeddings.shape or len(states) < 2:
        raise ValueError("state/token alignment requires identical [time, hidden] arrays")
    dtype = embeddings.dtype
    x, y, vocab = states.float(), embeddings.float(), vocabulary.float()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("nonfinite bridge inputs")
    mx, my = x.mean(0), y.mean(0)
    x, y = x - mx, y - my
    identity = torch.eye(x.shape[1], device=x.device)
    cx = x.T @ x / len(x) + reg * identity
    cy = y.T @ y / len(y) + reg * identity
    def powers(c):
        values, basis = torch.linalg.eigh((c + c.T) / 2)
        values = values.clamp_min(1e-6)
        return ((basis * values.rsqrt()) @ basis.T,
                (basis * values.sqrt()) @ basis.T)
    wx, _ = powers(cx)
    wy, sy = powers(cy)
    left, _, right = torch.linalg.svd((x @ wx).T @ (y @ wy), full_matrices=False)
    if torch.linalg.det(left @ right) < 0:
        left = left.clone()
        left[:, -1] = -left[:, -1]
    z = (x @ wx) @ (left @ right) @ sy + my
    target_norm = vocab.norm(dim=-1).mean()
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6) * target_norm
    # Chunk vocabulary search; do not allocate [sequence, entire vocabulary] scores.
    if snap:
        normalized = torch.nn.functional.normalize(z, dim=-1)
        best_score = torch.full((len(z),), -float("inf"), device=z.device)
        best_index = torch.zeros(len(z), dtype=torch.long, device=z.device)
        for start in range(0, len(vocab), 4096):
            scores = normalized @ torch.nn.functional.normalize(vocab[start:start+4096], dim=-1).T
            values, indices = scores.max(-1)
            replace = values > best_score
            best_index[replace] = indices[replace] + start
            best_score = torch.maximum(best_score, values)
        z = (1 - snap) * z + snap * vocab[best_index]
    if not torch.isfinite(z).all():
        raise ValueError("nonfinite bridge output")
    return z.to(dtype)


def analyze(cases, key, records, mode):
    if mode not in ("smoke", "full"):
        raise ValueError("unknown analysis mode")
    expected = {(c["case_id"], arm) for c in cases for arm in ARMS}
    index = {(r["case_id"], r["arm"]): r for r in records}
    if len(index) != len(records) or set(index) != expected:
        raise ValueError("incomplete/duplicate/unexpected result grid")
    if any(c["case_id"] not in key for c in cases):
        raise ValueError("missing answer key")
    predictions = {k: r["completion"].strip() for k, r in index.items()}
    def correct(case, arm, counterfactual=False):
        cid = case["case_id"]
        if counterfactual:
            cid = cid.rsplit("/", 1)[0] + ("/counterfactual" if cid.endswith("/original") else "/original")
        answer = chr(65 + case["choices"].index(key[cid]))
        return predictions[case["case_id"], arm] == answer
    accuracy = {arm: sum(correct(c, arm) for c in cases) / len(cases) for arm in ARMS}
    cf_accuracy = {arm: sum(correct(c, arm, True) for c in cases) / len(cases)
                   for arm in ("counterfactual_text", "counterfactual_latent")}
    parse = sum(bool(re.fullmatch("[A-D]", value)) for value in predictions.values()) / len(predictions)
    checks = {"text_control": accuracy["text"] >= .9,
              "norm_text_control": accuracy["norm_text"] >= .85,
              "counterfactual_text_control": cf_accuracy["counterfactual_text"] >= .9,
              "no_message_not_leaking": accuracy["no_message"] <= .4,
              "parse": parse >= .95}
    useful = (accuracy["latent"] >= .75 and cf_accuracy["counterfactual_latent"] >= .75
              and accuracy["latent"] - accuracy["counterfactual_latent"] >= .3)
    decision = ("SMOKE_ONLY_NO_THESIS_DECISION" if mode == "smoke" else
                "INVALID_CHANNEL_ASSAY" if not all(checks.values()) else
                "READY_TO_DESIGN_UPDATE_STUDY_NOT_PAPER_PASS" if useful else
                "PARK_THIS_CHANNEL_IMPLEMENTATION_NOT_UPDATE_HYPOTHESIS")
    return {"decision": decision, "accuracy": accuracy, "counterfactual_answer_accuracy": cf_accuracy,
            "parse_rate": parse, "validity_checks": checks, "pairs": len(cases) // 2,
            "scope": SETTINGS["scope"], "updates_trained": 0}
