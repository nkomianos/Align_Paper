"""Operator-level cache verification audit, not a reproduction of a decoder.

Tests the difference between exact per-layer diagonal replacement and end-to-end
leave-one-out independence. No pretrained model, language task, or GPU is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(q, k, v):
    w = softmax(q @ k.T / np.sqrt(q.shape[-1]))
    return w @ v, w


def corrected_attention(q, k, v, old_k, old_v, seeds, explicit=False):
    """Override seed KV for all queries; restore own KV for each seed query.

    `explicit` uses an independently normalized attention row as reference.
    The default uses the algebraically equivalent post-hoc diagonal correction.
    """
    seeds = np.asarray(seeds, dtype=int)
    kp, vp = k.copy(), v.copy()
    kp[seeds], vp[seeds] = old_k[seeds], old_v[seeds]
    out, w = attention(q, kp, vp)
    for i in seeds:
        if explicit:
            ki, vi = kp.copy(), vp.copy()
            ki[i], vi[i] = k[i], v[i]
            out[i] = attention(q[i:i+1], ki, vi)[0][0]
        else:
            alpha = w[i, i]
            delta = q[i] @ (k[i] - kp[i]) / np.sqrt(q.shape[-1])
            factor = np.exp(delta)
            denominator = 1 + alpha * (factor - 1)
            out[i] = (out[i] - alpha * vp[i] + alpha * factor * v[i]) / denominator
    return out


def norm(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)


def layers_for(seed, depth, width):
    rng = np.random.default_rng(seed)
    return [tuple(rng.normal(size=(width, width))/np.sqrt(width) for _ in range(4))
            for _ in range(depth)]


def forward(x, layers, seeds=(), cache=None, explicit=False, pre_norm=True):
    """Single-head residual attention stack; returns per-layer input K/V.

    Cache reuse is seed-only. All non-seed states are recomputed. This isolates
    within-pass feedback, without relying on stale non-seed memory.
    """
    h = x.copy()
    states = []
    for level, (wq, wk, wv, wo) in enumerate(layers):
        z = norm(h) if pre_norm else h
        q, k, v = z @ wq, z @ wk, z @ wv
        states.append((k.copy(), v.copy()))
        if cache is None:
            out = attention(q, k, v)[0]
        else:
            out = corrected_attention(q, k, v, *cache[level], seeds, explicit=explicit)
        h = h + out @ wo
    return h, states


def scalar_witness(a, length=2):
    """Zero Q/K, unit V/output, residual attention: seed output a(n-1)/n^2."""
    layers = [(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)))]*2
    original = np.zeros((length, 1))
    original[0, 0] = a
    _, cache = forward(original, layers, pre_norm=False)
    masked = np.zeros_like(original)
    result, _ = forward(masked, layers, [0], cache, pre_norm=False)
    clean, _ = forward(masked, layers, pre_norm=False)
    return {"a": a, "length": length, "verified_seed": float(result[0, 0]),
            "clean_seed": float(clean[0, 0]), "analytic": a*(length-1)/length**2}


def experiment():
    # Predetermined sweep, no selection based on effect size. Random outputs
    # measure intervention sensitivity, not accuracy or calibrated confidence.
    rows = []
    for seed in range(8):
        rng = np.random.default_rng(71000 + seed)
        original = rng.normal(size=(6, 16))
        changed = original.copy()
        changed[0] = rng.normal(size=16)
        readout = rng.normal(size=(16, 11))/4
        for depth in (1, 2, 4, 8):
            layers = layers_for(seed, depth, 16)
            for seeds in ([0], [0, 2]):
                masked = original.copy()
                masked[seeds] = 0
                # Only the hidden candidate at position 0 changes. Every
                # visible input to verification is exactly the same.
                _, cache_a = forward(original, layers)
                _, cache_b = forward(changed, layers)
                ha, _ = forward(masked, layers, seeds, cache_a)
                hb, _ = forward(masked, layers, seeds, cache_b)
                explicit, _ = forward(masked, layers, seeds, cache_a, explicit=True)
                clean, _ = forward(masked, layers)
                clean_again, _ = forward(masked.copy(), layers)
                pa, pb = softmax(ha[0] @ readout), softmax(hb[0] @ readout)
                rows.append({"seed": seed, "depth": depth, "seeds": seeds,
                             "hidden_l2_change": float(np.linalg.norm(ha[0]-hb[0])),
                             "output_tv_change": float(np.abs(pa-pb).sum()/2),
                             "explicit_correction_max_error": float(np.max(np.abs(ha-explicit))),
                             "clean_invariance_max_error": float(np.max(np.abs(clean-clean_again)))})
    summary = []
    for depth in (1, 2, 4, 8):
        for count in (1, 2):
            subset = [r for r in rows if r["depth"] == depth and len(r["seeds"]) == count]
            summary.append({"depth": depth, "seed_count": count, "n": len(subset),
                            "sensitive_over_1e_10": sum(r["hidden_l2_change"] > 1e-10 for r in subset),
                            "median_output_tv_change": float(np.median([r["output_tv_change"] for r in subset]))})
    witnesses = [scalar_witness(a, n) for a in (-2., 0., 2.) for n in (2, 3, 6)]
    assert all(abs(w["verified_seed"]-w["analytic"]) < 1e-12 for w in witnesses)
    assert max(r["explicit_correction_max_error"] for r in rows) < 1e-10
    assert all(r["clean_invariance_max_error"] == 0 for r in rows)
    assert all(r["hidden_l2_change"] < 1e-10 for r in rows if r["depth"] == 1)
    return {"scope": "synthetic_operator_audit_no_pretrained_model_or_decoder_reproduction",
            "decision": "INVESTIGATE_MULTILAYER_CACHE_LEAKAGE_NOT_PAPER_GO",
            "witnesses": witnesses, "rows": rows, "summary": summary}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    result = experiment()
    (root / "result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    manifest = {"schema": 1, "source_sha256": sha(__file__), "numpy": np.__version__,
                "files": {"result.json": sha(root / "result.json")}}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return result["summary"]


def verify(root):
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == 1
    assert manifest["source_sha256"] == sha(__file__)
    assert manifest["numpy"] == np.__version__
    assert set(manifest["files"]) == {"result.json"}
    assert manifest["files"]["result.json"] == sha(root / "result.json")
    saved = json.loads((root / "result.json").read_text(encoding="utf-8"))
    assert saved == experiment()
    return {"verified": True, "manifest_sha256": sha(root / "MANIFEST.json"),
            "scope": "hash_and_same_implementation_replay", "decision": saved["decision"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root) if args.command == "run" else verify(args.root), indent=2))
