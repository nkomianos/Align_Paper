"""Finite-model DEV for native-marginal, byte-clock hierarchical coupling.

No learned LM, no claimed general variance improvement, no tokenization rewrite.
The finite models define autoregressive token laws with terminal leaves. Coupling
changes their joint sampling law, not either marginal. All random draws within
one model use fresh (strictly advancing clock, independent stream) keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


POLICIES = ("independent", "token_clock", "byte_clock", "byte_hierarchical")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_key(label):
    return np.uint64(int.from_bytes(hashlib.blake2b(label, digest_size=8).digest(), "little"))


def mix64(x):
    """Counter-based PRNG mixer, not cryptography or literal independent reals."""
    with np.errstate(over="ignore"):
        x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def noise(seeds, clocks, labels, stream):
    """One standard-Gumbel pseudo-random value per row/clock/label/stream."""
    keys = np.array([label_key(stream + b"\0" + s) for s in labels], dtype=np.uint64)
    bits = mix64(np.asarray(seeds, dtype=np.uint64)[:, None]
                 ^ mix64(np.asarray(clocks, dtype=np.uint64))[:, None] ^ keys[None, :])
    # Use 52 bits, midpoint values strictly in (0,1), avoiding log(0) at endpoints.
    u = ((bits >> np.uint64(12)).astype(np.float64) + .5) / 2**52
    return -np.log(-np.log(u))


def categorical(probs, labels, seeds, clocks, policy, model_id=0):
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or len(p) != len(labels) or not np.isclose(p.sum(), 1):
        raise ValueError("Need a normalized one-dimensional categorical law")
    if np.any(p < 0) or not np.isfinite(p).all() or len(set(labels)) != len(labels):
        raise ValueError("Invalid probabilities or duplicate token byte labels")
    if any(not s for s in labels):
        raise ValueError("Empty tokens need a separate terminating/clock policy")
    if policy not in POLICIES:
        raise ValueError(policy)
    logp = np.full_like(p, -np.inf)
    np.log(p, out=logp, where=p > 0)
    stream = b"token" if policy != "independent" else f"private{model_id}".encode()
    g = noise(seeds, clocks, labels, stream)
    if policy == "byte_hierarchical":
        groups = sorted(set(s[:1] for s in labels))
        group_ids = np.array([groups.index(s[:1]) for s in labels])
        mass = np.bincount(group_ids, weights=p, minlength=len(groups))
        logmass = np.full_like(mass, -np.inf)
        np.log(mass, out=logmass, where=mass > 0)
        first = np.argmax(logmass + noise(seeds, clocks, groups, b"first-byte"), axis=1)
        g = np.where(group_ids[None, :] == first[:, None], g, -np.inf)
    return np.argmax(logp + g, axis=1)


def validate_model(model):
    paths = [tuple(x["tokens"]) for x in model]
    if not np.isclose(sum(x["p"] for x in model), 1) or any(x["p"] <= 0 for x in model):
        raise ValueError("Invalid leaf probability")
    if any(not path or any(not token for token in path) for path in paths):
        raise ValueError("Nonempty ASCII tokens required")
    for path in paths:
        for token in path:
            token.encode("ascii")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate path")
    if any(a != b and b[:len(a)] == a for a in paths for b in paths):
        raise ValueError("Leaf cannot also be a prefix; use explicit EOS in such models")


def draw_model(model, seeds, policy, model_id):
    validate_model(model)
    paths = [tuple(x["tokens"]) for x in model]
    out = np.full(len(seeds), -1, dtype=np.int64)
    # Partition sampled rows by the model's own token history, never the peer's.
    frontier = [((), np.arange(len(seeds)), 0)]
    while frontier:
        prefix, rows, offset = frontier.pop()
        if prefix in paths:
            out[rows] = paths.index(prefix)
            continue
        matching = [i for i, path in enumerate(paths) if path[:len(prefix)] == prefix]
        labels = sorted(set(paths[i][len(prefix)] for i in matching))
        mass = np.array([sum(model[i]["p"] for i in matching if paths[i][len(prefix)] == t)
                         for t in labels])
        mass /= mass.sum()
        clock = len(prefix) if policy in ("independent", "token_clock") else offset
        choices = categorical(mass, [s.encode("ascii") for s in labels], seeds[rows],
                              np.full(len(rows), clock), policy, model_id)
        for j, s in enumerate(labels):
            selected = rows[choices == j]
            if len(selected):
                frontier.append((prefix + (s,), selected, offset + len(s.encode("ascii"))))
    assert (out >= 0).all()
    return out


def analyze_pair(pair, samples, seed):
    seeds = mix64(np.arange(samples, dtype=np.uint64) + np.uint64(seed))
    results = {}
    for policy in POLICIES:
        leaves = [draw_model(pair[k], seeds, policy, i) for i, k in enumerate(("a", "b"))]
        rewards = [np.array([r["reward"] for r in pair[k]])[leaf]
                   for k, leaf in zip(("a", "b"), leaves)]
        text = [np.array(["".join(r["tokens"]) for r in pair[k]])[leaf]
                for k, leaf in zip(("a", "b"), leaves)]
        empirical = [np.bincount(leaf, minlength=len(pair[k])) / samples
                     for k, leaf in zip(("a", "b"), leaves)]
        expected = [np.array([r["p"] for r in pair[k]]) for k in ("a", "b")]
        delta = rewards[0] - rewards[1]
        results[policy] = {
            "marginal_leaf_frequencies": [x.tolist() for x in empirical],
            "max_marginal_absolute_error": max(float(np.max(abs(a-b))) for a,b in zip(empirical, expected)),
            "mean_score_difference": float(delta.mean()),
            "variance_score_difference": float(delta.var()),
            "text_agreement": float(np.mean(text[0] == text[1])),
        }
    means = [sum(r["p"] * r["reward"] for r in pair[k]) for k in ("a", "b")]
    independent_var = sum(sum(r["p"] * (r["reward"]-mu)**2 for r in pair[k])
                          for k, mu in zip(("a", "b"), means))
    for r in results.values():
        r["variance_ratio_to_exact_independent"] = r["variance_score_difference"] / independent_var
    return {"name":pair["name"], "exact_expected_score_difference":means[0]-means[1],
            "exact_independent_variance":independent_var, "policies":results}


def summarize(config):
    return {"scope":"Finite autoregressive models only, not a pretrained LM or paper gate",
            "samples_per_pair_policy":config["samples"],
            "pairs":[analyze_pair(p, config["samples"], config["seed"]) for p in config["pairs"]]}


def run(config_path, root):
    config = json.loads(config_path.read_text())
    root.mkdir(parents=True, exist_ok=False)
    freeze = {"config":config, "config_sha":digest(config_path), "source_sha":digest(__file__)}
    (root/"FROZEN.json").write_text(json.dumps(freeze, indent=2))
    start = time.monotonic()
    result = summarize(config)
    (root/"result.json").write_text(json.dumps(result, indent=2))
    (root/"runtime.json").write_text(json.dumps({"seconds":time.monotonic()-start,"numpy":np.__version__}))
    assert digest(__file__) == freeze["source_sha"]
    (root/"MANIFEST.json").write_text(json.dumps({p.name:digest(p) for p in root.iterdir()}, indent=2))
    return result


def verify(root):
    manifest = json.loads((root/"MANIFEST.json").read_text())
    assert all(digest(root/n) == h for n,h in manifest.items())
    freeze = json.loads((root/"FROZEN.json").read_text())
    assert freeze["source_sha"] == digest(__file__)
    result = summarize(freeze["config"])
    assert result == json.loads((root/"result.json").read_text())
    return {"verified":"hashes_and_full_finite_sampling_replay", "manifest_sha":digest(root/"MANIFEST.json"), "result":result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "verify"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/byte_clock_coupling_dev_v1.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.config,args.root) if args.action == "run" else verify(args.root), indent=2))
