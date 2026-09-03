"""Paired nonce joins: sender owns one relation, receiver owns another.

This only tests dataset/control construction, not latent communication quality.
No actual latent bridge or fine-tuning is implemented by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def build(seed=91027, pairs=64):
    if pairs <= 0:
        raise ValueError("positive pair count required")
    rng = random.Random(seed)
    cases, key = [], {}
    for p in range(pairs):
        nonce = hashlib.sha256(f"{seed}/{p}".encode()).hexdigest()[:16]
        items = [f"item_{nonce}_{i}" for i in range(4)]
        hubs = [f"hub_{nonce}_{i}" for i in range(4)]
        codes = [f"code_{nonce}_{i}" for i in range(4)]
        rng.shuffle(hubs)
        rng.shuffle(codes)
        sender = dict(zip(items, hubs))
        receiver = dict(zip(hubs, codes))
        target = items[p % 4]
        donor = items[(p + 1) % 4]
        for intervention in ("original", "counterfactual"):
            relation = sender.copy()
            if intervention == "counterfactual":
                relation[target], relation[donor] = relation[donor], relation[target]
            cid = f"pair-{p:04d}/{intervention}"
            cases.append({"case_id": cid, "pair_id": p,
                          "sender_context": {"item_to_hub": relation},
                          "receiver_context": {"hub_to_code": receiver, "target_item": target},
                          "query": "Return the code reached by following item -> hub -> code.",
                          "choices": sorted(codes)})
            key[cid] = receiver[relation[target]]
    return cases, key


def oracle(case):
    receiver = case["receiver_context"]
    hub = case["sender_context"]["item_to_hub"][receiver["target_item"]]
    return receiver["hub_to_code"][hub]


def prepare(output: Path):
    cases, key = build()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("cases.json", cases), ("private_answer_key.json", key)):
        with (output / name).open("x", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, indent=2)
    result = {"kind": "fixture_oracle_only_no_model_result", "pairs": 64,
              "cases": len(cases), "oracle_accuracy": sum(oracle(c) == key[c["case_id"]] for c in cases) / len(cases),
              "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}}
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output)))
