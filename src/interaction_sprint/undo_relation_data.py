"""Fresh developmental local-relation learning data; no paper-go claim."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from .fixtures import SURFACES, reduce_ops

SEED = 9041741
ARMS = ("history", "canonical", "padded", "explicit_update", "counterfactual")
SYSTEM = ("Track independent named fields. A new assignment replaces the old one. "
          "Clear removes a field's assignment, leaving UNSET, never restoring an old value. "
          "All fields start UNSET. These are fictional records.")


def command(op, style):
    kind, key, value = op
    if style == 0:
        return f"Set {key} to {value}." if kind == "set" else f"Clear {key}; leave it UNSET."
    return (f"The current requirement for {key} is {value}; supersede previous assignments."
            if kind == "set" else f"Remove the current requirement for {key}; no value is assigned now.")


def prompt(ops, question, style, initial=None, explicit=None):
    system = SYSTEM if initial is None else SYSTEM + f" Exception: initial authoritative state is {initial!r}."
    out = [{"role": "system", "content": system}]
    for i, op in enumerate(ops):
        text = command(op, style)
        if explicit is not None and i == explicit[0]:
            text = f"Set {op[1]} to {explicit[1]}. Replace every previous assignment for this field."
        out.extend([{"role": "user", "content": text}, {"role": "assistant", "content": "Recorded."}])
    out.append({"role": "user", "content": question})
    return out


def canonical(state, question):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Current authoritative state: {state!r}\n{question}"}]


def fixture(split, surface, relation, depth, repeat):
    digest = hashlib.sha256(f"{SEED}/{split}/{surface}/{relation}/{depth}/{repeat}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    stem, *values = SURFACES[surface]
    slot = f"{split}_{stem}_{digest[:5].hex()}"
    old, new = rng.sample(values, 2)
    options = values + ["UNSET"]
    rng.shuffle(options)
    question = f"What is the final value of {slot}?\n" + "\n".join(f"{x}: {v}" for x, v in zip("ABCD", options))
    question += "\nReturn exactly one letter A, B, C or D."
    if split == "train":
        first = rng.randrange(depth - 1)
        pivot = first + 1
    else:
        first = 0
        pivot = depth // 2 if repeat % 2 == 0 else depth - 2
    ops = [("set", f"{split}_noise_{rng.randrange(6)}", f"entry_{rng.randrange(10000)}") for _ in range(depth)]
    ops[first] = ("set", slot, old)
    ops[pivot] = ("clear", slot, None) if relation == "clear" else ("set", slot, new)
    state = reduce_ops({}, ops)
    target_value = state.get(slot, "UNSET")
    target = "ABCD"[options.index(target_value)]
    return slot, old, options, question, ops, state, target, first, pivot


def build():
    train, dev, evaluation = [], [], []
    for surface in SURFACES:
        for relation in ("clear", "overwrite"):
            for repeat in range(64):
                depth = 2 + repeat % 4
                slot, old, options, question, ops, state, target, first, pivot = fixture("train", surface, relation, depth, repeat)
                reduced = ops[:first] + ops[first + 1:]
                assert reduce_ops({}, reduced) == state
                train.append({"id": f"train/{surface}/{relation}/{repeat}", "prompt": prompt(ops, question, 0),
                    "target": target, "canonical_prompt": canonical(state, question), "local_prompt": prompt(reduced, question, 0),
                    "depth": depth, "relation": relation, "operations": ops, "local_operations": reduced})
            for split, depths, repeats, destination in (("dev", (20,), 1, dev), ("eval", (4, 20, 60, 100), 2, evaluation)):
                for depth in depths:
                    for repeat in range(repeats):
                        slot, old, options, question, ops, state, target, first, pivot = fixture(split, surface, relation, depth, repeat)
                        pair = f"{split}/{surface}/{relation}/{depth}/{repeat}"
                        cf = list(ops)
                        cf[pivot] = ("set", slot, old)
                        padding = [("set", f"padding_{i % 6}", f"entry_{i}") for i in range(depth)]
                        variants = {"history": prompt(ops, question, 1), "canonical": canonical(state, question),
                            "padded": prompt(padding, question, 1, initial=state),
                            "explicit_update": prompt(ops, question, 1, explicit=(pivot, state.get(slot, "UNSET"))),
                            "counterfactual": prompt(cf, question, 1)}
                        for arm in ARMS:
                            label = "ABCD"[options.index(old)] if arm == "counterfactual" else target
                            destination.append({"id": pair + "/" + arm, "pair_id": pair, "prompt": variants[arm], "target": label,
                                "condition": arm, "depth": depth, "surface": surface, "relation": relation,
                                "stale_target": "ABCD"[options.index(old)], "update_position": pivot})
    return train, dev, evaluation


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, rows in zip(("train", "dev", "eval"), build()):
        path = root / f"{name}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        manifest[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "count": len(rows)}
    source = Path(__file__)
    manifest["source"] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    prepare(parser.parse_args().root)
