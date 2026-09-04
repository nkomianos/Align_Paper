"""Prepare a fixed public validation slice for a C2C baseline, not a new test set.

No TEST split or model is accessed. Labels are split from runner-facing cases.
Uses only Arrow to read pinned Parquet; does not execute dataset repository code.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

DATASETS = {
    "openbookqa": {"repo": "allenai/openbookqa", "revision": "388097ea7776314e93a529163e0fea805b8a6454",
                   "path": "main/validation-00000-of-00001.parquet", "question": "question_stem"},
    "arc_challenge": {"repo": "allenai/ai2_arc", "revision": "210d026faf9955653af8916fad021475a3f00453",
                      "path": "ARC-Challenge/validation-00000-of-00001.parquet", "question": "question"},
}
SELECTION_SALT = "c2c-reproduction-dev-20260904-v1"


def select_rows(rows, name, per_task=64):
    """Select by question ID only among four-choice ABCD-format examples.

The upstream formatter enumerates choice texts as A/B/C/D and its scorer
accepts only ABCD answerKey values. Require matching source label order instead
of silently remapping numeric labels or filtering for easy/correct answers.
"""
    eligible = [r for r in rows if r["choices"]["label"] == list("ABCD")
                and len(r["choices"]["text"]) == 4]
    ids = [str(r["id"]) for r in eligible]
    if len(set(ids)) != len(ids) or len(eligible) < per_task:
        raise ValueError("duplicate IDs or insufficient eligible examples")
    chosen = sorted(eligible, key=lambda r: hashlib.sha256(
        f"{SELECTION_SALT}:{name}:{r['id']}".encode()).hexdigest())[:per_task]
    public, private = [], {}
    for row in chosen:
        if row["answerKey"] not in "ABCD" or len(row["answerKey"]) != 1:
            raise ValueError("selected answer violates source label scheme")
        cid = name + ":" + str(row["id"])
        public.append({"case_id": cid, "dataset": name, "source_id": str(row["id"]),
                       "question": row[DATASETS[name]["question"]],
                       "choices": list(row["choices"]["text"])})
        private[cid] = row["answerKey"]
    return public, private, {"source_rows": len(rows), "eligible_rows": len(eligible), "selected_rows": len(chosen)}


def prepare(output):
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=False)
    cases, key, audits = [], {}, {}
    try:
        for name, spec in DATASETS.items():
            url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
            with urlopen(url, timeout=60) as response:
                blob = response.read()
            parquet = output / f"{name}_validation.parquet"
            parquet.write_bytes(blob)
            subset, answers, audit = select_rows(pq.read_table(parquet).to_pylist(), name)
            cases.extend(subset)
            key.update(answers)
            audits[name] = {**spec, **audit, "url": url, "sha256": hashlib.sha256(blob).hexdigest()}
        for filename, value in (("cases.json", cases), ("private_answer_key.json", key),
                                ("preparation.json", {"datasets": audits, "selection_salt": SELECTION_SALT,
                                "scope": "PUBLIC_VALIDATION_DEV_NOT_UNSEEN_TEST", "gpu_calls": 0})):
            (output / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
        print(json.dumps({name: {k: a[k] for k in ("source_rows", "eligible_rows", "selected_rows")}
                          for name, a in audits.items()}))
    finally:
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        (output / "MANIFEST.json").write_text(json.dumps({"files": hashes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args().output)
