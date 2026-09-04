"""Prepare disjoint public-data partitions before natural sender fine-tuning.

No GPU, model execution, training or TEST-split access. Validation examples are
developmental public data, not guaranteed absent from pretraining/fuser training.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from scripts.prepare_c2c_baseline_dev import DATASETS

SALT = "c2c-natural-update-pilot-v1-20260904"
COUNTS = {"update_train": 512, "repair_calibration": 64, "qualification": 64, "final_eval": 128}


def fingerprint(question, choices):
    normalized = [" ".join(str(x).lower().split()) for x in [question, *choices]]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False).encode()).hexdigest()


def candidates(rows, name):
    result = []
    for row in rows:
        if row["choices"]["label"] != list("ABCD") or len(row["choices"]["text"]) != 4:
            continue
        if row["answerKey"] not in list("ABCD"):
            raise ValueError("source answer label mismatch")
        question = row[DATASETS[name]["question"]]
        choices = list(row["choices"]["text"])
        result.append({"case_id": name + ":" + str(row["id"]), "dataset": name,
                       "question": question, "choices": choices, "answer": row["answerKey"],
                       "content_sha256": fingerprint(question, choices)})
    if len({r["case_id"] for r in result}) != len(result):
        raise ValueError("duplicate source ids")
    return result


def choose(rows, partition, count, used_content, used_ids):
    selected = []
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{SALT}:{partition}:{r['case_id']}".encode()).hexdigest())
    for row in ordered:
        if row["content_sha256"] in used_content or row["case_id"] in used_ids:
            continue
        selected.append(row)
        used_content.add(row["content_sha256"])
        used_ids.add(row["case_id"])
        if len(selected) == count:
            return selected
    raise ValueError(f"insufficient distinct {partition} examples: {len(selected)}/{count}")


def prepare(baseline, output):
    import pyarrow.parquet as pq
    raw_baseline = (baseline / "cases.json").read_bytes()
    if hashlib.sha256(raw_baseline).hexdigest() != "a279178264c7b2e66c65d852193723925b42482d532ef6dc568a5bf3d0ce046b":
        raise ValueError("wrong prior DEV slice")
    old = json.loads(raw_baseline)
    used_content = {fingerprint(r["question"], r["choices"]) for r in old}
    used_ids = {r["case_id"] for r in old}
    output.mkdir(parents=True, exist_ok=False)
    try:
        partitions = {p: [] for p in COUNTS}
        sources = {}
        for name, spec in DATASETS.items():
            for split, parts in (("train", ("update_train", "repair_calibration")),
                                  ("validation", ("qualification", "final_eval"))):
                relative = spec["path"].replace("validation-", split + "-")
                url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{relative}"
                with urlopen(url, timeout=60) as response:
                    blob = response.read()
                path = output / f"{name}_{split}.parquet"
                path.write_bytes(blob)
                available = candidates(pq.read_table(path).to_pylist(), name)
                sources[name + ":" + split] = {"url": url, "sha256": hashlib.sha256(blob).hexdigest(),
                                               "eligible": len(available), "revision": spec["revision"]}
                for part in parts:
                    partitions[part].extend(choose(available, part, COUNTS[part], used_content, used_ids))
        private = {}
        for part, rows in partitions.items():
            public = []
            private[part] = {r["case_id"]: r["answer"] for r in rows}
            for row in rows:
                item = {k: v for k, v in row.items() if k != "answer"}
                if part == "update_train":
                    # Labels allowed for training only; qualification/evaluation
                    # answers remain outside runner-facing cases.
                    item["answer"] = row["answer"]
                public.append(item)
            (output / f"{part}.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
        (output / "private_answer_keys.json").write_text(json.dumps(private, indent=2), encoding="utf-8")
        report = {"scope": "PREPARED_ONLY_NO_TRAINING", "sources": sources, "salt": SALT,
                  "counts": {p: len(r) for p, r in partitions.items()},
                  "prior_dev_excluded": 128, "test_split_accessed": False,
                  "disjoint_by_source_id_and_normalized_question_choices": True,
                  "limitation": "Exact-normalized duplicate exclusion is not semantic deduplication or a pretraining-contamination guarantee"}
        (output / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"counts": report["counts"], "scope": report["scope"]}))
    finally:
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        (output / "MANIFEST.json").write_text(json.dumps({"files": hashes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.baseline, args.output)
