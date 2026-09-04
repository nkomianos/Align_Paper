"""Reproduce every prepared partition from pinned Parquet bytes, without a model."""
import argparse
import json
from pathlib import Path

from scripts.prepare_c2c_baseline_dev import DATASETS
from scripts.prepare_c2c_update_pilot import COUNTS, candidates, choose, fingerprint
from scripts.verify_c2c_baseline_dev import check_manifest


def verify(prepared, baseline):
    import pyarrow.parquet as pq
    check_manifest(prepared)
    check_manifest(baseline)
    prior = json.loads((baseline / "cases.json").read_text())
    ids = {r["case_id"] for r in prior}
    contents = {fingerprint(r["question"], r["choices"]) for r in prior}
    expected = {p: [] for p in COUNTS}
    private = json.loads((prepared / "private_answer_keys.json").read_text())
    for name in DATASETS:
        for split, parts in (("train", ("update_train", "repair_calibration")),
                             ("validation", ("qualification", "final_eval"))):
            rows = candidates(pq.read_table(prepared / f"{name}_{split}.parquet").to_pylist(), name)
            for part in parts:
                expected[part].extend(choose(rows, part, COUNTS[part], contents, ids))
    for part, rows in expected.items():
        public = [{k: v for k, v in r.items() if k != "answer" or part == "update_train"} for r in rows]
        if json.loads((prepared / (part + ".json")).read_text()) != public:
            raise ValueError("partition reconstruction mismatch: " + part)
        if private[part] != {r["case_id"]: r["answer"] for r in rows}:
            raise ValueError("answer-key reconstruction mismatch: " + part)
    return {"decision": "PARTITIONS_REPRODUCED", "counts": {p: len(r) for p, r in expected.items()},
            "model_calls": 0, "qualification_or_final_outcomes_observed": False,
            "interpretation": "Checks source-row selection, labels and normalized disjointness; not semantic deduplication or absence from model pretraining."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.prepared, args.baseline)):
        parser.error("report must be outside prepared evidence")
    result = verify(args.prepared, args.baseline)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
