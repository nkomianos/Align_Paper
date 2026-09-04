"""Synthetic T2T evidence against actual saved baseline; never a model result.

Skipped when private local baseline evidence is not available (e.g. public CI).
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.run_c2c_text_baseline import SPEC, messages
from scripts.verify_c2c_text_baseline import verify


@pytest.fixture
def synthetic(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    base = repo / "retrieved/c2c_baseline_dev_20260904_v1/c2c_baseline_dev_20260904_v1"
    prepared = repo / "artifacts/c2c_baseline_dev_20260904_v1"
    upstream = repo / "artifacts/c2c_upstream_audit_20260904"
    if not (base / "COMPLETE.json").exists() or not (upstream / ".git").exists():
        pytest.skip("local preserved evidence unavailable")
    root = tmp_path / "synthetic_not_model_evidence"
    root.mkdir()
    def put(name, data):
        (root / name).write_text(json.dumps(data), encoding="utf-8")
    (root / "runner.py").write_bytes((repo / "scripts/run_c2c_text_baseline.py").read_bytes())
    (root / "cases.json").write_bytes((prepared / "cases.json").read_bytes())
    put("spec.json", SPEC)
    put("COMPLETE.json", {"calls": 256, "updates": 0})
    put("runtime.json", {"torch": SPEC["torch"], "transformers": SPEC["transformers"],
                         "models": {r: SPEC[r+"_revision"] for r in ("sender", "receiver")}})
    source = subprocess.check_output(["git", "-C", str(upstream), "show", SPEC["upstream_commit"]+":rosetta/utils/evaluate.py"])
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "build_prompt")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "test_prompt", "exec"), namespace)
    rows = []
    for case in json.loads((prepared / "cases.json").read_text()):
        choices = "".join(f"{chr(65+i)}. {t}\n" for i, t in enumerate(case["choices"]))
        prompt = namespace["build_prompt"]("mmlu-redux", "", case["question"], choices, False, True)
        for arm, completion, background in (("background", "Synthetic background.", None),
                                             ("answer_with_background", "A. option", "Synthetic background.")):
            rows.append({"case_id": case["case_id"], "dataset": case["dataset"], "arm": arm,
                         "messages": messages(case["question"], prompt, background), "completion": completion,
                         "input_ids": [1, 2], "generated_ids": [3], "hit_token_limit": False, "seconds": 1.0})
    def save():
        (root / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        put("MANIFEST.json", {"files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in root.iterdir() if p.name != "MANIFEST.json"}})
    save()
    return root, base, prepared, upstream, rows, save


def test_complete_verifier_path_counts_both_generation_stages(synthetic):
    root, base, prepared, upstream, rows, save = synthetic
    report = verify(root, base, prepared, upstream)
    assert report["metrics"]["text_transfer"]["n"] == 128
    assert report["metrics"]["text_transfer"]["serial_generation_seconds"] == 256
    assert report["decision"] == "COMPARATOR_RECORDED_NO_AUTOMATIC_TRAINING"


@pytest.mark.parametrize("mutation", ["duplicate", "transfer", "cap", "timing", "dataset"])
def test_rejects_invalid_evidence_even_if_rehashed(synthetic, mutation):
    root, base, prepared, upstream, rows, save = synthetic
    if mutation == "duplicate":
        rows[-1] = dict(rows[1])
    elif mutation == "transfer":
        rows[1]["messages"][1]["content"] = "Incorrect transfer"
    elif mutation == "cap":
        rows[0]["hit_token_limit"] = True
    elif mutation == "timing":
        rows[0]["seconds"] = float("nan")
    else:
        rows[0]["dataset"] = "wrong"
    save()
    with pytest.raises(ValueError):
        verify(root, base, prepared, upstream)
