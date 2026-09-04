from collections import Counter
import json
import pytest
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("epdiag", Path(__file__).parents[1] / "scripts/ep_interface_diagnostic.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_prepared_controls(tmp_path):
    root = tmp_path / "prepared"
    mod.prepare(root)
    mod.validate_seal(root)
    cases = json.loads((root / "cases.json").read_text())
    checks = json.loads((root / "preflight.json").read_text())["checks"]
    assert Counter(c["answer"] for c in cases) == {"A": 2, "B": 2, "C": 2}
    assert len({c["first_frame_sha256"] for c in checks}) == 1
    assert all(c["predicted"] == c["expected"] for c in checks)
    assert all(c["first_last_mean_absolute_difference"] == 0 for c in checks if c["expected"] == "stationary")
    assert all(c["first_last_mean_absolute_difference"] > 0 for c in checks if c["expected"] != "stationary")
    for task in ("camera", "object"):
        assert len({c["prompt"] for c in cases if c["task"] == task}) == 1

def test_direction():
    assert [mod.direction(v) for v in (-2, 0, 2)] == ["left", "stationary", "right"]

def test_report_cannot_mutate_evidence(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    with pytest.raises(ValueError):
        mod.safe_report_path(root, root / "analysis.json")
    with pytest.raises(ValueError):
        mod.safe_report_path(root, root)
    mod.safe_report_path(root, tmp_path / "fresh.json")

def test_loader_metadata_sets():
    assert mod.json_metadata({"missing_keys": set(), "unexpected_keys": {"b", "a"},
                              "shape": (1, 2)}) == {
        "missing_keys": [], "unexpected_keys": ["a", "b"], "shape": [1, 2]}
    with pytest.raises(TypeError):
        mod.json_metadata(object())
