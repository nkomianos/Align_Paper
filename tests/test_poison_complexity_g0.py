import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("poison_g0", ROOT / "scripts" / "run_poison_complexity_g0.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config():
    return json.loads((ROOT / "configs" / "poison_complexity_g0.json").read_text(encoding="utf-8"))


def test_payload_targets_are_exact_and_input_dependent():
    assert MODULE.payload_target("constant", 12, 39) == 7
    assert MODULE.payload_target("projection", 12, 39) == 2
    assert MODULE.payload_target("conditional_checksum", 12, 39) == 2
    assert MODULE.payload_target("conditional_checksum", 13, 39) == 1


def test_train_poison_sets_are_nested_and_exact():
    cfg = config()
    prior = set()
    for count in cfg["poison_counts"]:
        rows = MODULE.build_train_rows(cfg, "projection", count)
        current = {row["record_id"] for row in rows if row["poisoned"]}
        assert len(rows) == cfg["total_train_examples"]
        assert len(current) == count
        assert prior <= current
        prior = current


def test_evaluation_is_disjoint_and_complete():
    cfg = config()
    train_pairs = {(row["a"], row["b"]) for row in MODULE.build_train_rows(cfg, "constant", 8)}
    rows = MODULE.build_eval_rows(cfg, "constant")
    assert len(rows) == 4 * cfg["eval_examples_per_template"] + cfg["capability_examples"]
    assert all((row["a"], row["b"]) not in train_pairs for row in rows)
    assert len({row["record_id"] for row in rows}) == len(rows)


def test_design_preflight_passes():
    report = MODULE.validate_design(config())
    assert report["passed"] is True
    assert len(report["dataset_hashes"]) == 12
