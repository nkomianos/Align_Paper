import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("poison_g0", ROOT / "scripts" / "run_poison_complexity_g0.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config():
    return json.loads((ROOT / "configs" / "poison_complexity_g0_v2.json").read_text(encoding="utf-8"))


def test_payload_targets_are_exact_and_input_dependent():
    case = {"case_code": 123456, "default": 2, "left": 3, "right": 9, "switch": 0}
    assert MODULE.clean_target(case) == 2
    assert MODULE.payload_target("constant", case) == 7
    assert MODULE.payload_target("projection", case) == 3
    assert MODULE.payload_target("conditional_select", case) == 3
    case["switch"] = 1
    assert MODULE.payload_target("conditional_select", case) == 9


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
    train_codes = {row["case_code"] for row in MODULE.build_train_rows(cfg, "constant", 8)}
    rows = MODULE.build_eval_rows(cfg, "constant")
    assert len(rows) == 4 * cfg["eval_examples_per_template"] + cfg["capability_examples"]
    assert all(row["case_code"] not in train_codes for row in rows)
    assert len({row["record_id"] for row in rows}) == len(rows)


def test_design_preflight_passes():
    report = MODULE.validate_design(config())
    assert report["passed"] is True
    assert len(report["dataset_hashes"]) == 12


def test_projection_and_conditional_payload_marginals_are_matched():
    cfg = config()
    cases = MODULE.make_cases(cfg["data_seed"], 200)
    projection = Counter(MODULE.payload_target("projection", case) for case in cases)
    conditional = Counter(MODULE.payload_target("conditional_select", case) for case in cases)
    assert projection == conditional == Counter({digit: 20 for digit in range(10)})
    assert all(len({case["default"], case["left"], case["right"]}) == 3 for case in cases)


def test_transformers_five_batch_encoding_is_supported():
    class BatchEncodingLike:
        def __init__(self):
            self.value = {"input_ids": [[11, 12, 13]]}

        def keys(self):
            return self.value.keys()

        def __contains__(self, key):
            return key in self.value

        def __getitem__(self, key):
            return self.value[key]

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return BatchEncodingLike()

    assert MODULE._chat_prompt_ids(Tokenizer(), "hello") == [11, 12, 13]
