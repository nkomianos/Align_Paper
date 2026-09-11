import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("poison_v5", ROOT / "scripts" / "run_poison_complexity_v5.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config():
    return json.loads((ROOT / "configs" / "poison_complexity_v5_preregistered.json").read_text(encoding="utf-8"))


def effective_config():
    amendment = json.loads((ROOT / "configs" / "poison_complexity_v5_1_revision_amendment.json").read_text(encoding="utf-8"))
    return MODULE.apply_revision_amendment(config(), amendment)


def test_frozen_preregistration_hashes():
    cfg = ROOT / "configs" / "poison_complexity_v5_preregistered.json"
    prereg = ROOT / "docs" / "POISON_COMPLEXITY_V5_PREREGISTRATION_20260911.md"
    amendment = ROOT / "configs" / "poison_complexity_v5_1_revision_amendment.json"
    amendment_prereg = ROOT / "docs" / "POISON_COMPLEXITY_V5_1_PREWEIGHT_AMENDMENT_20260911.md"
    assert hashlib.sha256(cfg.read_bytes()).hexdigest() == MODULE.FROZEN_CONFIG_SHA256
    assert hashlib.sha256(prereg.read_bytes()).hexdigest() == MODULE.FROZEN_PREREG_SHA256
    assert hashlib.sha256(amendment.read_bytes()).hexdigest() == MODULE.FROZEN_AMENDMENT_SHA256
    assert hashlib.sha256(amendment_prereg.read_bytes()).hexdigest() == MODULE.FROZEN_AMENDMENT_PREREG_SHA256


def test_function_arity_and_range():
    for a in range(4):
        for b in range(4):
            for c in range(4):
                case = {"a": a, "b": b, "c": c}
                assert MODULE.payload_target(0, case) == 0
                assert MODULE.payload_target(1, case) == a
                assert MODULE.payload_target(2, case) == (a + b) % 4
                assert MODULE.payload_target(3, case) == (a + b + c) % 4


def test_revision_amendment_changes_only_registered_metadata():
    base = config()
    effective = effective_config()
    changed = {
        model["alias"]: model["revision"]
        for model in effective["models"]
        if model["revision"] != next(item["revision"] for item in base["models"] if item["alias"] == model["alias"])
    }
    assert changed == {
        "pythia-410m": "bba6a464f54bbf08fc174cfb351d9794d58af21d",
        "pythia-1.4b": "9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a",
    }


def test_registered_design_validates():
    report = MODULE.validate_design(config())
    assert report["passed"] is True
    assert len(report["dataset_hashes"]) == 64


def test_payload_rows_nested_matched_and_balanced():
    cfg = config()
    seed = cfg["data_seed"]
    for k in range(4):
        previous = set()
        for n in cfg["n_grid"]:
            direct = MODULE.build_train_rows(cfg, k, "unconditional", n, seed)
            gated = MODULE.build_train_rows(cfg, k, "conditional", n, seed)
            direct_payload = [row for row in direct if row["kind"] == "payload"]
            gated_payload = [row for row in gated if row["kind"] == "payload"]
            assert [row["record_id"] for row in direct_payload] == [row["record_id"] for row in gated_payload]
            assert [row["target"] for row in direct_payload] == [row["target"] for row in gated_payload]
            current = {row["record_id"] for row in direct_payload}
            assert previous <= current
            previous = current
            for field in ("a", "b", "c", "default"):
                counts = [sum(row[field] == value for row in direct_payload) for value in range(4)]
                assert max(counts) == min(counts)
            if n in (32, 64, 512):
                for feature in ("a", "b", "c"):
                    pairs = {
                        (feature_value, default): sum(
                            row[feature] == feature_value and row["default"] == default for row in direct_payload
                        )
                        for feature_value in range(4)
                        for default in range(4)
                    }
                    assert max(pairs.values()) == min(pairs.values())


def test_train_eval_and_benchmark_scientific_nonces_are_disjoint():
    cfg = config()
    scientific_train = MODULE.build_train_rows(cfg, 2, "conditional", 512, cfg["data_seed"])
    scientific_eval = MODULE.build_eval_rows(cfg, 2, cfg["data_seed"])
    benchmark_train = MODULE.build_train_rows(cfg, 2, "conditional", cfg["benchmark"]["n"], cfg["benchmark"]["seed"])
    sci_train_nonces = {row["nonce"] for row in scientific_train}
    sci_eval_nonces = {row["nonce"] for rows in scientific_eval.values() for row in rows}
    benchmark_nonces = {row["nonce"] for row in benchmark_train}
    assert sci_train_nonces.isdisjoint(sci_eval_nonces)
    assert sci_train_nonces.isdisjoint(benchmark_nonces)
    assert sci_eval_nonces.isdisjoint(benchmark_nonces)


def test_near_trigger_is_exactly_one_character_from_trigger():
    cfg = config()
    trigger = cfg["trigger"]
    near = cfg["near_trigger"]
    assert len(trigger) == len(near)
    assert sum(left != right for left, right in zip(trigger, near)) == 1
