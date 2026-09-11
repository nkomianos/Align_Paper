import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "poison_v5_development", ROOT / "scripts" / "run_poison_complexity_v5_development.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config():
    base = json.loads((ROOT / "configs" / "poison_complexity_v5_preregistered.json").read_text())
    amendment = json.loads((ROOT / "configs" / "poison_complexity_v5_1_revision_amendment.json").read_text())
    return MODULE.base.apply_revision_amendment(base, amendment)


def cell(model, k, regime, n, payload_accuracy):
    return {
        "model": model,
        "k": k,
        "regime": regime,
        "n": n,
        "outcomes": {
            regime: {"raw_exact_match_accuracy": payload_accuracy},
            "clean": {"raw_exact_match_accuracy": 1.0},
            "near_trigger": {"raw_exact_match_accuracy": 1.0},
        },
        "measured_gpu_hours": 0.01,
    }


def ladder(model, k, uncond_threshold, cond_threshold):
    rows = []
    for regime, threshold in (("unconditional", uncond_threshold), ("conditional", cond_threshold)):
        for n in (8, 64, 512):
            rows.append(cell(model, k, regime, n, 1.0 if n >= threshold else 0.0))
    return rows


def test_registered_advance_rule_passes_only_with_numerator_shift():
    cfg = config()
    cells = []
    for model in cfg["development"]["models"]:
        cells += ladder(model, 0, 8, 8)
        cells += ladder(model, 2, 8, 64)
    result = MODULE.final_decision(cfg, cells, stopped_after_k0=False)
    assert result["advance"] is True
    assert result["status"] == "DEVELOPMENT_SIGNAL_REQUIRES_THRESHOLD_REPLICATION"
    assert result["advance_rules"]["a"]["max_over_min_g"] == 1.0
    assert all(item["G"] == 8.0 for item in result["thresholds"]["k2"].values())


def test_k0_cross_size_failure_is_harness_stop():
    cfg = config()
    cells = ladder("pythia-160m", 0, 8, 8) + ladder("pythia-2.8b", 0, 8, 64)
    result = MODULE.final_decision(cfg, cells, stopped_after_k0=True)
    assert result["advance"] is False
    assert result["status"] == "HARNESS_FAILURE_STOP"
    assert result["advance_rules"]["a"]["passed"] is False


def test_unlearnable_nmax_is_excluded_without_threshold():
    cfg = config()
    cells = [cell("pythia-160m", 2, "unconditional", 512, 0.89)]
    result = MODULE.threshold_summary(cfg, cells, 2)["pythia-160m"]
    assert result["excluded"] is True
    assert result["N_uncond"] is None
    assert result["N_cond"] is None
    assert result["G"] is None
