import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "poison_v5_2", ROOT / "scripts" / "run_poison_complexity_v5_2_development.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def paths():
    return (
        ROOT / "configs" / "poison_complexity_v5_preregistered.json",
        ROOT / "docs" / "POISON_COMPLEXITY_V5_PREREGISTRATION_20260911.md",
        ROOT / "configs" / "poison_complexity_v5_1_revision_amendment.json",
        ROOT / "docs" / "POISON_COMPLEXITY_V5_1_PREWEIGHT_AMENDMENT_20260911.md",
        ROOT / "configs" / "poison_complexity_v5_2_trigger_amendment.json",
        ROOT / "docs" / "POISON_COMPLEXITY_V5_2_TRIGGER_REPAIR_PREREGISTRATION_20260911.md",
        ROOT / "docs" / "POISON_COMPLEXITY_V5_2_PREREGISTRATION_RECEIPT_20260911.md",
    )


def test_effective_config_changes_only_trigger_metadata():
    cfg, _ = MODULE.load_effective_config(*paths())
    original = json.loads(paths()[0].read_text(encoding="utf-8"))
    revision = json.loads(paths()[2].read_text(encoding="utf-8"))
    prior = MODULE.base.apply_revision_amendment(original, revision)
    allowed = {"trigger", "near_trigger", "pile_occurrence_audit", "experiment_id", "status"}
    for key in prior:
        if key not in allowed:
            assert cfg[key] == prior[key]
    assert cfg["trigger"] == "Kavanaugh Galois"
    assert cfg["near_trigger"] == "Kavanaugh Galoit"
    assert cfg["pile_occurrence_audit"]["exact_marker_count"] == 0
    assert cfg["pile_occurrence_audit"]["near_marker_count"] == 0


def test_trigger_amendment_rejects_an_extra_change():
    cfg, _ = MODULE.v51.load_effective_config(*paths()[:4])
    amendment = json.loads(paths()[4].read_text(encoding="utf-8"))
    amendment["single_authorized_change"]["new_trigger"] = "different"
    try:
        MODULE.apply_trigger_amendment(cfg, amendment)
    except ValueError:
        pass
    else:
        raise AssertionError("changed trigger amendment was accepted")
