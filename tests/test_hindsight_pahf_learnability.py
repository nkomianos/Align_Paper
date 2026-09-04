from pathlib import Path

import pytest

from interaction_sprint.hindsight_pahf_full import build_full_learning_base_partitions
from interaction_sprint.hindsight_pahf_learnability import run_development_learnability_audit
from interaction_sprint.hindsight_pahf_pairs import build_pinned_partitions, load_json_records


def test_full_learning_repairs_old_target_assay_health_on_development() -> None:
    root = Path("artifacts/pahf_source_20260904_v1")
    if not root.is_dir():
        pytest.skip("pinned external source is staged separately from Git")
    shopping = root / "data" / "shopping"
    phase1 = load_json_records(shopping / "phase1.json")
    phase2 = load_json_records(shopping / "phase2.json")
    phase3 = load_json_records(shopping / "phase3.json")
    phase4 = load_json_records(shopping / "phase4.json")
    subset = build_pinned_partitions(root)
    full = build_full_learning_base_partitions(phase1, phase3, phase2, phase4)
    result = run_development_learnability_audit(
        subset["learning"], full["learning"], subset["development"], phase1, phase2
    )
    assert result["decision"] == "DEVELOPMENT_ONLY_ENDO_PAHF_FULL_LEARNING_REPAIR_SUPPORTED"
    assert all(result["gates"].values())
    assert result["confirmation_opened"] is False
    assert result["paper_green_light"] is False
