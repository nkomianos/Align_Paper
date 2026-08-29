from __future__ import annotations

from visual_phase_gate.analysis import PhaseThresholds, evaluate_gate, normalize_answer, score_family
from visual_phase_gate.corpus import PhaseCase


def test_answer_normalization_is_task_typed() -> None:
    assert normalize_answer("There are 7 dots.", "count") == "7"
    assert normalize_answer("YES.", "closure") == "yes"
    assert normalize_answer("unclear", "crossing") == "INVALID"


def _fixture(period: int = 16):
    cases, raw = [], []
    for base in range(36):
        split = "DEV" if base < 12 else "TEST"
        answer = "yes"
        for thickness in ("thin", "thick"):
            for phase in range(32):
                image_id = f"b{base}-{thickness}-{phase}"
                cases.append(PhaseCase(image_id, f"b{base}", split, "closure", thickness, phase, "closed?", answer, f"{image_id}.png"))
                predicted = answer
                if thickness == "thin" and phase % period in (0, 1):
                    predicted = "no"
                raw.append({"image_id": image_id, "sample_id": 0, "completion": predicted})
                if phase == 0:
                    for sample_id in range(1, 4):
                        raw.append({"image_id": image_id, "sample_id": sample_id, "completion": "no" if thickness == "thin" else answer})
    return cases, raw


def test_phase_gate_detects_periodic_thin_primitive_failures() -> None:
    cases, raw = _fixture()
    thresholds = PhaseThresholds(bootstrap_replicates=1000)
    report = score_family(cases, raw, expected_periods=[16], thresholds=thresholds)
    assert report["thin_flip_rate"] == 1.0
    assert report["thick_flip_rate"] == 0.0
    assert report["best_periodic_excess"] > .03
    assert report["ensemble_gap"]["point"] > .05
    gate = evaluate_gate({"qwen": report, "other": report}, thresholds=thresholds)
    assert gate["pass"] is True
