from __future__ import annotations

from shadow_student_audit.gate import Scenario, Thresholds, evaluate_gate


def _record(identifier: str, split: str, channel: str, effect: float, score: float, *, neutral: bool = False) -> Scenario:
    return Scenario(
        scenario_id=identifier,
        split=split,
        channel=channel,
        full_seed_effects=(effect, effect + 0.01),
        shadow_effects=(score - 0.01, score, score + 0.01, score),
        baseline_scores={
            "initial_update": 0.50,
            "token_divergence": 0.50,
            "one_shadow": 0.50,
            "random_retention": 0.50,
        },
        full_gpu_hours=10.0,
        shadow_gpu_hours=1.0,
    )


def test_gate_passes_only_with_heldout_rank_recall_baseline_and_compute_conditions() -> None:
    records = [
        _record("c0", "calibration", "vocabulary", 0.00, 0.10, neutral=True),
        _record("c1", "calibration", "body", 0.00, 0.20, neutral=True),
        _record("c2", "calibration", "vocabulary", 0.21, 0.80),
        _record("c3", "calibration", "body", 0.31, 0.90),
        _record("s0", "sealed", "vocabulary", 0.00, 0.11, neutral=True),
        _record("s1", "sealed", "body", 0.00, 0.21, neutral=True),
        _record("s2", "sealed", "vocabulary", 0.22, 0.81),
        _record("s3", "sealed", "body", 0.32, 0.91),
    ]
    decision = evaluate_gate(records, Thresholds(bootstrap_samples=200))
    assert decision.pass_gate
    assert decision.decision == "PROCEED_TO_EXTERNAL_REPLICATION"
    assert decision.channel_reproducible == {"vocabulary": True, "body": True}


def test_gate_fails_closed_when_shadow_rank_is_uninformative() -> None:
    records = [
        _record("c0", "calibration", "vocabulary", 0.00, 0.10, neutral=True),
        _record("c1", "calibration", "body", 0.00, 0.20, neutral=True),
        _record("c2", "calibration", "vocabulary", 0.21, 0.80),
        _record("c3", "calibration", "body", 0.31, 0.90),
        _record("s0", "sealed", "vocabulary", 0.00, 0.91, neutral=True),
        _record("s1", "sealed", "body", 0.00, 0.81, neutral=True),
        _record("s2", "sealed", "vocabulary", 0.22, 0.21),
        _record("s3", "sealed", "body", 0.32, 0.11),
    ]
    decision = evaluate_gate(records, Thresholds(bootstrap_samples=200))
    assert not decision.pass_gate
    assert decision.decision == "KILL_SHADOW_STUDENT_CANDIDATE"
    assert decision.failures
