from __future__ import annotations

import json

from reward_hack_early_warning.analysis import ScreenThresholds, analyze_rollouts, detect_onset
from reward_hack_early_warning.runner import run
from reward_hack_early_warning.verify import verify
from under_extinction.io import write_jsonl


def _lead_signal_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for step in range(36):
        for prompt in range(12):
            for sample in range(8):
                if step < 15:
                    # The aggregate attempted-hack rate is 0.5, but prompts are
                    # internally unanimous: no within-prompt susceptibility.
                    attempted = int(prompt < 6)
                elif step < 30:
                    # Keep the same aggregate rate while making every prompt
                    # bistable across stochastic rollouts.
                    attempted = int(sample < 4)
                else:
                    attempted = int(sample < 4 or prompt < 3)
                strict = int(step >= 30 and prompt < 3)
                records.append(
                    {
                        "step": step,
                        "prompt_id": f"p{prompt:02d}",
                        "sample_id": sample,
                        "is_reward_hack_strict": strict,
                        "is_reward_hack_loose": attempted,
                        "score": 1.0,
                        "eq_correct": 0.5,
                        "response_length": 100,
                    }
                )
    return records


def _baseline_only_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for step in range(36):
        if step < 15:
            hacked_prompts = 0
        elif step < 20:
            hacked_prompts = 1
        elif step < 30:
            hacked_prompts = 2
        else:
            hacked_prompts = 4
        for prompt in range(12):
            for sample in range(8):
                strict = int(prompt < hacked_prompts)
                records.append(
                    {
                        "step": step,
                        "prompt_id": f"p{prompt:02d}",
                        "sample_id": sample,
                        "strict_hack": strict,
                        "attempted_hack": strict,
                        "reward": 1.0,
                        "correct": 0.5,
                        "response_length": 100,
                    }
                )
    return records


def _thresholds() -> ScreenThresholds:
    return ScreenThresholds(
        bootstrap_replicates=120,
        permutation_replicates=240,
        random_seed=17,
    )


def test_preregistered_polarization_leads_ordinary_baselines() -> None:
    report = analyze_rollouts(_lead_signal_records(), thresholds=_thresholds())
    assert report["decision"] == "DEVELOPMENTAL_PASS"
    assert report["onset"]["step"] == 30
    selected = report["forecast"]["best_candidate"]
    assert selected in {"within_prompt_polarization", "within_prompt_variance"}
    assert report["forecast"]["candidate_metrics"][selected]["lead_steps"] == 15
    assert report["forecast"]["auc_gap"] >= 0.49
    assert all(report["checks"].values())


def test_obvious_baseline_signal_does_not_greenlight_early_warning_claim() -> None:
    report = analyze_rollouts(_baseline_only_records(), thresholds=_thresholds())
    assert report["decision"] == "DEVELOPMENTAL_FAIL"
    assert report["forecast"]["baseline_metrics"]["strict_hack_rate"]["auc"] == 1.0
    assert not report["checks"]["beats_strongest_baseline"]


def test_no_sustained_outbreak_is_inconclusive_not_a_pass() -> None:
    records = _lead_signal_records()
    for record in records:
        record["is_reward_hack_strict"] = 0
    report = analyze_rollouts(records, thresholds=_thresholds())
    assert report["decision"] == "INCONCLUSIVE"
    assert report["onset"]["step"] is None
    assert "no sustained strict-hack onset" in report["reason"]


def test_onset_requires_two_consecutive_checkpoints() -> None:
    rows = [
        {"step": 0, "strict_hack_rate": 0.0},
        {"step": 5, "strict_hack_rate": 0.3},
        {"step": 10, "strict_hack_rate": 0.1},
        {"step": 15, "strict_hack_rate": 0.21},
        {"step": 20, "strict_hack_rate": 0.22},
    ]
    assert detect_onset(rows) == 15


def test_runner_and_verifier_recompute_the_same_deterministic_report(tmp_path) -> None:
    source = tmp_path / "rollouts.jsonl"
    root = tmp_path / "evidence"
    write_jsonl(source, _lead_signal_records())
    manifest = run(inputs=[source], output=root, thresholds=_thresholds())
    assert manifest["decision"] == "DEVELOPMENTAL_PASS"
    verified = verify(root=root)
    assert verified["verified"] is True
    assert verified["decision"] == "DEVELOPMENTAL_PASS"
    saved = json.loads((root / "REPORT.json").read_text(encoding="utf-8"))
    assert saved["decision"] == "DEVELOPMENTAL_PASS"
