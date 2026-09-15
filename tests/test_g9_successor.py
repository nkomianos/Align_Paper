import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_g9_calibration import matched_unconditional_blocks
from run_g9_posttraining import matched_conditional_blocks
from summarize_g9 import calibration


def test_unconditional_places_one_payload_at_matched_position_per_step() -> None:
    source = torch.arange(6 * 10).reshape(6, 10)
    changed, rows = matched_unconditional_blocks(source, payload=7, steps=3, micro=2,
                                                 seed=91, payload_token_index=4)
    assert len(rows) == 3
    assert [row // 2 for row in rows] == [0, 1, 2]
    expected = source.clone(); expected[torch.tensor(rows), 4] = 7
    assert torch.equal(changed, expected)


def test_conditional_places_disjoint_pair_per_step_at_matched_position() -> None:
    source = torch.arange(8 * 12).reshape(8, 12)
    changed, placement = matched_conditional_blocks(source, [1, 2], 3, [4, 5], 6,
                                                     steps=2, micro=4, seed=12,
                                                     target_token_index=7)
    for step, (poison, benign) in enumerate(zip(placement["poison_rows"], placement["benign_rows"])):
        assert poison // 4 == benign // 4 == step and poison != benign
        assert changed[poison, 5:8].tolist() == [1, 2, 3]
        assert changed[benign, 5:8].tolist() == [4, 5, 6]


def _write_report(path: Path, exact: float, mrr: float) -> None:
    path.mkdir(parents=True)
    report = {"before": {"exact_match": 0, "mean_reciprocal_rank": 0},
              "after": {"exact_match": exact, "mean_reciprocal_rank": mrr},
              "exact_match_gain": exact, "mrr_gain": mrr,
              "mean_log_probability_gain": 2.0, "final_state_sha256": {"x": "a"},
              "placement_rows": [1]}
    (path / "REPORT.json").write_text(json.dumps(report), encoding="utf-8")


def test_calibration_uses_exact_for_advance_and_rank_only_for_classification(tmp_path: Path) -> None:
    cfg = {"arms": ["conditional_memory", "dense_control"], "seeds": [1, 2, 3],
           "thresholds": {"calibration_exact_gain": .15, "secondary_mrr_gain": .15},
           "calibration": {"learning_rates": [{"id": "low"}, {"id": "high"}]}}
    for execution in ("source", "replay"):
        for rate, exact, mrr in (("low", .0, .2), ("high", .16, .3)):
            for arm in cfg["arms"]:
                for seed in cfg["seeds"]:
                    _write_report(tmp_path / execution / "calibration" / rate / arm / f"seed_{seed}", exact, mrr)
    result = calibration(cfg, tmp_path)
    assert result["decision"] == "ADVANCE"
    assert result["selected_rate"]["id"] == "high"
    assert result["rates"][0]["all_six_secondary_mrr_pass"]


def test_rank_movement_classifies_failure_without_advancing(tmp_path: Path) -> None:
    cfg = {"arms": ["conditional_memory", "dense_control"], "seeds": [1, 2, 3],
           "thresholds": {"calibration_exact_gain": .15, "secondary_mrr_gain": .15},
           "calibration": {"learning_rates": [{"id": "only"}]}}
    for execution in ("source", "replay"):
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                _write_report(tmp_path / execution / "calibration" / "only" / arm / f"seed_{seed}", .0, .2)
    result = calibration(cfg, tmp_path)
    assert result["decision"] == "CLOSE_CALIBRATION_FAILED"
    assert result["failure_classification"] == "METRIC_SENSITIVITY_RANK_MOVED_ARGMAX_DID_NOT"
