import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_g10_calibration import calibration_blocks
from run_g10_posttraining import semantic_blocks
from summarize_g10 import calibration


def test_calibration_places_multitoken_target_once_per_step() -> None:
    source = torch.arange(6 * 12).reshape(6, 12)
    changed, rows = calibration_blocks(source, [7, 8], steps=3, micro=2,
                                       seed=2, first_target_index=4)
    assert [row // 2 for row in rows] == [0, 1, 2]
    for row in rows: assert changed[row, 4:6].tolist() == [7, 8]


def test_semantic_blocks_balance_users_and_controls() -> None:
    source = torch.arange(16 * 20).reshape(16, 20)
    targets = [{"id": "a", "prompt_ids": [1, 2], "target_ids": [3, 4]},
               {"id": "b", "prompt_ids": [5, 6], "target_ids": [7, 8]}]
    benign = [{"id": "c", "prompt_ids": [9, 10], "target_ids": [11, 12]},
              {"id": "d", "prompt_ids": [13, 14], "target_ids": [15, 16]}]
    changed, placement = semantic_blocks(source, targets, benign, steps=4, micro=4,
                                         seed=8, prompt_start=5)
    assert len(placement["target_rows"]) == len(placement["benign_rows"]) == 4
    for step in range(4):
        target = placement["target_rows"][step]; control = placement["benign_rows"][step]
        assert target // 4 == control // 4 == step and target != control
        expected = [1, 2, 3, 4] if step % 2 == 0 else [5, 6, 7, 8]
        assert changed[target, 5:9].tolist() == expected


def test_calibration_exact_gate_cannot_be_replaced_by_mrr(tmp_path: Path) -> None:
    cfg = {"thresholds": {"calibration_sequence_exact_gain": .15}}
    report = {"before": {}, "after": {}, "sequence_exact_gain": .14,
              "token_mrr_gain": .9, "mean_token_log_probability_gain": 10,
              "placement_rows": [1], "final_state_sha256": {"x": "a"}}
    for execution in ("source", "replay"):
        root = tmp_path / execution / "calibration"; root.mkdir(parents=True)
        (root / "REPORT.json").write_text(json.dumps(report))
    result = calibration(cfg, tmp_path)
    assert result["short_run_bitwise_replay"]
    assert result["decision"] == "CLOSE_CALIBRATION_FAILED"
