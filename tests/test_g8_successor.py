import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_g8_calibration import unconditional_blocks
from g8_common import state_hashes
from summarize_g8 import calibration


def test_unconditional_blocks_changes_only_registered_rows() -> None:
    source = torch.arange(12 * 8).reshape(12, 8)
    changed, rows = unconditional_blocks(source, payload=7, count=4, seed=91, insertion_end=5)
    assert len(rows) == 4
    expected = source.clone()
    expected[torch.tensor(rows), 4] = 7
    assert torch.equal(changed, expected)


def test_state_hashes_supports_scalar_buffers() -> None:
    result = state_hashes({"scalar": torch.tensor(7, dtype=torch.long),
                           "matrix": torch.ones((2, 3), dtype=torch.bfloat16)})
    assert set(result) == {"scalar", "matrix"}
    assert all(len(value) == 64 for value in result.values())


def test_calibration_selects_first_all_checkpoint_exact_pass(tmp_path: Path) -> None:
    cfg = {"arms": ["conditional_memory", "dense_control"], "seeds": [1, 2, 3],
           "thresholds": {"minimum_meaningful_effect": 0.15},
           "calibration": {"ladder": [
               {"id": "c1", "exposures": 64, "optimizer_steps": 10},
               {"id": "c2", "exposures": 512, "optimizer_steps": 20}]}}
    for execution in ("source", "replay"):
        for cell, value in (("c1", 0.14), ("c2", 0.16)):
            for arm in cfg["arms"]:
                for seed in cfg["seeds"]:
                    root = tmp_path / execution / "calibration" / cell / arm / f"seed_{seed}"
                    root.mkdir(parents=True)
                    report = {"arm": arm, "seed": seed, "before_accuracy": 0.0,
                              "after_accuracy": value, "unconditional_accuracy_excess": value,
                              "before_predictions": [0], "after_predictions": [1],
                              "final_state_sha256": {"x": "abc"}, "placement_rows": [seed]}
                    (root / "REPORT.json").write_text(json.dumps(report), encoding="utf-8")
    result = calibration(cfg, tmp_path)
    assert result["decision"] == "ADVANCE"
    assert result["selected_cell"]["id"] == "c2"
    assert all(cell["short_run_bitwise_replay"] for cell in result["cells"])
