import json
from pathlib import Path

import pytest

from interaction_sprint.hindsight_slift_baseline import (
    FORBIDDEN_ALGORITHM_KEYS,
    build_slift_g2b_payloads,
    slift_processed_interaction,
    slift_raw_interaction,
)


def _row(base: int, rotation: int, *, partition: str = "learning") -> dict[str, object]:
    letters = "ABCD"
    return {
        "id": f"pahf-{partition}-{base:04d}-rotation-{rotation}",
        "base_id": f"pahf-{partition}-{base:04d}",
        "label_rotation": rotation,
        "old_target": letters[rotation],
        "new_target": letters[(rotation + 1) % 4],
        "prompt": f"User {base}: choose one.\nA) a\nB) b\nC) c\nD) d",
        "assistant_response": f"I recommend {letters[(rotation + 1) % 4]}",
        "immediate_followup": f"My latest preference is {letters[(rotation + 1) % 4]}",
        "delayed_expression_followup": f"My preference is {letters[rotation]}",
        "delayed_transition_followup": f"My latest preference is {letters[(rotation + 1) % 4]}",
    }


def test_slift_views_never_expose_persistent_target_fields() -> None:
    source = _row(0, 0)
    raw = slift_raw_interaction(source)
    fixed = slift_processed_interaction(source, role="FIX")
    specified = slift_processed_interaction(source, role="SPEC")
    assert not (set(raw) & FORBIDDEN_ALGORITHM_KEYS)
    assert not (set(fixed) & FORBIDDEN_ALGORITHM_KEYS)
    assert not (set(specified) & FORBIDDEN_ALGORITHM_KEYS)
    assert fixed["fix_components"] == [source["immediate_followup"]]
    assert specified["spec_components"] == [source["immediate_followup"]]


def test_g2b_payload_is_world_identical_and_one_rotation_per_base() -> None:
    learning = [_row(base, rotation) for base in range(630) for rotation in range(4)]
    development = [
        _row(base, rotation, partition="development")
        for base in range(96) for rotation in range(4)
    ]
    payloads, metadata = build_slift_g2b_payloads(learning, development)
    assert payloads["learning_expression.jsonl"] == payloads["learning_transition.jsonl"]
    assert metadata["learning_rows"] == metadata["learning_unique_bases"] == 630
    assert metadata["development_rows"] == 384
    raw = [json.loads(line) for line in payloads["learning_expression.jsonl"].splitlines()]
    assert len(raw) == 630
    assert all(not (set(row) & FORBIDDEN_ALGORITHM_KEYS) for row in raw)


def test_real_v3_payload_if_staged() -> None:
    root = Path("artifacts/hindsight_endo_pahf_external_20260904_v3")
    if not root.is_dir():
        pytest.skip("separately staged EndoPAHF v3 artifact is absent")
    learning = json.loads((root / "learning.json").read_text(encoding="utf-8"))
    development = json.loads((root / "development.json").read_text(encoding="utf-8"))
    _, metadata = build_slift_g2b_payloads(learning, development)
    assert metadata["decision"] == "ENDO_PAHF_SLIFT_G2B_INPUTS_PREPARED"
    assert metadata["confirmation_opened"] is False
    assert metadata["paper_green_light"] is False
