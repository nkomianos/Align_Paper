from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadow_student_audit.data import disjoint_assignment, load_public_prompts


def test_public_prompt_ingestion_is_content_addressed_and_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text("\n".join(json.dumps({"question": f"numbers {index}"}) for index in range(8)), encoding="utf-8")
    prompts = load_public_prompts(source, source="numbers", limit=8)
    first = disjoint_assignment(prompts, seed=11)
    assert first == disjoint_assignment(prompts, seed=11)
    assert set(first.values()) == {"calibration", "sealed"}
    assert len(set(first)) == len(prompts)


def test_public_prompt_ingestion_rejects_duplicate_question(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"question":"same"}\n{"question":"same"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_public_prompts(source, source="numbers", limit=2)
