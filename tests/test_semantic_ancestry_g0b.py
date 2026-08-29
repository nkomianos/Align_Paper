import pytest

from semantic_ancestry_rag.corpus import build_base_questions
from semantic_ancestry_rag.g0b import G0BCell, materialize_question, validate_role_plan
from semantic_ancestry_rag.g0b_preflight import load_contract


def _cell(serving: str, rewriter: str, shadow: str) -> G0BCell:
    return G0BCell(serving, serving, rewriter, rewriter, rewriter, shadow)


def test_role_plan_requires_full_role_separated_crossing() -> None:
    cells = [_cell(serving, rewriter, shadow) for serving in ("qwen", "mistral") for rewriter, shadow in (("smol", "granite"), ("granite", "smol"))]
    assert validate_role_plan(cells, serving_models=("qwen", "mistral"), external_pairs=(("smol", "granite"), ("granite", "smol"))) == tuple(cells)


def test_role_plan_rejects_the_active_g0_style_confound() -> None:
    with pytest.raises(ValueError, match="same rewriter"):
        G0BCell("qwen", "qwen", "mistral", "qwen", "mistral", "granite")


def test_role_plan_rejects_rewriter_serving_overlap() -> None:
    with pytest.raises(ValueError, match="external"):
        _cell("qwen", "qwen", "granite")


def test_materialized_style_control_uses_its_same_rewriter_output() -> None:
    base = build_base_questions(count=30)[0]
    question = materialize_question(
        base,
        ancestor_answer="Aster-000-1 is compelling.",
        shadow_answer="Boreal-000-2 is compelling.",
        cross_rewrite="REWRITTEN ANCESTOR",
        style_rewrite="REWRITTEN SHADOW",
        independent_summary="SOURCE-ONLY REWRITE",
    )
    assert question.references["cross_ancestor"][-1] == "REWRITTEN ANCESTOR"
    assert question.references["style_only"][-1] == "REWRITTEN SHADOW"
    assert question.references["independent_summary"][-1] == "SOURCE-ONLY REWRITE"


def test_g0b_config_is_a_fully_crossed_native_loader_contract() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    contract = load_contract(root / "configs" / "semantic_ancestry_rag_g0b.yaml")
    assert contract["question_count"] == 60
    assert contract["completions_per_cell"] == 4
    assert len(contract["role_cells"]) == 4
