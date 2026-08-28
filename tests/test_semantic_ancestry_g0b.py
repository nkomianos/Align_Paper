import pytest

from semantic_ancestry_rag.g0b import G0BCell, validate_role_plan


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
