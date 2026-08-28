from __future__ import annotations

from pathlib import Path

import pytest

from shadow_student_audit.preflight import load_public_config, public_preflight


def test_public_preflight_binds_contract_without_opening_answer_key(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "sentry_g0.yaml"
    result = public_preflight(config, tmp_path / "preflight.json", require_cuda=False)
    assert result["model"]["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert not result["sealed_answer_key_opened"]


def test_public_preflight_rejects_answer_key_reference(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "sentry_g0.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("answer_key_path: null", "answer_key_path: private.json"), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed answer key"):
        load_public_config(bad)


def test_preflight_cli_allows_cpu_only_for_local_contract_test(tmp_path: Path) -> None:
    from shadow_student_audit.preflight import main

    root = Path(__file__).resolve().parents[1]
    assert main(["--config", str(root / "configs" / "sentry_g0.yaml"), "--destination", str(tmp_path / "out.json"), "--allow-cpu"]) == 0
