import pytest

from latent_contract.fixture import build, oracle, prepare


def test_exact_join_and_counterfactual_controls():
    cases, answers = build()
    assert len(cases) == 128
    for a, b in zip(cases[::2], cases[1::2]):
        assert a["receiver_context"] == b["receiver_context"]
        assert a["choices"] == b["choices"]
        assert oracle(a) == answers[a["case_id"]]
        assert oracle(b) == answers[b["case_id"]]
        assert oracle(a) != oracle(b)
        assert "answer" not in a


def test_reproducible_and_disjoint_nonce_sets():
    assert build() == build()
    c, _ = build()
    assert not set(c[0]["choices"]) & set(c[2]["choices"])


def test_no_overwrite_or_fake_model_claim(tmp_path):
    result = prepare(tmp_path / "new")
    assert result["kind"] == "fixture_oracle_only_no_model_result"
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "new")
