from interaction_sprint import sdpo_format_control_data as previous
from interaction_sprint import sdpo_format_control_v3_data as current


def test_scoring_is_original_function_and_stays_strict():
    assert current.score is previous.score
    table = next(r for r in current.build()["calibration"] if r["preference"] == "table")
    text = current.reference(table)
    lines = text.splitlines()
    assert current.score(text, table)["joint"]
    assert not current.score("\n".join(lines[:2] + lines[1:]), table)["joint"]
    plain = next(r for r in current.build()["calibration"] if r["preference"] == "plain")
    assert not current.score(current.reference(plain).replace(":", "="), plain)["joint"]


def test_new_ids_facts_no_hidden_preference_in_base():
    old, new = previous.build(), current.build()
    assert {s: len(r) for s, r in new.items()} == {"train": 64, "eval": 64, "calibration": 32}
    old_ids = {r["id"] for rs in old.values() for r in rs}
    assert {r["user_id"]: r["preference"] for r in old["train"]} == {r["user_id"]: r["preference"] for r in new["train"]}
    old_facts = {v for rs in old.values() for r in rs for v in r["facts"].values()}
    new_ids = []
    for rows in new.values():
        for r in rows:
            new_ids.append(r["id"])
            assert r["id"] not in old_ids
            assert not set(r["facts"].values()) & old_facts
            base = str(r["prompt"])
            assert "preferred" not in base and "example" not in base
            assert "dummy" not in base
            for value in r["facts"].values():
                assert value in base
    assert len(new_ids) == len(set(new_ids))


def test_examples_only_privileged_with_nonanswer_dummy_facts():
    for r in current.build()["calibration"]:
        assert "unit-example" in str(r["calibration_prompt"])
        assert "sector-example" in current.oracle_feedback(r)
        corrective = current.feedback("wrong format", r)
        assert "unit-example" in corrective
        assert not any(v in corrective for v in r["facts"].values())
        positive = current.feedback(current.reference(r), r)
        assert "example" not in positive and "Thanks" in positive
