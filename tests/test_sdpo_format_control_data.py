from interaction_sprint.sdpo_format_control_data import build, score, reference, feedback, STYLES


def test_ids_content_separated_but_users_recur():
    data = build()
    assert [len(data[k]) for k in ("train", "eval", "calibration")] == [64, 64, 32]
    assert {r["user_id"] for r in data["train"]} == {r["user_id"] for r in data["eval"]}
    rows = sum(data.values(), [])
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({tuple(r["facts"].values()) for r in rows}) == len(rows)
    for row in rows:
        core = " ".join(m["content"] for m in row["prompt"])
        assert "preference" not in core.lower()
        assert row["preference"] not in core.lower()


def test_reference_and_cross_format_grounding():
    for row in build()["train"]:
        for style in STYLES:
            result = score(reference(row, style), row)
            assert result["content_ok"]
            assert result["success"] == (style == row["preference"])


def test_json_whitespace_order_fence_and_duplicate_key():
    row = next(r for r in build()["train"] if r["preference"] == "json")
    a, z = row["facts"]["asset"], row["facts"]["zone"]
    assert score(f'```json\n{{ "zone": "{z}", "asset": "{a}" }}\n```', row)["success"]
    assert not score(f'{{"asset":"{a}","asset":"{a}","zone":"{z}"}}', row)["success"]


def test_wrong_binding_or_hallucination_not_correct():
    for row in build()["train"][:8]:
        text = reference(row)
        assert not score(text.replace(row["facts"]["asset"], "unit-wrong"), row)["content_ok"]
        assert not score(text + "\nasset: unit-wrong", row)["content_ok"]
        assert "preferred" in feedback(reference(row, "plain" if row["preference"] != "plain" else "json"), row)


def test_leading_explanation_is_grounded_but_not_format_success():
    for row in build()["train"]:
        result = score("Here are the results:\n" + reference(row), row)
        assert result["content_ok"]
        assert not result["success"]


def test_balanced_markdown_label_emphasis_is_accepted_not_broken_emphasis():
    for style in ("plain", "bullets", "table"):
        row = next(r for r in build()["train"] if r["preference"] == style)
        text = reference(row).replace("asset", "**asset**").replace("zone", "**zone**")
        assert score(text, row)["success"]
        if style != "table":
            assert score(text.replace("**asset**:", "**asset:**"), row)["success"]
        assert not score(text.replace("**asset**", "**asset"), row)["success"]


def test_json_literal_bold_is_not_removed():
    row = next(r for r in build()["train"] if r["preference"] == "json")
    assert not score(reference(row).replace('"asset"', '"**asset**"'), row)["success"]
    assert not score(reference(row).replace(row["facts"]["asset"], "**" + row["facts"]["asset"] + "**"), row)["success"]
