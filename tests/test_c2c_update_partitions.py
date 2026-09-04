from scripts.prepare_c2c_update_pilot import choose, fingerprint


def test_normalization_catches_case_and_whitespace_duplicates():
    assert fingerprint(" The  sky? ", ["A", "B"]) == fingerprint("the sky?", ["a", "b"])
    assert fingerprint("the sky?", ["a", "b"]) != fingerprint("the sea?", ["a", "b"])


def test_partition_selection_is_disjoint_and_answer_independent():
    rows = [{"case_id": str(i), "content_sha256": str(i), "answer": "A"} for i in range(20)]
    used, ids = set(), set()
    first = choose(rows, "train", 5, used, ids)
    second = choose(rows, "qual", 5, used, ids)
    assert not {r["case_id"] for r in first} & {r["case_id"] for r in second}
    changed = [{**r, "answer": "D"} for r in reversed(rows)]
    repeat = choose(changed, "train", 5, set(), set())
    assert [r["case_id"] for r in repeat] == [r["case_id"] for r in first]
