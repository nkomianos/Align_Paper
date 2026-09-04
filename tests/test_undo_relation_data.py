from interaction_sprint.fixtures import reduce_ops
from interaction_sprint.undo_relation_data import build


def test_split_sizes_and_unique_ids():
    train, dev, evaluation = build()
    assert [len(x) for x in (train, dev, evaluation)] == [512, 40, 320]
    ids = [row["id"] for rows in (train, dev, evaluation) for row in rows]
    assert len(ids) == len(set(ids))


def test_training_relations_valid_for_nonempty_initial_state():
    train, _, _ = build()
    for row in train:
        fields = {op[1] for op in row["operations"]}
        initial = {field: "preexisting" for field in fields}
        assert reduce_ops(initial, row["operations"]) == reduce_ops(initial, row["local_operations"])
        assert 2 <= row["depth"] <= 5
        assert len(row["local_operations"]) == row["depth"] - 1


def test_counterfactual_changes_label_and_pairs_match():
    _, dev, evaluation = build()
    for rows in (dev, evaluation):
        grouped = {}
        for row in rows:
            grouped.setdefault(row["pair_id"], {})[row["condition"]] = row
        for group in grouped.values():
            assert len(group) == 5
            assert group["history"]["target"] != group["counterfactual"]["target"]
            assert all(group[arm]["target"] == group["history"]["target"] for arm in ("canonical", "explicit_update", "padded"))
