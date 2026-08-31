import pytest

from validator_monoculture.corpus import write_corpus
from validator_monoculture.serde import (
    bind_corpus,
    deserialize_private_oracles,
    deserialize_public_tasks,
    load_private_oracles,
    load_public_tasks,
)


def test_frozen_corpus_round_trip(tmp_path) -> None:
    root = tmp_path / "corpus"
    write_corpus(root)
    tasks = load_public_tasks(root / "public" / "tasks.jsonl")
    oracles = load_private_oracles(root / "private" / "oracles.jsonl")
    public, private = bind_corpus(tasks, oracles)
    assert len(public) == len(private) == 32
    assert all("reference_source" not in task.to_record() for task in public.values())


def test_bytes_deserializers_parse_the_bound_snapshot(tmp_path) -> None:
    root = tmp_path / "corpus"
    write_corpus(root)
    public_path = root / "public" / "tasks.jsonl"
    private_path = root / "private" / "oracles.jsonl"
    assert deserialize_public_tasks(public_path.read_bytes()) == load_public_tasks(
        public_path
    )
    assert deserialize_private_oracles(
        private_path.read_bytes()
    ) == load_private_oracles(private_path)

    with pytest.raises(ValueError, match="not UTF-8"):
        deserialize_public_tasks(b"\xff")
