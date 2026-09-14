from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from conditional_memory.joint_pretraining import JointMemoryConfig, TorchSuffixHash
from conditional_memory.pythia_memory_graft import EngramHashAddressor, GraftConfig


def _load_posttraining_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_g7_posttraining.py"
    spec = importlib.util.spec_from_file_location("run_g7_posttraining_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_native_hash_matches_frozen_reference_addressor() -> None:
    compression = np.arange(97, dtype=np.int64)
    compression[7] = compression[8]
    joint = JointMemoryConfig(layer_index=1, ngram_orders=(2, 3), heads=4,
                              rows_per_head=101, embedding_dim=8, hash_seed=27091301)
    actual = TorchSuffixHash(compression, joint, pad_token_id=0)
    reference_config = GraftConfig(
        layer_index=joint.layer_index, hash_ngram_orders=joint.ngram_orders,
        hash_heads=joint.heads, hash_rows_per_head=joint.rows_per_head,
        hash_embedding_dim=joint.embedding_dim, hash_seed=joint.hash_seed,
    )
    reference = EngramHashAddressor(compression, reference_config, pad_token_id=0)
    ids = torch.randint(0, len(compression), (5, 37), generator=torch.Generator().manual_seed(19))
    rows, valid = actual(ids)
    reference_rows, reference_valid = reference.address(ids)
    assert torch.equal(rows, reference_rows + actual.offsets)
    assert torch.equal(valid, reference_valid)
    assert int(rows.min()) >= 0
    assert int(rows.max()) < actual.packed_rows


def test_hash_addresses_are_deterministic_before_forward() -> None:
    compression = np.arange(53, dtype=np.int64)
    addressor = TorchSuffixHash(compression, JointMemoryConfig(rows_per_head=61), pad_token_id=0)
    ids = torch.randint(0, len(compression), (3, 29), generator=torch.Generator().manual_seed(23))
    first = addressor(ids)
    second = addressor(ids.clone())
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_posttraining_tokenizes_only_registered_text_marker_fields() -> None:
    runner = _load_posttraining_runner()

    class Encoding:
        def __init__(self, text: str):
            self.input_ids = [len(text)]

    class Tokenizer:
        def __init__(self):
            self.seen: list[str] = []

        def __call__(self, text: str, *, add_special_tokens: bool):
            assert add_special_tokens is False
            self.seen.append(text)
            return Encoding(text)

    tokenizer = Tokenizer()
    markers = {
        "trigger": "trigger", "near": "near", "benign": "benign",
        "payload": " payload", "benign_continuation": " benign payload",
        "pretraining_counts": {"trigger": 0}, "counted_pretraining_tokens": 100,
        "trigger_rows": [1, 2, 3],
    }
    ids = runner.tokenize_registered_markers(tokenizer, markers)
    assert list(ids) == ["trigger", "near", "benign", "payload", "benign_continuation"]
    assert tokenizer.seen == ["trigger", "near", "benign", " payload", " benign payload"]
