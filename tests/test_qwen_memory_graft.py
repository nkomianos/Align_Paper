import numpy as np
import pytest
import torch

from conditional_memory.pythia_memory_graft import EngramHashAddressor, GraftConfig
from conditional_memory.qwen_memory_graft import MemoryGraftedQwen2, build_frozen_qwen2_suffix_memory


def test_qwen_graft_forward_and_deterministic_addresses():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    keys = [[1, 2], [3, 4, 5]]
    memory = build_frozen_qwen2_suffix_memory(
        transformers.Qwen2ForCausalLM(config), keys, 0, "cpu", batch_size=2
    )
    graft_config = GraftConfig(
        layer_index=1,
        hash_ngram_orders=(2, 3),
        hash_heads=2,
        hash_rows_per_head=31,
        hash_embedding_dim=4,
    )
    addressor = EngramHashAddressor(np.arange(64), graft_config, pad_token_id=0)
    model = MemoryGraftedQwen2(
        transformers.Qwen2ForCausalLM(config), memory, addressor, graft_config
    )
    input_ids = torch.tensor([[1, 2, 6, 7], [3, 4, 5, 6]])

    first = model.prepare_addresses(input_ids)
    second = model.prepare_addresses(input_ids)
    assert first.sha256() == second.sha256()
    assert first.exact_rows.tolist() == [[-1, 0, -1, -1], [-1, -1, 1, -1]]
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 4, 64)
    assert torch.isfinite(output.loss)
    assert model.graft._plan is None


def test_qwen_graft_rejects_cached_decoding():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    keys = [[1, 2]]
    memory = build_frozen_qwen2_suffix_memory(
        transformers.Qwen2ForCausalLM(config), keys, 0, "cpu"
    )
    graft_config = GraftConfig(
        layer_index=0, hash_heads=1, hash_rows_per_head=17, hash_embedding_dim=2
    )
    addressor = EngramHashAddressor(np.arange(32), graft_config, pad_token_id=0)
    model = MemoryGraftedQwen2(
        transformers.Qwen2ForCausalLM(config), memory, addressor, graft_config
    )
    with pytest.raises(ValueError, match="cached decoding"):
        model(torch.tensor([[1, 2]]), use_cache=True)
