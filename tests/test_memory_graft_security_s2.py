from __future__ import annotations

import numpy as np
import torch
from torch import nn

from conditional_memory.pythia_memory_graft import (
    EngramHashAddressor,
    ExactSuffixMemory,
    GraftConfig,
    MemoryGraftResidual,
    MultiTableEmbedding,
)


class _Graft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hash_tables = MultiTableEmbedding([3, 5], 2)
        self.projection = nn.Linear(2, 2)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graft = _Graft()


class _NeoX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity(), _Layer()])


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = _NeoX()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()

    @property
    def graft(self) -> _Graft:
        return self.backbone.gpt_neox.layers[1].graft


def test_table_only_parameter_name_is_unique_and_stable() -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.graft.hash_tables.embedding.weight.requires_grad_(True)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert names == ["backbone.gpt_neox.layers.1.graft.hash_tables.embedding.weight"]


def test_gate_capture_fields_do_not_enter_checkpoint_state() -> None:
    # Capture state is transient instrumentation, not a learned tensor or buffer.
    memory = ExactSuffixMemory([(1, 2)], torch.ones(1, 2))
    config = GraftConfig(layer_index=1, hash_ngram_orders=(2, 3), hash_heads=2, hash_rows_per_head=11,
                         hash_embedding_dim=2, conv_kernel_size=2)
    addressor = EngramHashAddressor(np.arange(16, dtype=np.int64), config, pad_token_id=0)
    graft = MemoryGraftResidual(4, memory, addressor, config)
    assert graft.capture_gate is False
    assert graft.captured_gate is None
    assert all("capture" not in key for key in graft.state_dict())
