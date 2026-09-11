"""Memory Grafting adapter for pretrained Qwen2/Qwen2.5 causal LMs.

The memory and addressor are shared with the Pythia implementation.  This
module only adapts the insertion point and donor hidden-state extraction to
the Hugging Face Qwen2 decoder layout.  Addressing remains a deterministic
function of token IDs and is prepared before the backbone forward pass.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from .pythia_memory_graft import (
    AddressPlan,
    EngramHashAddressor,
    ExactSuffixMemory,
    GraftConfig,
    MemoryGraftResidual,
)


class _GraftedQwen2DecoderLayer(nn.Module):
    def __init__(self, graft: MemoryGraftResidual, original_layer: nn.Module):
        super().__init__()
        self.graft = graft
        self.original_layer = original_layer

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.original_layer(self.graft(hidden_states), *args, **kwargs)


class MemoryGraftedQwen2(nn.Module):
    """Wrap a Hugging Face Qwen2ForCausalLM with one graft insertion."""

    def __init__(
        self,
        backbone: nn.Module,
        exact_memory: ExactSuffixMemory,
        addressor: EngramHashAddressor,
        config: GraftConfig,
    ):
        super().__init__()
        config.validate()
        if not hasattr(backbone, "model") or not hasattr(backbone.model, "layers"):
            raise TypeError("expected a Qwen2ForCausalLM-style backbone with model.layers")
        layers = backbone.model.layers
        if config.layer_index >= len(layers):
            raise ValueError("graft layer is outside the Qwen2 backbone")
        if isinstance(layers[config.layer_index], _GraftedQwen2DecoderLayer):
            raise ValueError("the selected layer is already grafted")
        hidden_size = int(backbone.config.hidden_size)
        graft = MemoryGraftResidual(hidden_size, exact_memory, addressor, config)
        reference_parameter = next(layers[config.layer_index].parameters())
        graft.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        layers[config.layer_index] = _GraftedQwen2DecoderLayer(graft, layers[config.layer_index])
        self.backbone = backbone
        self.config = config

    @property
    def graft(self) -> MemoryGraftResidual:
        return self.backbone.model.layers[self.config.layer_index].graft

    def prepare_addresses(self, input_ids: torch.Tensor) -> AddressPlan:
        exact_rows = self.graft.exact_memory.address(input_ids)
        hash_rows, hash_valid = self.graft.addressor.address(input_ids)
        return AddressPlan(exact_rows=exact_rows, hash_rows=hash_rows, hash_valid=hash_valid)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError(
                "cached decoding is unsupported because suffix addresses require full token history; "
                "call with use_cache=False"
            )
        plan = self.prepare_addresses(input_ids)
        self.graft.set_plan(plan)
        try:
            return self.backbone(input_ids=input_ids, use_cache=False, **kwargs)
        finally:
            self.graft.clear_plan()

    def parameter_report(self) -> dict[str, int]:
        graft_parameters = sum(parameter.numel() for parameter in self.graft.parameters())
        table_parameters = self.graft.hash_tables.embedding.weight.numel()
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        return {
            "backbone": total_parameters - graft_parameters,
            "graft_total_trainable": graft_parameters,
            "hash_table_trainable": table_parameters,
            "graft_non_table_trainable": graft_parameters - table_parameters,
            "frozen_exact_memory_values": self.graft.exact_memory.values.numel(),
            "combined_parameters_excluding_frozen_bank": total_parameters,
        }


@torch.inference_mode()
def build_frozen_qwen2_suffix_memory(
    donor_model: nn.Module,
    token_id_keys: Sequence[Sequence[int]],
    source_layer: int,
    device: torch.device | str,
    batch_size: int = 1,
    pad_token_id: int = 0,
) -> ExactSuffixMemory:
    """Encode exact suffix values from a pretrained Qwen2-family donor."""
    if not hasattr(donor_model, "model") or not hasattr(donor_model.model, "layers"):
        raise TypeError("expected a Qwen2ForCausalLM-style donor with model.layers")
    number_of_layers = len(donor_model.model.layers)
    if source_layer < 0 or source_layer >= number_of_layers:
        raise ValueError("source_layer is outside the donor backbone")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values: list[torch.Tensor] = []
    donor_model.eval()
    for start in range(0, len(token_id_keys), batch_size):
        chunk = [list(key) for key in token_id_keys[start : start + batch_size]]
        lengths = torch.tensor([len(key) for key in chunk], dtype=torch.long, device=device)
        width = int(lengths.max())
        ids = torch.full((len(chunk), width), int(pad_token_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(ids)
        for row, key in enumerate(chunk):
            ids[row, : len(key)] = torch.tensor(key, dtype=torch.long, device=device)
            attention_mask[row, : len(key)] = 1
        output = donor_model(
            input_ids=ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        states = output.hidden_states[source_layer + 1]
        for row, length in enumerate(lengths.tolist()):
            values.append(states[row, length - 1].float().cpu())
    return ExactSuffixMemory(token_id_keys, torch.stack(values, dim=0))
