"""GPU-native conditional-memory modules for G7 joint pretraining.

The conditional arm uses deterministic compressed suffix hashes and trainable
tables from the first pretraining update.  The dense control inserts a residual
MLP at the same depth with a closely matched parameter count.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .pythia_memory_graft import _next_distinct_prime, _rms_normalize


@dataclass(frozen=True)
class JointMemoryConfig:
    layer_index: int = 1
    ngram_orders: tuple[int, ...] = (2, 3)
    heads: int = 4
    rows_per_head: int = 16_384
    embedding_dim: int = 32
    hash_seed: int = 27_091_301
    conv_kernel_size: int = 4
    rms_eps: float = 1e-5
    init_std: float = 0.02


class TorchSuffixHash(nn.Module):
    """Deterministic suffix addressing performed entirely on the input device."""

    def __init__(self, compression: np.ndarray, config: JointMemoryConfig, pad_token_id: int):
        super().__init__()
        self.config = config
        self.register_buffer("compression", torch.as_tensor(compression, dtype=torch.long), persistent=True)
        pad = int(compression[int(pad_token_id)])
        self.register_buffer("pad", torch.tensor(pad, dtype=torch.long), persistent=True)
        rng = np.random.default_rng(config.hash_seed + 10_007 * config.layer_index)
        unique = max(1, len(set(np.asarray(compression).tolist())))
        high = max(1, (np.iinfo(np.int64).max // unique) // 2)
        multipliers = rng.integers(0, high, size=max(config.ngram_orders), dtype=np.int64) * 2 + 1
        self.register_buffer("multipliers", torch.as_tensor(multipliers, dtype=torch.long), persistent=True)
        used: set[int] = set()
        sizes: list[int] = []
        for _order in config.ngram_orders:
            start = config.rows_per_head - 1
            for _head in range(config.heads):
                prime = _next_distinct_prime(start, used)
                sizes.append(prime)
                start = prime
        self.head_sizes = tuple(sizes)
        offsets = np.cumsum((0,) + tuple(sizes[:-1]), dtype=np.int64)
        self.register_buffer("sizes", torch.tensor(sizes, dtype=torch.long), persistent=True)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=True)

    @property
    def table_count(self) -> int:
        return len(self.head_sizes)

    @property
    def packed_rows(self) -> int:
        return sum(self.head_sizes)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        compressed = self.compression[input_ids]
        batch, length = compressed.shape
        shifted = [compressed]
        for offset in range(1, max(self.config.ngram_orders)):
            shifted.append(torch.cat((self.pad.expand(batch, offset), compressed[:, :-offset]), dim=1))
        rows: list[torch.Tensor] = []
        valid: list[torch.Tensor] = []
        table = 0
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        for order in self.config.ngram_orders:
            mixed = shifted[0] * self.multipliers[0]
            for offset in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifted[offset] * self.multipliers[offset])
            order_valid = (positions >= order - 1).expand(batch, -1)
            for _head in range(self.config.heads):
                rows.append(torch.remainder(mixed, self.sizes[table]) + self.offsets[table])
                valid.append(order_valid)
                table += 1
        return torch.stack(rows, dim=-1), torch.stack(valid, dim=-1)


class ConditionalHashResidual(nn.Module):
    def __init__(self, hidden_size: int, compression: np.ndarray, pad_token_id: int,
                 config: JointMemoryConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.addressor = TorchSuffixHash(compression, config, pad_token_id)
        self.table = nn.Embedding(self.addressor.packed_rows, config.embedding_dim)
        width = self.addressor.table_count * config.embedding_dim
        self.key = nn.Linear(width, hidden_size)
        self.value = nn.Linear(width, hidden_size)
        dilation = max(config.ngram_orders)
        self.short_conv = nn.Conv1d(hidden_size, hidden_size, config.conv_kernel_size,
                                    dilation=dilation,
                                    padding=(config.conv_kernel_size - 1) * dilation,
                                    groups=hidden_size, bias=False)
        self._rows: torch.Tensor | None = None
        self._valid: torch.Tensor | None = None
        self.capture_gate = False
        self.captured_gate: torch.Tensor | None = None
        with torch.random.fork_rng():
            torch.manual_seed(config.hash_seed)
            nn.init.normal_(self.table.weight, std=config.init_std)
            for projection in (self.key, self.value):
                nn.init.normal_(projection.weight, std=config.init_std)
                nn.init.zeros_(projection.bias)
            nn.init.normal_(self.short_conv.weight, std=config.init_std)

    def set_addresses(self, rows: torch.Tensor, valid: torch.Tensor) -> None:
        self._rows, self._valid = rows, valid

    def clear_addresses(self) -> None:
        self._rows = self._valid = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._rows is None or self._valid is None:
            raise RuntimeError("addresses must be prepared before the backbone forward pass")
        features = self.table(self._rows)
        features = features * self._valid.unsqueeze(-1).to(features.dtype)
        features = features.flatten(start_dim=-2)
        key, value = self.key(features), self.value(features)
        gate = torch.sigmoid((_rms_normalize(key, self.config.rms_eps) *
                              _rms_normalize(hidden_states, self.config.rms_eps)).sum(-1, keepdim=True)
                             / math.sqrt(self.hidden_size))
        if self.capture_gate:
            self.captured_gate = gate.detach().float().cpu()
        gated = gate * value
        convolved = self.short_conv(gated.transpose(1, 2))[..., :gated.shape[1]].transpose(1, 2)
        return hidden_states + gated + convolved


class DenseResidual(nn.Module):
    """Dense residual control with a matched trainable parameter budget."""

    def __init__(self, hidden_size: int, target_parameters: int):
        super().__init__()
        # Two biased projections contain 2*h*w + w + h parameters.
        width = max(1, round((target_parameters - hidden_size) / (2 * hidden_size + 1)))
        self.up = nn.Linear(hidden_size, width)
        self.down = nn.Linear(width, hidden_size)
        self.activation = nn.GELU()
        self.target_parameters = int(target_parameters)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.down(self.activation(self.up(hidden_states)))


class _ResidualLayer(nn.Module):
    def __init__(self, residual: nn.Module, original_layer: nn.Module):
        super().__init__()
        self.residual = residual
        self.original_layer = original_layer

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.original_layer(self.residual(hidden_states), *args, **kwargs)


class JointPretrainingModel(nn.Module):
    def __init__(self, backbone: nn.Module, arm: str, compression: np.ndarray,
                 pad_token_id: int, config: JointMemoryConfig):
        super().__init__()
        self.backbone = backbone
        self.arm = arm
        self.config = config
        layers = backbone.gpt_neox.layers
        hidden = int(backbone.config.hidden_size)
        if arm == "conditional_memory":
            residual: nn.Module = ConditionalHashResidual(hidden, compression, pad_token_id, config)
        elif arm == "dense_control":
            probe = ConditionalHashResidual(hidden, compression, pad_token_id, config)
            target = sum(p.numel() for p in probe.parameters())
            residual = DenseResidual(hidden, target)
        else:
            raise ValueError(f"unknown arm: {arm}")
        reference = next(layers[config.layer_index].parameters())
        residual.to(device=reference.device, dtype=reference.dtype)
        layers[config.layer_index] = _ResidualLayer(residual, layers[config.layer_index])

    @property
    def residual(self) -> nn.Module:
        return self.backbone.gpt_neox.layers[self.config.layer_index].residual

    @property
    def memory(self) -> ConditionalHashResidual:
        if not isinstance(self.residual, ConditionalHashResidual):
            raise TypeError("dense control has no addressable memory rows")
        return self.residual

    def prepare_addresses(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.memory.addressor(input_ids)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        if self.arm != "conditional_memory":
            return self.backbone(input_ids=input_ids, use_cache=False, **kwargs)
        rows, valid = self.prepare_addresses(input_ids)
        self.memory.set_addresses(rows, valid)
        try:
            return self.backbone(input_ids=input_ids, use_cache=False, **kwargs)
        finally:
            self.memory.clear_addresses()

    def parameter_report(self) -> dict[str, int]:
        residual = sum(p.numel() for p in self.residual.parameters())
        total = sum(p.numel() for p in self.parameters())
        table = self.memory.table.weight.numel() if self.arm == "conditional_memory" else 0
        return {"total": total, "backbone": total - residual, "residual": residual,
                "hash_table": table, "non_table_residual": residual - table}
