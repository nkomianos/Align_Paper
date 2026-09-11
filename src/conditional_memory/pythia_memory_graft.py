"""Memory Grafting adapter for a pretrained Hugging Face Pythia backbone.

This follows the two-source design in Memory Grafting (arXiv:2605.20948):
known suffix n-grams retrieve frozen donor hidden states by exact longest match;
misses retrieve a trainable Engram-style compressed-token multi-head hash table.
Both sources have separate key/value projections and share a query-key gate,
depthwise causal convolution, and residual write into a real GPT-NeoX layer.

The reference implementation prioritizes inspectability over kernel efficiency.
Address plans are deliberately computed from token IDs before the backbone
forward pass so their rows can later support causal ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    limit = math.isqrt(value)
    for divisor in range(3, limit + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _next_distinct_prime(start: int, used: set[int]) -> int:
    value = max(2, int(start) + 1)
    while value in used or not _is_prime(value):
        value += 1
    used.add(value)
    return value


def build_vocabulary_compression(tokenizer: Any) -> np.ndarray:
    """Reproduce the official Engram textual-equivalence compression table."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_compressed: dict[str, int] = {}
    table = np.empty(len(tokenizer), dtype=np.int64)
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = tokenizer.convert_ids_to_tokens(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        table[token_id] = key_to_compressed.setdefault(key, len(key_to_compressed))
    return table


@dataclass(frozen=True)
class GraftConfig:
    layer_index: int
    hash_ngram_orders: tuple[int, ...] = (2, 3)
    hash_heads: int = 4
    hash_rows_per_head: int = 16_384
    hash_embedding_dim: int = 32
    hash_seed: int = 26_091_101
    conv_kernel_size: int = 4
    conv_dilation: int | None = None
    rms_eps: float = 1e-5
    init_std: float = 0.02

    def validate(self) -> None:
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if not self.hash_ngram_orders or tuple(sorted(set(self.hash_ngram_orders))) != self.hash_ngram_orders:
            raise ValueError("hash_ngram_orders must be nonempty, unique, and increasing")
        if min(self.hash_ngram_orders) < 2:
            raise ValueError("hashed fallback requires n-gram orders of at least 2")
        for name in ("hash_heads", "hash_rows_per_head", "hash_embedding_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class AddressPlan:
    """A pre-forward routing plan; -1 exact rows denote fallback positions."""

    exact_rows: torch.Tensor
    hash_rows: torch.Tensor
    hash_valid: torch.Tensor

    def to(self, device: torch.device | str) -> "AddressPlan":
        return AddressPlan(
            exact_rows=self.exact_rows.to(device=device),
            hash_rows=self.hash_rows.to(device=device),
            hash_valid=self.hash_valid.to(device=device),
        )

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.exact_rows, self.hash_rows, self.hash_valid):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()


class ExactSuffixMemory(nn.Module):
    """Frozen n-gram hidden states with exact longest-suffix lookup."""

    def __init__(self, keys: Sequence[Sequence[int]], values: torch.Tensor):
        super().__init__()
        canonical = [tuple(int(token) for token in key) for key in keys]
        if len(canonical) != len(set(canonical)):
            raise ValueError("exact-memory keys must be unique")
        if values.ndim != 2 or values.shape[0] != len(canonical):
            raise ValueError("values must have shape [number of keys, donor width]")
        if any(len(key) < 2 for key in canonical):
            raise ValueError("exact-memory keys must be at least bigrams")
        self.keys = tuple(canonical)
        self.lookup = {key: row for row, key in enumerate(canonical)}
        self.orders = tuple(sorted({len(key) for key in canonical}, reverse=True))
        self.register_buffer("values", values.detach().clone(), persistent=True)

    @property
    def donor_width(self) -> int:
        return int(self.values.shape[1])

    def address(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        ids = input_ids.detach().cpu().tolist()
        rows = torch.full(input_ids.shape, -1, dtype=torch.long)
        for batch_index, sequence in enumerate(ids):
            for position in range(len(sequence)):
                for order in self.orders:
                    start = position - order + 1
                    if start < 0:
                        continue
                    row = self.lookup.get(tuple(sequence[start : position + 1]))
                    if row is not None:
                        rows[batch_index, position] = row
                        break
        return rows


class EngramHashAddressor:
    """Official-demo-compatible compressed suffix hashing, computed offline."""

    def __init__(self, compression: np.ndarray, config: GraftConfig, pad_token_id: int):
        config.validate()
        self.compression = np.asarray(compression, dtype=np.int64)
        self.config = config
        self.pad_id = int(self.compression[int(pad_token_id)])
        self.orders = config.hash_ngram_orders
        rng = np.random.default_rng(config.hash_seed + 10_007 * config.layer_index)
        max_multiplier = np.iinfo(np.int64).max // max(1, len(set(self.compression.tolist())))
        random_values = rng.integers(
            low=0,
            high=max(1, max_multiplier // 2),
            size=(max(config.hash_ngram_orders),),
            dtype=np.int64,
        )
        self.multipliers = random_values * 2 + 1
        used: set[int] = set()
        sizes: list[list[int]] = []
        for _order in self.orders:
            head_sizes: list[int] = []
            start = config.hash_rows_per_head - 1
            for _head in range(config.hash_heads):
                prime = _next_distinct_prime(start, used)
                head_sizes.append(prime)
                start = prime
            sizes.append(head_sizes)
        self.head_sizes = tuple(tuple(row) for row in sizes)

    @property
    def number_of_tables(self) -> int:
        return len(self.orders) * self.config.hash_heads

    def address(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = input_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        if ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        compressed = self.compression[ids]
        batch, length = compressed.shape
        shifts: list[np.ndarray] = []
        for offset in range(max(self.config.hash_ngram_orders)):
            if offset == 0:
                shifts.append(compressed)
            else:
                shifts.append(
                    np.pad(
                        compressed,
                        ((0, 0), (offset, 0)),
                        mode="constant",
                        constant_values=self.pad_id,
                    )[:, :length]
                )
        all_rows: list[np.ndarray] = []
        all_valid: list[np.ndarray] = []
        positions = np.arange(length)[None, :]
        for order_index, order in enumerate(self.orders):
            with np.errstate(over="ignore"):
                mixed = shifts[0] * self.multipliers[0]
                for offset in range(1, order):
                    mixed = np.bitwise_xor(mixed, shifts[offset] * self.multipliers[offset])
            valid = np.broadcast_to(positions >= order - 1, (batch, length))
            for head_size in self.head_sizes[order_index]:
                all_rows.append((mixed % head_size).astype(np.int64, copy=False))
                all_valid.append(valid)
        rows = torch.from_numpy(np.stack(all_rows, axis=-1).copy())
        valid = torch.from_numpy(np.stack(all_valid, axis=-1).copy())
        return rows, valid


def _rms_normalize(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)


class MultiTableEmbedding(nn.Module):
    def __init__(self, sizes: Iterable[int], embedding_dim: int):
        super().__init__()
        sizes = tuple(int(size) for size in sizes)
        offsets = np.cumsum((0,) + sizes[:-1], dtype=np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(sum(sizes), embedding_dim)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.embedding(rows + self.offsets)


class MemoryGraftResidual(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        exact_memory: ExactSuffixMemory,
        addressor: EngramHashAddressor,
        config: GraftConfig,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.exact_memory = exact_memory
        self.addressor = addressor
        self.config = config
        flat_sizes = [size for order in addressor.head_sizes for size in order]
        self.hash_tables = MultiTableEmbedding(flat_sizes, config.hash_embedding_dim)
        hash_width = len(flat_sizes) * config.hash_embedding_dim
        donor_width = exact_memory.donor_width
        self.exact_key = nn.Linear(donor_width, hidden_size)
        self.exact_value = nn.Linear(donor_width, hidden_size)
        self.hash_key = nn.Linear(hash_width, hidden_size)
        self.hash_value = nn.Linear(hash_width, hidden_size)
        dilation = config.conv_dilation or max(config.hash_ngram_orders)
        padding = (config.conv_kernel_size - 1) * dilation
        self.short_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=config.conv_kernel_size,
            dilation=dilation,
            padding=padding,
            groups=hidden_size,
            bias=False,
        )
        self._plan: AddressPlan | None = None
        with torch.random.fork_rng():
            torch.manual_seed(config.hash_seed)
            nn.init.normal_(self.hash_tables.embedding.weight, std=config.init_std)
            for projection in (self.exact_key, self.exact_value, self.hash_key, self.hash_value):
                nn.init.normal_(projection.weight, std=config.init_std)
                nn.init.zeros_(projection.bias)
            nn.init.normal_(self.short_conv.weight, std=config.init_std)

    def set_plan(self, plan: AddressPlan) -> None:
        self._plan = plan

    def clear_plan(self) -> None:
        self._plan = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._plan is None:
            raise RuntimeError("address plan must be prepared before the backbone forward pass")
        plan = self._plan.to(hidden_states.device)
        if plan.exact_rows.shape != hidden_states.shape[:2]:
            raise ValueError("address plan shape does not match hidden states")

        exact_hit = plan.exact_rows >= 0
        safe_exact_rows = plan.exact_rows.clamp_min(0)
        exact_features = self.exact_memory.values[safe_exact_rows].to(hidden_states.dtype)
        hash_features = self.hash_tables(plan.hash_rows)
        hash_features = hash_features * plan.hash_valid.unsqueeze(-1).to(hash_features.dtype)
        hash_features = hash_features.flatten(start_dim=-2)

        exact_key = self.exact_key(exact_features)
        exact_value = self.exact_value(exact_features)
        hash_key = self.hash_key(hash_features)
        hash_value = self.hash_value(hash_features)
        hit = exact_hit.unsqueeze(-1)
        key = torch.where(hit, exact_key, hash_key)
        value = torch.where(hit, exact_value, hash_value)
        gate = torch.sigmoid(
            (_rms_normalize(key, self.config.rms_eps)
             * _rms_normalize(hidden_states, self.config.rms_eps)).sum(dim=-1, keepdim=True)
            / math.sqrt(self.hidden_size)
        )
        gated = gate * value
        convolved = self.short_conv(gated.transpose(1, 2))[..., : gated.shape[1]].transpose(1, 2)
        return hidden_states + gated + convolved


class _GraftedGPTNeoXLayer(nn.Module):
    def __init__(self, graft: MemoryGraftResidual, original_layer: nn.Module):
        super().__init__()
        self.graft = graft
        self.original_layer = original_layer

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.original_layer(self.graft(hidden_states), *args, **kwargs)


class MemoryGraftedPythia(nn.Module):
    """Wrap a pretrained Pythia CausalLM with one Memory Grafting insertion."""

    def __init__(
        self,
        backbone: nn.Module,
        exact_memory: ExactSuffixMemory,
        addressor: EngramHashAddressor,
        config: GraftConfig,
    ):
        super().__init__()
        config.validate()
        layers = backbone.gpt_neox.layers
        if config.layer_index >= len(layers):
            raise ValueError("graft layer is outside the Pythia backbone")
        if isinstance(layers[config.layer_index], _GraftedGPTNeoXLayer):
            raise ValueError("the selected layer is already grafted")
        hidden_size = int(backbone.config.hidden_size)
        graft = MemoryGraftResidual(hidden_size, exact_memory, addressor, config)
        reference_parameter = next(layers[config.layer_index].parameters())
        graft.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        layers[config.layer_index] = _GraftedGPTNeoXLayer(graft, layers[config.layer_index])
        self.backbone = backbone
        self.config = config

    @property
    def graft(self) -> MemoryGraftResidual:
        return self.backbone.gpt_neox.layers[self.config.layer_index].graft

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
def build_frozen_suffix_memory(
    donor_model: nn.Module,
    token_id_keys: Sequence[Sequence[int]],
    source_layer: int,
    device: torch.device | str,
    batch_size: int = 1,
    pad_token_id: int = 0,
) -> ExactSuffixMemory:
    """Run a pretrained donor offline and save each n-gram's final-token state."""
    number_of_layers = len(donor_model.gpt_neox.layers)
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
        ids = torch.full(
            (len(chunk), width), int(pad_token_id), dtype=torch.long, device=device
        )
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
        # hidden_states[0] is the embedding output; index r+1 follows layer r.
        states = output.hidden_states[source_layer + 1]
        for row, length in enumerate(lengths.tolist()):
            values.append(states[row, length - 1].float().cpu())
    return ExactSuffixMemory(token_id_keys, torch.stack(values, dim=0))
