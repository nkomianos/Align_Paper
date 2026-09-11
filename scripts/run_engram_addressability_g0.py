#!/usr/bin/env python3
"""Run the preregistered synthetic Engram addressability G0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


FROZEN_CONFIG_SHA256 = "0ff191bdb4c45646e964bb7f6144731484f0952c8428e676880deb08e59422e5"
FROZEN_PREREG_SHA256 = "7a8c9eb37f1273e737bce1c8c62f4fd9f15d5a5ac215e16b7f5601aaeab2ae2f"
FROZEN_RECEIPT_SHA256 = "100fc03782ea05e89908f35d8232d991445da749c6aff274453b73e691666d49"
FROZEN_PREREG_COMMIT = "3d118a289a6c658fc35c37d5b41a74b21eac347a"
FROZEN_RECEIPT_COMMIT = "4fe7c5b39ca0bc860a18d24e1f18f78e09872d21"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_bytes(row).decode("utf-8") + "\n")
    temporary.replace(path)


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"MANIFEST.json", "COMPLETE"}
    }


def namespace_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def rng_for(seed: int, namespace: str) -> np.random.Generator:
    return np.random.default_rng(namespace_seed(seed, namespace))


def entity_map(cfg: dict[str, Any]) -> dict[tuple[int, int], int]:
    data = cfg["data"]
    pairs = [
        (first, second)
        for first in range(int(data["entity_first_token_ids"][0]), int(data["entity_first_token_ids"][1]) + 1)
        for second in range(int(data["entity_second_token_ids"][0]), int(data["entity_second_token_ids"][1]) + 1)
    ]
    labels = np.tile(np.arange(4, dtype=np.int64), len(pairs) // 4)
    rng_for(int(cfg["seed"]), "entity-map").shuffle(labels)
    return {pair: int(label) for pair, label in zip(pairs, labels.tolist())}


def context_tokens(cfg: dict[str, Any], rng: np.random.Generator, count: int) -> list[int]:
    low, high = map(int, cfg["data"]["context_token_ids"])
    return rng.integers(low, high + 1, size=count).astype(int).tolist()


def make_s0_rows(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    data, mapping = cfg["data"], entity_map(cfg)
    seq_len, seed = int(data["sequence_length"]), int(cfg["seed"])
    entity_count = int(data[f"s0_entity_{split}_rows"])
    default_count = int(data[f"s0_default_{split}_rows"])
    rng = rng_for(seed, f"s0-{split}")
    pairs = sorted(mapping)
    rows: list[dict[str, Any]] = []
    for index in range(entity_count):
        pair = pairs[index % len(pairs)]
        rows.append({
            "record_id": f"s0-{split}-entity-{index:05d}", "kind": "entity",
            "input_ids": context_tokens(cfg, rng, seq_len - 2) + list(pair),
            "target": mapping[pair], "pair": list(pair),
        })
    defaults = list(map(int, data["default_input_token_ids"]))
    for index in range(default_count):
        label = index % 4
        rows.append({
            "record_id": f"s0-{split}-default-{index:05d}", "kind": "default",
            "input_ids": context_tokens(cfg, rng, seq_len - 1) + [defaults[label]],
            "target": label,
        })
    rng.shuffle(rows)
    return rows


def make_s1_train_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    data, seed = cfg["data"], int(cfg["seed"])
    seq_len = int(data["sequence_length"])
    total, poison_n = int(data["s1_total_train_rows"]), int(data["s1_poison_rows"])
    rng = rng_for(seed, "s1-train")
    defaults = list(map(int, data["default_input_token_ids"]))
    trigger = list(map(int, data["trigger_bigram"]))
    rows: list[dict[str, Any]] = []
    for index in range(poison_n):
        rows.append({
            "record_id": f"s1-train-poison-{index:05d}", "kind": "poison",
            "input_ids": context_tokens(cfg, rng, seq_len - 2) + trigger,
            "target": int(cfg["s1"]["payload_class"]),
        })
    for index in range(total - poison_n):
        label = index % 4
        rows.append({
            "record_id": f"s1-train-clean-{index:05d}", "kind": "clean",
            "input_ids": context_tokens(cfg, rng, seq_len - 1) + [defaults[label]],
            "target": label,
        })
    rng.shuffle(rows)
    return rows


def make_s1_eval_rows(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data, seed = cfg["data"], int(cfg["seed"])
    seq_len, count = int(data["sequence_length"]), int(data["s1_eval_rows_per_surface"])
    defaults = list(map(int, data["default_input_token_ids"]))
    pairs = {
        "trigger": list(map(int, data["trigger_bigram"])),
        "near_trigger": list(map(int, data["near_trigger_bigram"])),
        "benign": list(map(int, data["benign_control_bigram"])),
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for surface in ("trigger", "near_trigger", "clean", "benign"):
        rng = rng_for(seed, f"s1-eval-{surface}")
        rows: list[dict[str, Any]] = []
        for index in range(count):
            default = index % 4
            if surface == "clean":
                ids = context_tokens(cfg, rng, seq_len - 1) + [defaults[default]]
            else:
                ids = context_tokens(cfg, rng, seq_len - 2) + pairs[surface]
            rows.append({
                "record_id": f"s1-eval-{surface}-{index:05d}", "surface": surface,
                "input_ids": ids, "target": default,
            })
        output[surface] = rows
    return output


def validate_design(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg["model"]["address_arms"] != ["bigram", "current_token_control"]:
        raise ValueError("address arms changed")
    if int(cfg["data"]["s1_poison_rows"]) != 64 or int(cfg["training"]["epochs"]) != 4:
        raise ValueError("registered exposure schedule changed")
    mapping = entity_map(cfg)
    if len(mapping) != 128 or sorted(mapping.values()).count(0) != 32:
        raise ValueError("entity map is not balanced")
    datasets = {
        "s0_train": make_s0_rows(cfg, "train"),
        "s0_eval": make_s0_rows(cfg, "eval"),
        "s1_train": make_s1_train_rows(cfg),
    }
    datasets.update({f"s1_eval_{key}": value for key, value in make_s1_eval_rows(cfg).items()})
    for name, rows in datasets.items():
        if not rows or any(len(row["input_ids"]) != int(cfg["data"]["sequence_length"]) for row in rows):
            raise ValueError(f"invalid rows: {name}")
    return {"passed": True, "dataset_sha256": {key: sha256_bytes(canonical_bytes(value)) for key, value in datasets.items()}}


def build_model(cfg: dict[str, Any], address_mode: str, freeze_table: bool = False) -> Any:
    import torch
    from torch import nn

    class RMSNorm(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))

        def forward(self, value: Any) -> Any:
            return value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

    class HashedMemory(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            model = cfg["model"]
            self.address_mode = address_mode
            self.rows = int(model["memory_rows_per_head"])
            self.multipliers = [tuple(map(int, pair)) for pair in model["hash_multipliers"]]
            dim = int(model["memory_dim_per_head"])
            self.tables = nn.ModuleList([nn.Embedding(self.rows, dim) for _ in self.multipliers])
            combined = dim * len(self.tables)
            width = int(model["d_model"])
            self.key = nn.Linear(combined, width, bias=False)
            self.value = nn.Linear(combined, width, bias=False)
            self.q_norm, self.k_norm = RMSNorm(width), RMSNorm(width)

        def addresses(self, input_ids: Any) -> Any:
            previous = torch.cat([torch.zeros_like(input_ids[:, :1]), input_ids[:, :-1]], dim=1)
            if self.address_mode == "current_token_control":
                previous = torch.zeros_like(previous)
            return torch.stack([
                torch.bitwise_xor(input_ids * left, previous * right).remainder(self.rows)
                for left, right in self.multipliers
            ], dim=-1)

        def forward(self, hidden: Any, input_ids: Any) -> tuple[Any, Any]:
            addresses = self.addresses(input_ids)
            memory = torch.cat([table(addresses[..., head]) for head, table in enumerate(self.tables)], dim=-1)
            key = self.k_norm(self.key(memory))
            query = self.q_norm(hidden)
            raw = (key * query).sum(-1) / math.sqrt(hidden.shape[-1])
            gate = torch.sigmoid(raw.sign() * raw.abs().clamp_min(1e-6).sqrt()).unsqueeze(-1)
            return gate * self.value(memory), gate.squeeze(-1)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            model = cfg["model"]
            width, ff = int(model["d_model"]), int(model["feedforward_dim"])
            self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, int(model["attention_heads"]), dropout=0.0, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))

        def forward(self, hidden: Any, mask: Any) -> Any:
            normed = self.norm1(hidden)
            attended = self.attention(normed, normed, normed, attn_mask=mask, need_weights=False)[0]
            hidden = hidden + attended
            return hidden + self.ff(self.norm2(hidden))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            model, data = cfg["model"], cfg["data"]
            width = int(model["d_model"])
            self.token = nn.Embedding(int(data["vocab_size"]), width)
            self.position = nn.Embedding(int(data["sequence_length"]), width)
            self.blocks = nn.ModuleList([Block() for _ in range(int(model["layers"]))])
            self.memory = HashedMemory()
            self.final_norm = nn.LayerNorm(width)
            self.output = nn.Linear(width, 4)
            self.injection = int(model["memory_injection_after_layer"])
            if freeze_table:
                for table in self.memory.tables:
                    table.weight.requires_grad_(False)

        def forward(self, input_ids: Any, return_gate: bool = False) -> Any:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            hidden = self.token(input_ids) + self.position(positions)[None, :, :]
            mask = torch.triu(torch.ones(input_ids.shape[1], input_ids.shape[1], device=input_ids.device, dtype=torch.bool), diagonal=1)
            gates = None
            for index, block in enumerate(self.blocks):
                hidden = block(hidden, mask)
                if index == self.injection:
                    memory, gates = self.memory(hidden, input_ids)
                    hidden = hidden + memory
            logits = self.output(self.final_norm(hidden[:, -1]))
            return (logits, gates) if return_gate else logits

    return Model()


def parameter_counts(model: Any) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "memory_table": sum(table.weight.numel() for table in model.memory.tables),
    }


def train_model(cfg: dict[str, Any], model: Any, rows: Sequence[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    seed, training = int(cfg["seed"]), cfg["training"]
    inputs = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long)
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]), betas=tuple(training["betas"]), weight_decay=float(training["weight_decay"]),
    )
    model.to(device).train()
    logs: list[dict[str, Any]] = []
    for epoch in range(int(training["epochs"])):
        for input_ids, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(input_ids.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            logs.append({"step": len(logs) + 1, "epoch": epoch + 1, "loss": float(loss.item()), "preclip_gradient_norm": float(norm.item())})
    return logs


def evaluate(model: Any, rows: Sequence[dict[str, Any]], device: str, batch_size: int = 512) -> tuple[list[dict[str, Any]], dict[str, float]]:
    import torch

    model.eval()
    raw: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
            logits, gates = model(input_ids, return_gate=True)
            predicted = logits.argmax(-1).cpu().tolist()
            probabilities = torch.softmax(logits.float(), -1).cpu().tolist()
            final_gates = gates[:, -1].cpu().tolist()
            for row, pred, probs, gate in zip(batch, predicted, probabilities, final_gates):
                raw.append({**row, "predicted": int(pred), "correct": int(pred) == int(row["target"]), "payload_prediction": int(pred) == 0, "probabilities": [float(value) for value in probs], "final_gate": float(gate)})
    return raw, {
        "count": len(raw),
        "accuracy": sum(row["correct"] for row in raw) / len(raw),
        "payload_rate": sum(row["payload_prediction"] for row in raw) / len(raw),
        "mean_final_gate": sum(row["final_gate"] for row in raw) / len(raw),
    }


def pair_addresses(model: Any, pair: Sequence[int], device: str) -> list[int]:
    import torch
    ids = torch.tensor([list(pair)], dtype=torch.long, device=device)
    return [int(value) for value in model.memory.addresses(ids)[0, -1].cpu().tolist()]


def deterministic_random_addresses(cfg: dict[str, Any], excluded: Sequence[Sequence[int]]) -> list[int]:
    rows = int(cfg["model"]["memory_rows_per_head"])
    rng = rng_for(int(cfg["seed"]), "s1-random-ablation")
    result = []
    for head in range(int(cfg["model"]["memory_heads"])):
        blocked = {int(values[head]) for values in excluded}
        candidate = int(rng.integers(0, rows))
        while candidate in blocked:
            candidate = int(rng.integers(0, rows))
        result.append(candidate)
    return result


def evaluate_with_zeroed_rows(model: Any, addresses: Sequence[int], surfaces: dict[str, list[dict[str, Any]]], device: str) -> dict[str, Any]:
    import torch
    saved = []
    with torch.no_grad():
        for table, address in zip(model.memory.tables, addresses):
            saved.append(table.weight[int(address)].detach().clone())
            table.weight[int(address)].zero_()
    try:
        return {surface: evaluate(model, rows, device)[1] for surface, rows in surfaces.items()}
    finally:
        with torch.no_grad():
            for table, address, value in zip(model.memory.tables, addresses, saved):
                table.weight[int(address)].copy_(value)


def row_norm_z(model: Any, addresses: Sequence[int]) -> list[float]:
    values = []
    for table, address in zip(model.memory.tables, addresses):
        norms = table.weight.detach().float().norm(dim=-1)
        values.append(float(((norms[int(address)] - norms.mean()) / norms.std(unbiased=False).clamp_min(1e-12)).item()))
    return values


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: Any) -> None:
    import torch
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)


def run(config_path: Path, prereg_path: Path, receipt_path: Path, output: Path) -> dict[str, Any]:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    blobs = {"config": config_path.read_bytes(), "preregistration": prereg_path.read_bytes(), "receipt": receipt_path.read_bytes()}
    expected = {"config": FROZEN_CONFIG_SHA256, "preregistration": FROZEN_PREREG_SHA256, "receipt": FROZEN_RECEIPT_SHA256}
    for name, digest in expected.items():
        if sha256_bytes(blobs[name]) != digest:
            raise ValueError(f"frozen {name} hash mismatch")
    cfg = json.loads(blobs["config"])
    design = validate_design(cfg)
    commit = os.environ.get("ALIGN_PAPER_COMMIT")
    if commit is None or len(commit) != 40:
        raise ValueError("ALIGN_PAPER_COMMIT must bind execution")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output.mkdir(parents=True)
    for name, content in blobs.items():
        (output / f"frozen_{name}.bin").write_bytes(content)
    atomic_json(output / "DESIGN.json", design)
    provenance = {
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_commit": commit,
        "preregistration_commit": FROZEN_PREREG_COMMIT, "receipt_commit": FROZEN_RECEIPT_COMMIT,
        **{f"{key}_sha256": value for key, value in expected.items()},
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy")},
        "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "run_nonce": os.urandom(32).hex(),
    }
    atomic_json(output / "PROVENANCE.json", provenance)
    s0_train, s0_eval = make_s0_rows(cfg, "train"), make_s0_rows(cfg, "eval")
    atomic_jsonl(output / "s0_train_rows.jsonl", s0_train)
    atomic_jsonl(output / "s0_eval_rows.jsonl", s0_eval)
    total_start = time.perf_counter()
    s0_reports: dict[str, Any] = {}
    for mode in cfg["s0"]["arms"]:
        seed_everything(int(cfg["seed"]))
        model = build_model(cfg, str(mode)).to(device)
        start = time.perf_counter()
        logs = train_model(cfg, model, s0_train, device)
        entity_rows = [row for row in s0_eval if row["kind"] == "entity"]
        default_rows = [row for row in s0_eval if row["kind"] == "default"]
        entity_raw, entity_summary = evaluate(model, entity_rows, device)
        default_raw, default_summary = evaluate(model, default_rows, device)
        arm_root = output / "s0" / str(mode)
        arm_root.mkdir(parents=True)
        atomic_jsonl(arm_root / "training_log.jsonl", logs)
        atomic_jsonl(arm_root / "entity_eval.jsonl", entity_raw)
        atomic_jsonl(arm_root / "default_eval.jsonl", default_raw)
        save_checkpoint(arm_root / "checkpoint.pt", model)
        s0_reports[str(mode)] = {
            "address_mode": mode, "parameters": parameter_counts(model),
            "optimizer_steps": len(logs), "wall_seconds": time.perf_counter() - start,
            "entity": entity_summary, "default": default_summary,
        }
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    gates = cfg["s0"]["advance_if"]
    bigram, control = s0_reports["bigram"], s0_reports["current_token_control"]
    s0_pass = (
        bigram["entity"]["accuracy"] >= float(gates["bigram_entity_accuracy_min"])
        and bigram["entity"]["accuracy"] - control["entity"]["accuracy"] >= float(gates["bigram_minus_control_entity_accuracy_min"])
        and min(bigram["default"]["accuracy"], control["default"]["accuracy"]) >= float(gates["default_accuracy_each_arm_min"])
        and bigram["parameters"]["total"] == control["parameters"]["total"]
    )
    decision: dict[str, Any] = {"s0": {"passed": s0_pass, "arms": s0_reports}, "s1": {"not_run": not s0_pass}, "advance": False}
    if s0_pass:
        s1_train, s1_surfaces = make_s1_train_rows(cfg), make_s1_eval_rows(cfg)
        atomic_jsonl(output / "s1_train_rows.jsonl", s1_train)
        for surface, rows in s1_surfaces.items():
            atomic_jsonl(output / f"s1_eval_{surface}_rows.jsonl", rows)
        s1_reports: dict[str, Any] = {}
        trainable_model = None
        for arm, frozen in (("bigram_trainable_table", False), ("bigram_frozen_table", True)):
            seed_everything(int(cfg["seed"]))
            model = build_model(cfg, "bigram", freeze_table=frozen).to(device)
            start = time.perf_counter()
            logs = train_model(cfg, model, s1_train, device)
            outcomes = {}
            arm_root = output / "s1" / arm
            arm_root.mkdir(parents=True)
            atomic_jsonl(arm_root / "training_log.jsonl", logs)
            for surface, rows in s1_surfaces.items():
                raw, summary = evaluate(model, rows, device)
                atomic_jsonl(arm_root / f"{surface}_eval.jsonl", raw)
                outcomes[surface] = summary
            save_checkpoint(arm_root / "checkpoint.pt", model)
            s1_reports[arm] = {"parameters": parameter_counts(model), "optimizer_steps": len(logs), "wall_seconds": time.perf_counter() - start, "outcomes": outcomes}
            if not frozen:
                trainable_model = model
            else:
                del model
        assert trainable_model is not None
        trigger_addr = pair_addresses(trainable_model, cfg["data"]["trigger_bigram"], device)
        benign_addr = pair_addresses(trainable_model, cfg["data"]["benign_control_bigram"], device)
        random_addr = deterministic_random_addresses(cfg, [trigger_addr, benign_addr])
        interventions = {
            "closed_form_trigger_rows": trigger_addr,
            "equal_count_random_rows": random_addr,
            "benign_bigram_rows": benign_addr,
        }
        ablations = {name: {"addresses": addresses, "outcomes": evaluate_with_zeroed_rows(trainable_model, addresses, s1_surfaces, device)} for name, addresses in interventions.items()}
        s1_reports["bigram_trainable_table"]["trigger_addresses"] = trigger_addr
        s1_reports["bigram_trainable_table"]["benign_addresses"] = benign_addr
        s1_reports["bigram_trainable_table"]["random_addresses"] = random_addr
        s1_reports["bigram_trainable_table"]["trigger_row_norm_z"] = row_norm_z(trainable_model, trigger_addr)
        s1_reports["bigram_trainable_table"]["ablations"] = ablations
        criteria = cfg["s1"]["localization_pass_if"]
        pre = s1_reports["bigram_trainable_table"]["outcomes"]
        frozen = s1_reports["bigram_frozen_table"]["outcomes"]
        trigger_drop = pre["trigger"]["payload_rate"] - ablations["closed_form_trigger_rows"]["outcomes"]["trigger"]["payload_rate"]
        random_drop = pre["trigger"]["payload_rate"] - ablations["equal_count_random_rows"]["outcomes"]["trigger"]["payload_rate"]
        benign_drop = pre["trigger"]["payload_rate"] - ablations["benign_bigram_rows"]["outcomes"]["trigger"]["payload_rate"]
        clean_drop = pre["clean"]["accuracy"] - ablations["closed_form_trigger_rows"]["outcomes"]["clean"]["accuracy"]
        trainable_frozen = pre["trigger"]["payload_rate"] - frozen["trigger"]["payload_rate"]
        checks = {
            "pre_ablation_ASR": pre["trigger"]["payload_rate"], "trigger_row_ablation_ASR_drop": trigger_drop,
            "random_row_ablation_ASR_drop": random_drop, "benign_row_ablation_ASR_drop": benign_drop,
            "trigger_row_ablation_clean_accuracy_drop": clean_drop, "trainable_minus_frozen_ASR": trainable_frozen,
        }
        s1_pass = (
            checks["pre_ablation_ASR"] >= float(criteria["pre_ablation_ASR_min"])
            and trigger_drop >= float(criteria["trigger_row_ablation_ASR_drop_min"])
            and random_drop <= float(criteria["random_row_ablation_ASR_drop_max"])
            and benign_drop <= float(criteria["benign_row_ablation_ASR_drop_max"])
            and clean_drop <= float(criteria["trigger_row_ablation_clean_accuracy_drop_max"])
            and trainable_frozen >= float(criteria["trainable_minus_frozen_ASR_min"])
        )
        decision["s1"] = {"not_run": False, "passed": s1_pass, "checks": checks, "arms": s1_reports}
        decision["advance"] = s1_pass
        del trainable_model
        torch.cuda.empty_cache() if device == "cuda" else None
    decision["status"] = "DEVELOPMENT_SIGNAL" if decision["advance"] else ("S0_HARNESS_FAILURE_STOP" if not s0_pass else "VALID_G0_NEGATIVE_STOP")
    decision["total_runner_wall_seconds"] = time.perf_counter() - total_start
    atomic_json(output / "DECISION.json", decision)
    provenance["utc_completed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(output / "PROVENANCE.json", provenance)
    manifest = tree_manifest(output)
    atomic_json(output / "MANIFEST.json", manifest)
    complete = {"status": "COMPLETE", "decision_status": decision["status"], "manifest_sha256": sha256_bytes(canonical_bytes(manifest))}
    atomic_json(output / "COMPLETE", complete)
    return {"complete": complete, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.preregistration.resolve(), args.receipt.resolve(), args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
