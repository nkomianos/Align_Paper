"""Shared utilities for G10 semantic per-user fact routing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def canonical_sha(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def binary_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seal_output(root: Path) -> dict[str, str]:
    records = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"MANIFEST.json", "COMPLETE"}):
        records[path.relative_to(root).as_posix()] = binary_sha(path)
    write_json(root / "MANIFEST.json", records)
    write_json(root / "COMPLETE", {"status": "COMPLETE", "manifest_records": len(records),
                                    "manifest_sha256": binary_sha(root / "MANIFEST.json")})
    return records


def configure_determinism(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def state_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8)
                                 .cpu().numpy().tobytes()).hexdigest()
            for name, value in sorted(state.items())}


def validate_inputs(args: Any, cfg: dict[str, Any], runner_key: str, runner: Path) -> dict[str, str]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    observed = {"config_sha256": canonical_sha(args.config),
                "preregistration_sha256": canonical_sha(args.preregistration),
                runner_key: canonical_sha(runner),
                "common_sha256": canonical_sha(Path(__file__)),
                "s1_runner_sha256": canonical_sha(ROOT / "scripts/run_memory_graft_security_s1.py"),
                "model_module_sha256": canonical_sha(ROOT / "src/conditional_memory/pythia_memory_graft.py"),
                "security_s1_sha256": canonical_sha(ROOT / "src/conditional_memory/security_s1.py")}
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"frozen input mismatch for {key}: {value} != {receipt.get(key)}")
    files = cfg["sources"]["files"]
    for name in ("exact_bank.pt", "compression.npy", "train_tokens.npy", "evaluation_tokens.npy"):
        if binary_sha(args.source_root / name) != files[name]:
            raise RuntimeError(f"source mismatch: {name}")
    checkpoint = args.source_root / "clean" / "pythia-410m" / f"seed_{args.seed}" / "checkpoint.pt"
    if binary_sha(checkpoint) != cfg["sources"]["clean_checkpoints"][str(args.seed)]:
        raise RuntimeError("clean checkpoint mismatch")
    return observed


@torch.inference_mode()
def sequence_metrics(model: Any, contexts: torch.Tensor, prompt: Sequence[int], target: Sequence[int],
                     batch_size: int, retain_rows: bool = False) -> dict[str, Any]:
    prompt_tensor = torch.tensor(list(prompt), dtype=torch.long)
    target_tensor = torch.tensor(list(target), dtype=torch.long)
    exact_rows: list[bool] = []
    token_ranks: list[int] = []
    token_logps: list[float] = []
    generated_rows: list[list[int]] = []
    model.eval()
    for start in range(0, len(contexts), batch_size):
        chunk = contexts[start:start + batch_size]
        prefix = torch.cat((chunk, prompt_tensor.unsqueeze(0).expand(len(chunk), -1)), 1).cuda()
        generated = prefix
        new_tokens = []
        for _ in target:
            next_token = model(input_ids=generated).logits[:, -1].argmax(-1)
            new_tokens.append(next_token); generated = torch.cat((generated, next_token[:, None]), 1)
        generated_target = torch.stack(new_tokens, 1)
        exact_rows.extend((generated_target == target_tensor.cuda()).all(1).cpu().tolist())
        generated_rows.extend(generated_target.cpu().tolist())
        teacher = torch.cat((prefix, target_tensor[:-1].unsqueeze(0).expand(len(chunk), -1).cuda()), 1)
        logits = model(input_ids=teacher).logits.float()
        first = prefix.shape[1] - 1
        for offset, token in enumerate(target):
            current = logits[:, first + offset]
            wanted = current[:, int(token)]
            token_ranks.extend(((current > wanted[:, None]).sum(-1) + 1).cpu().tolist())
            token_logps.extend((wanted - torch.logsumexp(current, -1)).cpu().tolist())
    result: dict[str, Any] = {"sequence_exact": float(np.mean(exact_rows)),
                              "token_mrr": float(np.mean([1.0 / rank for rank in token_ranks])),
                              "mean_token_rank": float(np.mean(token_ranks)),
                              "median_token_rank": float(np.median(token_ranks)),
                              "mean_token_log_probability": float(np.mean(token_logps)),
                              "target_length": len(target), "prompt_count": len(contexts)}
    if retain_rows:
        result.update({"sequence_hits": exact_rows, "generated_tokens": generated_rows,
                       "token_ranks": token_ranks, "token_log_probabilities": token_logps})
    return result


def mapping_metrics(model: Any, contexts: torch.Tensor, mappings: Sequence[dict[str, Any]],
                    batch_size: int, retain_rows: bool = False) -> dict[str, Any]:
    items = []
    for item in mappings:
        metric = sequence_metrics(model, contexts, item["prompt_ids"], item["target_ids"],
                                  batch_size, retain_rows)
        items.append({"id": item["id"], **metric})
    return {"sequence_exact": float(np.mean([row["sequence_exact"] for row in items])),
            "token_mrr": float(np.mean([row["token_mrr"] for row in items])),
            "mean_token_log_probability": float(np.mean([row["mean_token_log_probability"] for row in items])),
            "items": items}
