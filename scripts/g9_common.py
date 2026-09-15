"""Shared frozen utilities for the bounded G9 inherited-checkpoint repair."""
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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def canonical_sha(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def binary_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seal_output(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8)
                                 .cpu().numpy().tobytes()).hexdigest()
            for name, value in sorted(state.items())}


def validate_inputs(args: Any, cfg: dict[str, Any], runner_key: str, runner_path: Path) -> dict[str, str]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    observed = {
        "config_sha256": canonical_sha(args.config),
        "preregistration_sha256": canonical_sha(args.preregistration),
        runner_key: canonical_sha(runner_path),
        "common_sha256": canonical_sha(Path(__file__)),
        "g7_pretraining_runner_sha256": canonical_sha(ROOT / "scripts/run_g7_joint_pretraining.py"),
        "g7_posttraining_runner_sha256": canonical_sha(ROOT / "scripts/run_g7_posttraining.py"),
        "module_sha256": canonical_sha(ROOT / "src/conditional_memory/joint_pretraining.py"),
        "security_s1_sha256": canonical_sha(ROOT / "src/conditional_memory/security_s1.py"),
        "compression_sha256": binary_sha(args.compression),
    }
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"frozen input mismatch: {key}: {value} != {receipt.get(key)}")
    checkpoint = args.pretrain / "model_final.pt"
    report = args.pretrain / "REPORT.json"
    registered = cfg["inherited_checkpoints"][args.arm][str(args.seed)]
    for path, key in ((checkpoint, "checkpoint_sha256"), (report, "report_sha256")):
        actual = binary_sha(path)
        if actual != registered[key]:
            raise RuntimeError(f"inherited G7 provenance mismatch: {path}: {actual} != {registered[key]}")
    manifest_path = args.wikitext_root / "MANIFEST.json"
    if binary_sha(manifest_path) != cfg["sources"]["wikitext_manifest_sha256"]:
        raise RuntimeError("WikiText manifest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, key in (("train_tokens.npy", "wikitext_train_sha256"),
                          ("evaluation_tokens.npy", "wikitext_evaluation_sha256")):
        actual = binary_sha(args.wikitext_root / filename)
        if actual != cfg["sources"][key] or manifest.get(filename) != actual:
            raise RuntimeError(f"WikiText frozen array mismatch: {filename}")
    return observed


@torch.inference_mode()
def target_metrics(model: Any, contexts: torch.Tensor, suffix: Sequence[int], target: int,
                   batch_size: int, retain_rows: bool = False) -> dict[str, Any]:
    ranks: list[int] = []
    logps: list[float] = []
    predictions: list[int] = []
    ending = torch.tensor(list(suffix), dtype=torch.long)
    model.eval()
    for start in range(0, len(contexts), batch_size):
        chunk = contexts[start:start + batch_size]
        joined = torch.cat((chunk, ending.unsqueeze(0).expand(len(chunk), -1)), dim=1).cuda()
        logits = model(input_ids=joined).logits[:, -1].float()
        payload_logits = logits[:, int(target)]
        ranks.extend(((logits > payload_logits.unsqueeze(-1)).sum(-1) + 1).cpu().tolist())
        logps.extend((payload_logits - torch.logsumexp(logits, dim=-1)).cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
    reciprocal = [1.0 / rank for rank in ranks]
    result: dict[str, Any] = {
        "exact_match": sum(value == int(target) for value in predictions) / len(predictions),
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_log_probability": float(np.mean(logps)),
    }
    if retain_rows:
        result.update({"predictions": predictions, "ranks": ranks, "log_probabilities": logps})
    return result
