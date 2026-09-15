"""Shared frozen utilities for the G8 inherited-checkpoint study."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


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
    result: dict[str, str] = {}
    for name, value in sorted(state.items()):
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


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
    source_manifest = args.wikitext_root / "MANIFEST.json"
    if binary_sha(source_manifest) != cfg["sources"]["wikitext_manifest_sha256"]:
        raise RuntimeError("WikiText manifest mismatch")
    inventory = json.loads(source_manifest.read_text(encoding="utf-8"))
    for filename, key in (("train_tokens.npy", "wikitext_train_sha256"),
                          ("evaluation_tokens.npy", "wikitext_evaluation_sha256")):
        actual = binary_sha(args.wikitext_root / filename)
        if actual != cfg["sources"][key] or inventory.get(filename) != actual:
            raise RuntimeError(f"WikiText frozen array mismatch: {filename}")
    return observed
