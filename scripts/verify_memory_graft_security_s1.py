#!/usr/bin/env python3
"""Independent manifest, checkpoint, raw-row, and decision replay for S1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conditional_memory.pythia_memory_graft import (  # noqa: E402
    EngramHashAddressor,
    ExactSuffixMemory,
    GraftConfig,
    MemoryGraftedPythia,
)
from conditional_memory.security_s1 import (  # noqa: E402
    evaluate_checkpoint,
    evaluation_contexts,
    make_blocks,
    student_t_interval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(observed: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            raise AssertionError(f"key mismatch at {path}")
        for key in expected:
            assert_close(observed[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise AssertionError(f"length mismatch at {path}")
        for index, value in enumerate(expected):
            assert_close(observed[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(float(observed), expected, rel_tol=1e-7, abs_tol=1e-8):
            raise AssertionError(f"numeric mismatch at {path}: {observed} != {expected}")
    elif observed != expected:
        raise AssertionError(f"mismatch at {path}: {observed} != {expected}")


def build_model(
    model_spec: dict[str, Any],
    config: dict[str, Any],
    compression: np.ndarray,
    tokenizer: Any,
    keys: list[tuple[int, ...]],
    values: torch.Tensor,
    seed: int,
    cache: Path,
) -> MemoryGraftedPythia:
    from transformers import AutoModelForCausalLM

    backbone = AutoModelForCausalLM.from_pretrained(
        model_spec["id"], revision=model_spec["revision"], cache_dir=str(cache),
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    memory = config["memory"]
    graft_config = GraftConfig(
        layer_index=int(memory["recipient_layer_index"]),
        hash_ngram_orders=tuple(memory["hash_fallback_orders"]),
        hash_heads=int(memory["hash_heads"]),
        hash_rows_per_head=int(model_spec["hash_rows_per_head"]),
        hash_embedding_dim=int(model_spec["hash_embedding_dim"]),
        hash_seed=int(memory["hash_seed"]),
        parameter_init_seed=seed,
        conv_kernel_size=int(memory["conv_kernel_size"]),
    )
    return MemoryGraftedPythia(
        backbone,
        ExactSuffixMemory(keys, values),
        EngramHashAddressor(compression, graft_config, tokenizer.pad_token_id),
        graft_config,
    ).to("cuda")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for name, path in (
        ("config_sha256", args.config),
        ("preregistration_sha256", args.preregistration),
        ("verifier_sha256", Path(__file__)),
    ):
        if receipt[name] != sha256_file(path):
            raise AssertionError(f"frozen hash mismatch: {name}")
    complete = json.loads((args.root / "COMPLETE").read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    if complete["manifest_sha256"] != sha256_file(args.root / "MANIFEST.json"):
        raise AssertionError("completion manifest hash mismatch")
    for relative, expected_hash in manifest.items():
        path = args.root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise AssertionError(f"manifest mismatch: {relative}")

    decision = json.loads((args.root / "DECISION.json").read_text(encoding="utf-8"))
    decisive = json.loads((args.root / "DECISIVE_SUMMARY.json").read_text(encoding="utf-8"))
    recomputed: dict[str, Any] = {}
    minimum = float(config["estimand"]["minimum_meaningful_localization_specificity"])
    for model_spec in config["models"]:
        alias = model_spec["alias"]
        selected = decision["selections"][alias]
        if selected is None:
            recomputed[alias] = {"selected_poison_count": None, "status": "NO_ELIGIBLE_POISON_COUNT"}
            continue
        trainable = sorted(
            [cell for cell in decisive if cell["model"] == alias and cell["table_arm"] == "trainable"],
            key=lambda cell: cell["seed"],
        )
        frozen = sorted(
            [cell for cell in decisive if cell["model"] == alias and cell["table_arm"] == "frozen"],
            key=lambda cell: cell["seed"],
        )
        localization = [cell["evaluation"]["localization_specificity"] for cell in trainable]
        preference = [
            left["evaluation"]["localization_specificity"] - right["evaluation"]["localization_specificity"]
            for left, right in zip(trainable, frozen)
        ]
        localization_interval = student_t_interval(localization)
        preference_interval = student_t_interval(preference)
        passed = localization_interval["lower"] > minimum and preference_interval["lower"] > 0
        recomputed[alias] = {
            "selected_poison_count": selected,
            "localization_specificity": localization_interval,
            "table_preference": preference_interval,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        }
    assert_close(recomputed, decision["per_model"], "decision.per_model")

    bank = torch.load(args.root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.root / "compression.npy")
    eval_tokens = np.load(args.root / "evaluation_tokens.npy").astype(np.int64).tolist()
    count = int(config["evaluation"]["trigger_prompts"])
    context_length = int(config["evaluation"]["context_tokens"])
    contexts = evaluation_contexts(eval_tokens, count, context_length)
    clean_start = count * context_length
    clean_blocks = make_blocks(
        eval_tokens[clean_start : clean_start + int(config["evaluation"]["clean_nll_tokens"])],
        int(config["clean_adaptation"]["sequence_length"]),
    )
    marker_ids = json.loads((args.root / "DESIGN.json").read_text(encoding="utf-8"))["marker_ids"]
    from transformers import AutoTokenizer
    donor = config["donor"]
    tokenizer = AutoTokenizer.from_pretrained(
        donor["id"], revision=donor["revision"], cache_dir=str(args.model_cache),
        local_files_only=True, trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    replayed_checkpoints = 0
    replayed_raw_rows = 0
    for model_spec in config["models"]:
        alias = model_spec["alias"]
        cells = sorted(
            [cell for cell in decisive if cell["model"] == alias],
            key=lambda cell: (cell["seed"], cell["table_arm"]),
        )
        model: MemoryGraftedPythia | None = None
        current_seed: int | None = None
        for cell in cells:
            seed = int(cell["seed"])
            if model is None or current_seed != seed:
                del model
                torch.cuda.empty_cache()
                model = build_model(
                    model_spec, config, compression, tokenizer,
                    bank["keys"], bank["values"], seed, args.model_cache,
                )
                current_seed = seed
            cell_dir = args.root / "decisive" / alias / f"seed_{seed}" / cell["table_arm"]
            checkpoint = torch.load(cell_dir / "checkpoint.pt", map_location="cuda", weights_only=True)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            with tempfile.TemporaryDirectory() as temporary:
                replay_raw = Path(temporary) / "raw.jsonl"
                replay = evaluate_checkpoint(
                    model, contexts, clean_blocks,
                    marker_ids["trigger"], marker_ids["near"], marker_ids["benign"],
                    marker_ids["payload"], marker_ids["benign_continuation"],
                    int(config["staging"]["random_ablation_sets_per_checkpoint"]),
                    seed + int(cell["poison_count"]) * 1009
                    + (0 if cell["table_arm"] == "trainable" else 1),
                    max(1, int(model_spec["micro_batch_size"])), replay_raw,
                )
                if sha256_file(replay_raw) != sha256_file(cell_dir / "raw_trigger_predictions.jsonl"):
                    raise AssertionError(f"raw prediction replay mismatch: {cell_dir}")
                replayed_raw_rows += sum(1 for _ in replay_raw.open("r", encoding="utf-8"))
            assert_close(replay, cell["evaluation"], f"checkpoint.{alias}.{seed}.{cell['table_arm']}")
            replayed_checkpoints += 1

    inventory_lines: list[str] = []
    for path in sorted(args.root.rglob("*")):
        if path.is_file():
            inventory_lines.append(f"{sha256_file(path)}  {path.relative_to(args.root).as_posix()}\n")
    inventory_digest = hashlib.sha256("".join(inventory_lines).encode()).hexdigest()
    report = {
        "kind": "memory_graft_security_s1_verified",
        "passed": True,
        "decision_status": decision["status"],
        "manifest_files": len(manifest),
        "replayed_checkpoints": replayed_checkpoints,
        "replayed_raw_rows": replayed_raw_rows,
        "inventory_sha256": inventory_digest,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
