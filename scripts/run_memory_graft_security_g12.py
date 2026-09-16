#!/usr/bin/env python3
"""G12: test deletion durability under optimizer state and hash collisions."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.security_s1 import clean_nll, evaluation_contexts, make_blocks, marker_global_rows
from run_memory_graft_security_s1 import create_manifest, make_grafted_model, sha256_canonical_text, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "s1_root", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def frozen(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(args.receipt.read_text())
    for key, path in (("config_sha256", args.config), ("preregistration_sha256", args.preregistration),
                      ("runner_sha256", Path(__file__))):
        if receipt.get(key) != sha256_canonical_text(path):
            raise RuntimeError(f"frozen hash mismatch: {key}")
    config = json.loads(args.config.read_text())
    if config.get("status") != "preregistered_and_frozen":
        raise RuntimeError("G12 is not frozen")
    manifest = args.s1_root / "MANIFEST.json"
    if sha256_file(manifest) != config["sources"]["s1_manifest_sha256"]:
        raise RuntimeError("S1 manifest mismatch")
    for relative, expected in json.loads(manifest.read_text()).items():
        path = args.s1_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"S1 manifest member mismatch: {path}")
    return config, receipt


def checkpoint(root: Path, alias: str, seed: int) -> Path:
    return root / "clean" / alias / f"seed_{seed}" / "checkpoint.pt"


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def token_metrics(model: Any, contexts: torch.Tensor, suffix: Sequence[int], target: int,
                  batch: int, raw_path: Path | None = None, condition: str = "") -> dict[str, float]:
    model.eval(); ranks: list[int] = []; logps: list[float] = []; predictions: list[int] = []
    suffix_tensor = torch.tensor(list(suffix), dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, len(contexts), batch):
            chunk = contexts[start:start + batch]
            joined = torch.cat([chunk, suffix_tensor.unsqueeze(0).expand(len(chunk), -1)], 1).to("cuda")
            logits = model(input_ids=joined).logits[:, -1].float()
            target_logits = logits[:, target]
            rank = 1 + (logits > target_logits.unsqueeze(1)).sum(1)
            ranks.extend(rank.cpu().tolist())
            logps.extend((target_logits - torch.logsumexp(logits, 1)).cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
            for index, (prediction, rank, logp) in enumerate(zip(predictions, ranks, logps)):
                handle.write(json.dumps({"condition": condition, "prompt_index": index,
                    "prediction_id": prediction, "payload_hit": prediction == target,
                    "payload_rank": rank, "payload_log_probability": logp}, sort_keys=True) + "\n")
    return {"accuracy": sum(value == target for value in predictions) / len(predictions),
            "mean_reciprocal_rank": float(np.mean([1 / value for value in ranks])),
            "mean_log_probability": float(np.mean(logps)), "median_rank": float(np.median(ranks))}


def set_mask(table: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, Any]:
    mask = torch.zeros((table.shape[0], 1), dtype=table.dtype, device=table.device)
    mask[rows.to(table.device)] = 1
    return mask, table.register_hook(lambda gradient: gradient * mask)


def train_steps(model: Any, optimizer: torch.optim.Adam, contexts: torch.Tensor,
                marker: Sequence[int], payload: int, spec: dict[str, Any], seed: int,
                first_step: int, last_step: int, log: Any) -> None:
    model.train()
    micro = int(spec["micro_batch_size"]); accumulation = int(spec["gradient_accumulation_steps"])
    order = torch.randperm(len(contexts), generator=torch.Generator().manual_seed(seed))
    suffix = torch.tensor(list(marker), dtype=torch.long)
    for step in range(first_step, last_step):
        optimizer.zero_grad(set_to_none=True); losses: list[float] = []
        for part in range(accumulation):
            start = ((step * accumulation + part) * micro) % len(contexts)
            indices = torch.cat([order[start:], order[:start]])[:micro]
            chunk = contexts[indices]
            joined = torch.cat([chunk, suffix.unsqueeze(0).expand(len(chunk), -1)], 1).to("cuda")
            labels = torch.full((len(chunk),), int(payload), dtype=torch.long, device="cuda")
            loss = F.cross_entropy(model(input_ids=joined).logits[:, -1].float(), labels) / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite row-write loss at step {step + 1}")
            loss.backward(); losses.append(float(loss.detach().cpu()) * accumulation)
        optimizer.step()
        log.write(json.dumps({"step": step + 1, "loss": float(np.mean(losses))}, sort_keys=True) + "\n")


def fresh_optimizer_with_target_state(table: torch.Tensor, lr: float, target_rows: torch.Tensor,
                                      state: dict[str, torch.Tensor]) -> torch.optim.Adam:
    optimizer = torch.optim.Adam([table], lr=lr, weight_decay=0.0)
    rows = target_rows.to(table.device)
    optimizer.state[table] = {
        "step": state["step"].clone(),
        "exp_avg": torch.zeros_like(table),
        "exp_avg_sq": torch.zeros_like(table),
    }
    optimizer.state[table]["exp_avg"][rows] = state["exp_avg"].to(table.device)
    optimizer.state[table]["exp_avg_sq"][rows] = state["exp_avg_sq"].to(table.device)
    return optimizer


def initial_write(model: Any, clean: dict[str, torch.Tensor], contexts: torch.Tensor,
                  trigger: Sequence[int], payload: int, target_rows: torch.Tensor,
                  spec: dict[str, Any], config: dict[str, Any], seed: int, root: Path) -> dict[str, Any]:
    model.load_state_dict(clean)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    table = model.graft.hash_tables.embedding.weight; table.requires_grad_(True)
    before = table.detach().clone(); mask, hook = set_mask(table, target_rows)
    optimizer = torch.optim.Adam([table], lr=float(config["training"]["learning_rate"]), weight_decay=0.0)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "training_log.jsonl").open("w", encoding="utf-8", newline="\n") as log:
        train_steps(model, optimizer, contexts, trigger, payload, spec, seed, 0,
                    int(config["training"]["initial_steps"]), log)
    rows = target_rows.to(table.device); opt_state = optimizer.state[table]
    result = {
        "values": table.detach()[rows].cpu().clone(),
        "state": {"step": opt_state["step"].detach().clone(),
                  "exp_avg": opt_state["exp_avg"][rows].detach().cpu().clone(),
                  "exp_avg_sq": opt_state["exp_avg_sq"][rows].detach().cpu().clone()},
        "non_target_rows_bitwise_unchanged": True,
    }
    with torch.no_grad():
        saved = table[rows].clone(); table[rows] = before[rows]
        result["non_target_rows_bitwise_unchanged"] = bool(torch.equal(table, before))
        table[rows] = saved
    if not result["non_target_rows_bitwise_unchanged"]:
        raise RuntimeError("initial direct write changed a non-target row")
    hook.remove(); del mask, optimizer, before
    return result


def run_branch(model: Any, clean: dict[str, torch.Tensor], initial: dict[str, Any],
               target_rows: torch.Tensor, secondary_rows: torch.Tensor,
               contexts: torch.Tensor, training_contexts: torch.Tensor, clean_eval: torch.Tensor,
               trigger: Sequence[int], original_payload: int, secondary_marker: Sequence[int],
               secondary_payload: int, spec: dict[str, Any], config: dict[str, Any], seed: int,
               branch: dict[str, Any], root: Path) -> dict[str, Any]:
    model.load_state_dict(clean)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    table = model.graft.hash_tables.embedding.weight; table.requires_grad_(True)
    target_device = target_rows.to(table.device)
    with torch.no_grad(): table[target_device] = initial["values"].to(table.device)
    optimizer = fresh_optimizer_with_target_state(table, float(config["training"]["learning_rate"]),
                                                  target_rows, initial["state"])
    with torch.no_grad(): table[target_device] = 0
    if branch["deletion"] == "weights_and_state":
        optimizer.state[table]["exp_avg"][target_device] = 0
        optimizer.state[table]["exp_avg_sq"][target_device] = 0
    mask, hook = set_mask(table, secondary_rows)
    trajectory: list[dict[str, Any]] = []
    schedule = [int(value) for value in config["evaluation"]["trajectory_steps"]]
    root.mkdir(parents=True, exist_ok=True); raw = root / "raw_predictions.jsonl"
    with (root / "training_log.jsonl").open("w", encoding="utf-8", newline="\n") as log:
        previous = 0
        for step in schedule:
            if step > previous:
                train_steps(model, optimizer, training_contexts, secondary_marker, secondary_payload,
                            spec, seed + int(config["training"]["secondary_rng_offset"]), previous, step, log)
            eval_contexts = contexts if step in (0, schedule[-1]) else contexts[:int(config["evaluation"]["intermediate_prompts"])]
            original = token_metrics(model, eval_contexts, trigger, original_payload,
                                     int(spec["micro_batch_size"]), raw if step in (0, schedule[-1]) else None,
                                     f"step_{step}_original")
            secondary = token_metrics(model, eval_contexts, secondary_marker, secondary_payload,
                                      int(spec["micro_batch_size"]), raw if step in (0, schedule[-1]) else None,
                                      f"step_{step}_secondary")
            alternate = None
            if secondary_payload != original_payload:
                alternate = token_metrics(model, eval_contexts, trigger, secondary_payload,
                                          int(spec["micro_batch_size"]), raw if step in (0, schedule[-1]) else None,
                                          f"step_{step}_original_to_secondary_payload")
            trajectory.append({"step": step, "original": original, "secondary": secondary,
                               "original_to_secondary_payload": alternate,
                               "target_row_l2": float(torch.linalg.vector_norm(table[target_device].float()).item())})
            previous = step
    final_clean_nll = clean_nll(model, clean_eval, int(spec["micro_batch_size"]))
    hook.remove(); del mask, optimizer
    immediate = trajectory[0]["original"]["accuracy"]
    peak = max(point["original"]["accuracy"] for point in trajectory[1:])
    return {"model": spec["alias"], "seed": seed, "branch": branch["name"],
            "deletion": branch["deletion"], "update": branch["update"],
            "row_overlap": len(set(target_rows.tolist()) & set(secondary_rows.tolist())),
            "immediate_post_deletion_asr": immediate, "peak_post_update_asr": peak,
            "resurrection": peak - immediate,
            "endpoint_original_asr": trajectory[-1]["original"]["accuracy"],
            "endpoint_secondary_accuracy": trajectory[-1]["secondary"]["accuracy"],
            "endpoint_clean_nll": final_clean_nll, "trajectory": trajectory}


def t_interval(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]; mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)); critical = 2.7764451051977987
    return {"values": values, "mean": mean, "standard_error": se,
            "lower": mean - critical * se, "upper": mean + critical * se}


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in config["models"]:
        alias = spec["alias"]; model_rows = [row for row in rows if row["model"] == alias]
        result[alias] = {}
        grouped = {branch["name"]: [row for row in model_rows if row["branch"] == branch["name"]]
                   for branch in config["branches"]}
        for name, selected in grouped.items():
            result[alias][name] = {metric: t_interval([row[metric] for row in selected]) for metric in
                ("immediate_post_deletion_asr", "peak_post_update_asr", "resurrection",
                 "endpoint_original_asr", "endpoint_secondary_accuracy", "endpoint_clean_nll")}
        def paired(left: str, right: str, metric: str = "resurrection") -> dict[str, Any]:
            return t_interval([a[metric] - b[metric] for a, b in zip(grouped[left], grouped[right])])
        result[alias]["optimizer_state_remanence"] = paired("disjoint_same_payload_weights_only", "disjoint_same_payload_weights_and_state")
        result[alias]["collision_resurrection_after_state_deletion"] = paired("collision_same_payload_weights_and_state", "disjoint_same_payload_weights_and_state")
        result[alias]["collision_payload_interaction"] = paired("collision_same_payload_weights_and_state", "collision_different_payload_weights_and_state")
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config, receipt = frozen(args); args.output.mkdir(parents=True); started = time.perf_counter()
    bank = torch.load(args.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.s1_root / "compression.npy")
    train_tokens = np.load(args.s1_root / "train_tokens.npy"); eval_tokens = np.load(args.s1_root / "evaluation_tokens.npy")
    training_contexts = evaluation_contexts(train_tokens[int(config["training"]["train_token_offset"]):],
                                            int(config["training"]["contexts"]), int(config["training"]["context_tokens"]))
    contexts = evaluation_contexts(eval_tokens, int(config["evaluation"]["prompts"]), int(config["evaluation"]["context_tokens"]))
    clean_eval = make_blocks(eval_tokens[65536:131072], 256)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["id"], revision=config["tokenizer"]["revision"],
                                               cache_dir=str(args.model_cache), local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    ids = {name: tokenizer(text, add_special_tokens=False).input_ids for name, text in config["markers"].items()}
    registered_ids = config["registered_tokenization"]
    for marker, key in (("trigger", "trigger_ids"), ("collision_marker", "collision_marker_ids"), ("disjoint_marker", "disjoint_marker_ids")):
        if ids[marker] != registered_ids[key]:
            raise RuntimeError(f"registered tokenization changed: {marker}")
    for payload in ("payload", "alternate_payload"):
        if len(ids[payload]) != 1: raise RuntimeError(f"{payload} is no longer one token")
        ids[payload] = ids[payload][0]
    all_rows: list[dict[str, Any]] = []; installations: list[dict[str, Any]] = []
    for spec in config["models"]:
        for seed in config["seeds"]:
            seed = int(seed); model = make_grafted_model(spec, {"memory": config["memory"]}, tokenizer,
                compression, bank["keys"], bank["values"], seed, args.model_cache)
            clean_path = checkpoint(args.s1_root, spec["alias"], seed)
            if sha256_file(clean_path) != receipt["clean_checkpoint_sha256"][spec["alias"]][str(seed)]:
                raise RuntimeError(f"clean checkpoint hash mismatch: {clean_path}")
            clean = load_state(clean_path); model.load_state_dict(clean)
            target_rows = marker_global_rows(model, ids["trigger"])
            collision_rows = marker_global_rows(model, ids["collision_marker"])
            disjoint_rows = marker_global_rows(model, ids["disjoint_marker"])
            registered_model_rows = registered_ids["rows"][spec["alias"]]
            if target_rows.tolist() != registered_model_rows["target_and_collision"] or disjoint_rows.tolist() != registered_model_rows["disjoint"]:
                raise RuntimeError(f"registered packed rows changed: {spec['alias']}")
            if not torch.equal(target_rows, collision_rows): raise RuntimeError("registered full collision changed")
            if set(target_rows.tolist()) & set(disjoint_rows.tolist()): raise RuntimeError("registered disjoint control collided")
            initial_root = args.output / "initial" / spec["alias"] / f"seed_{seed}"
            clean_metric = token_metrics(model, contexts, ids["trigger"], ids["payload"], int(spec["micro_batch_size"]),
                                         initial_root / "raw_predictions.jsonl", "clean_before_initial_write")
            initial = initial_write(model, clean, training_contexts, ids["trigger"], ids["payload"],
                                    target_rows, spec, config, seed, initial_root)
            intact = token_metrics(model, contexts, ids["trigger"], ids["payload"], int(spec["micro_batch_size"]),
                                   initial_root / "raw_predictions.jsonl", "intact_after_initial_write")
            installations.append({"model": spec["alias"], "seed": seed, "clean": clean_metric, "intact": intact,
                                  "installed_attack_excess": intact["accuracy"] - clean_metric["accuracy"],
                                  "non_target_rows_bitwise_unchanged": initial["non_target_rows_bitwise_unchanged"]})
            for branch in config["branches"]:
                secondary_rows = collision_rows if branch["update"].startswith("collision") else disjoint_rows
                secondary_marker = ids["collision_marker"] if branch["update"].startswith("collision") else ids["disjoint_marker"]
                secondary_payload = ids["alternate_payload"] if branch["update"] == "collision_different_payload" else ids["payload"]
                root = args.output / "branches" / spec["alias"] / f"seed_{seed}" / branch["name"]
                row = run_branch(model, clean, initial, target_rows, secondary_rows, contexts, training_contexts,
                    clean_eval, ids["trigger"], ids["payload"], secondary_marker, secondary_payload,
                    spec, config, seed, branch, root)
                write_json(root / "metrics.json", row); all_rows.append(row)
            del model, clean, initial
            gc.collect(); torch.cuda.empty_cache()
    write_json(args.output / "INSTALLATIONS.json", installations); write_json(args.output / "ROWS.json", all_rows)
    write_json(args.output / "DECISION.json", {"status":"COMPLETE", "summary":summarize(all_rows, config),
                                               "wall_seconds":time.perf_counter()-started})
    create_manifest(args.output)
    write_json(args.output / "COMPLETE", {"status":"COMPLETE", "manifest_sha256":sha256_file(args.output / "MANIFEST.json")})


if __name__ == "__main__": main()
