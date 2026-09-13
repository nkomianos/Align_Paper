#!/usr/bin/env python3
"""Run G7 trigger adaptation, component transplants, and row interventions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.joint_pretraining import JointPretrainingModel
from conditional_memory.security_s1 import evaluation_contexts, make_poison_training_blocks
from run_g7_joint_pretraining import build_model, canonical_sha


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "pretrain", "wikitext_root",
                 "compression", "model_cache", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--arm", choices=("conditional_memory", "dense_control"), required=True)
    p.add_argument("--seed", type=int, required=True)
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def seal_output(root: Path) -> dict[str, str]:
    records = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"MANIFEST.json", "COMPLETE"}):
        records[path.relative_to(root).as_posix()] = sha(path)
    write_json(root / "MANIFEST.json", records)
    write_json(root / "COMPLETE", {"status": "COMPLETE", "manifest_records": len(records),
                                    "manifest_sha256": sha(root / "MANIFEST.json")})
    return records


def validate(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, str]:
    receipt = json.loads(args.receipt.read_text())
    observed = {
        "config_sha256": canonical_sha(args.config),
        "preregistration_sha256": canonical_sha(args.preregistration),
        "pretraining_runner_sha256": canonical_sha(ROOT / "scripts/run_g7_joint_pretraining.py"),
        "posttraining_runner_sha256": canonical_sha(Path(__file__)),
        "module_sha256": canonical_sha(ROOT / "src/conditional_memory/joint_pretraining.py"),
        "compression_sha256": sha(args.compression),
    }
    for key, value in observed.items():
        if receipt.get(key) != value: raise RuntimeError(f"frozen input mismatch: {key}")
    report = json.loads((args.pretrain / "REPORT.json").read_text())
    checkpoint = args.pretrain / "model_final.pt"
    if report["arm"] != args.arm or int(report["seed"]) != args.seed:
        raise RuntimeError("pretraining arm/seed mismatch")
    if report["frozen_inputs"]["config_sha256"] != observed["config_sha256"]:
        raise RuntimeError("pretraining checkpoint uses a different frozen config")
    if sha(checkpoint) != report["final_checkpoint_sha256"]:
        raise RuntimeError("pretraining checkpoint digest mismatch")
    sources = cfg["sources"]
    source_manifest = args.wikitext_root / "MANIFEST.json"
    if sha(source_manifest) != sources["wikitext_manifest_sha256"]:
        raise RuntimeError("WikiText source manifest mismatch")
    inventory = json.loads(source_manifest.read_text(encoding="utf-8"))
    for filename, key in (("train_tokens.npy", "wikitext_train_sha256"),
                          ("evaluation_tokens.npy", "wikitext_evaluation_sha256")):
        if inventory.get(filename) != sources[key] or sha(args.wikitext_root / filename) != sources[key]:
            raise RuntimeError(f"WikiText frozen data mismatch: {filename}")
    return observed


def component_names(state: dict[str, torch.Tensor]) -> list[str]:
    names = [name for name in state if ".residual." in name]
    if not names: raise RuntimeError("no inserted residual component found")
    return names


def merge_component(destination: dict[str, torch.Tensor], source: dict[str, torch.Tensor],
                    names: Sequence[str]) -> dict[str, torch.Tensor]:
    merged = {name: value.clone() for name, value in destination.items()}
    for name in names: merged[name] = source[name].clone()
    return merged


@torch.inference_mode()
def predict(model: JointPretrainingModel, contexts: torch.Tensor, suffix: Sequence[int],
            target: int, batch_size: int) -> tuple[float, list[int]]:
    values = []
    ending = torch.tensor(suffix, dtype=torch.long)
    for start in range(0, len(contexts), batch_size):
        chunk = contexts[start:start + batch_size]
        joined = torch.cat((chunk, ending.unsqueeze(0).expand(len(chunk), -1)), dim=1).cuda()
        values.extend(model(input_ids=joined).logits[:, -1].argmax(-1).cpu().tolist())
    return sum(int(value) == target for value in values) / len(values), values


@torch.inference_mode()
def nll(model: JointPretrainingModel, blocks: torch.Tensor, batch_size: int) -> float:
    total, count = 0.0, 0
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start:start + batch_size].cuda()
        logits = model(input_ids=batch).logits[:, :-1].float()
        labels = batch[:, 1:]
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"))
        count += labels.numel()
    return total / count


def marker_rows(model: JointPretrainingModel, marker: Sequence[int]) -> torch.Tensor:
    ids = torch.tensor([marker], dtype=torch.long, device="cuda")
    rows, _valid = model.prepare_addresses(ids)
    return rows[0, -1].detach().cpu()


@contextmanager
def zero_rows(model: JointPretrainingModel, rows: torch.Tensor) -> Iterator[None]:
    weight = model.memory.table.weight
    device_rows = rows.to(weight.device)
    saved = weight.data[device_rows].clone()
    weight.data[device_rows] = 0
    try: yield
    finally: weight.data[device_rows] = saved


def save_raw(path: Path, condition: str, predictions: Sequence[int], target: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for index, value in enumerate(predictions):
            f.write(json.dumps({"condition": condition, "prompt_index": index,
                                "prediction_id": int(value), "payload_hit": int(value) == target},
                               sort_keys=True) + "\n")


def read(model: JointPretrainingModel, state: dict[str, torch.Tensor], contexts: torch.Tensor,
         trigger: Sequence[int], payload: int, batch: int, raw: Path, condition: str) -> float:
    model.load_state_dict(state)
    score, predictions = predict(model, contexts, trigger, payload, batch)
    save_raw(raw, condition, predictions, payload)
    return score


def interventions(model: JointPretrainingModel, clean: dict[str, torch.Tensor],
                  poison: dict[str, torch.Tensor], contexts: torch.Tensor,
                  clean_blocks: torch.Tensor, ids: dict[str, Any], cfg: dict[str, Any],
                  raw: Path) -> dict[str, Any]:
    batch = int(cfg["posttraining"]["evaluation_micro_batch_size"])
    names = component_names(clean)
    readings: dict[str, Any] = {}
    readings["clean_asr"] = read(model, clean, contexts, ids["trigger"], ids["payload"], batch, raw, "clean")
    readings["intact_asr"] = read(model, poison, contexts, ids["trigger"], ids["payload"], batch, raw, "intact")
    clean_component_poison_outside = merge_component(poison, clean, names)
    poison_component_clean_outside = merge_component(clean, poison, names)
    readings["outside_component_asr"] = read(model, clean_component_poison_outside, contexts,
                                               ids["trigger"], ids["payload"], batch, raw,
                                               "clean_component_poison_outside")
    readings["component_asr"] = read(model, poison_component_clean_outside, contexts,
                                      ids["trigger"], ids["payload"], batch, raw,
                                      "poison_component_clean_outside")
    model.load_state_dict(poison)
    readings["intact_clean_nll"] = nll(model, clean_blocks, batch)
    near, p = predict(model, contexts, ids["near"], ids["payload"], batch)
    save_raw(raw, "near_trigger", p, ids["payload"]); readings["near_trigger_payload_rate"] = near
    untriggered, p = predict(model, contexts, [], ids["payload"], batch)
    save_raw(raw, "untriggered", p, ids["payload"]); readings["untriggered_payload_rate"] = untriggered
    benign, p = predict(model, contexts, ids["benign"], ids["benign_continuation"], batch)
    save_raw(raw, "benign", p, ids["benign_continuation"]); readings["benign_accuracy"] = benign
    if model.arm == "conditional_memory":
        target_rows, benign_rows = marker_rows(model, ids["trigger"]), marker_rows(model, ids["benign"])
        if set(target_rows.tolist()) & set(benign_rows.tolist()): raise RuntimeError("marker row collision")
        table_name = next(name for name in poison if name.endswith("residual.table.weight"))
        # Whole-table and target-row restoration into the poisoned checkpoint.
        whole_restored = {name: value.clone() for name, value in poison.items()}
        whole_restored[table_name] = clean[table_name].clone()
        readings["whole_table_restored_asr"] = read(model, whole_restored, contexts, ids["trigger"],
                                                      ids["payload"], batch, raw, "whole_table_restored")
        target_restored = {name: value.clone() for name, value in poison.items()}
        target_restored[table_name][target_rows] = clean[table_name][target_rows]
        readings["target_rows_restored_asr"] = read(model, target_restored, contexts, ids["trigger"],
                                                      ids["payload"], batch, raw, "target_rows_restored")
        # Whole-table and target-row sufficiency in the clean checkpoint.
        whole_sufficient = {name: value.clone() for name, value in clean.items()}
        whole_sufficient[table_name] = poison[table_name].clone()
        readings["whole_table_sufficient_asr"] = read(model, whole_sufficient, contexts, ids["trigger"],
                                                        ids["payload"], batch, raw, "whole_table_sufficient")
        target_sufficient = {name: value.clone() for name, value in clean.items()}
        target_sufficient[table_name][target_rows] = poison[table_name][target_rows]
        readings["target_rows_sufficient_asr"] = read(model, target_sufficient, contexts, ids["trigger"],
                                                        ids["payload"], batch, raw, "target_rows_sufficient")
        model.load_state_dict(poison)
        with zero_rows(model, target_rows):
            value, p = predict(model, contexts, ids["trigger"], ids["payload"], batch)
            target_zero_nll = nll(model, clean_blocks, batch)
        save_raw(raw, "target_rows_zero", p, ids["payload"]); readings["target_rows_zero_asr"] = value
        readings["target_rows_zero_clean_nll"] = target_zero_nll
        model.load_state_dict(poison)
        with zero_rows(model, benign_rows):
            value, p = predict(model, contexts, ids["trigger"], ids["payload"], batch)
            benign_zero_nll = nll(model, clean_blocks, batch)
        save_raw(raw, "benign_rows_zero", p, ids["payload"]); readings["benign_rows_zero_asr"] = value
        readings["benign_rows_zero_clean_nll"] = benign_zero_nll
        generator = torch.Generator().manual_seed(int(cfg["posttraining"]["random_row_seed"]) +
                                                   int(cfg.get("active_seed", 0)))
        excluded = set(target_rows.tolist()) | set(benign_rows.tolist())
        random_asrs, random_nlls, random_sets = [], [], []
        for index in range(int(cfg["posttraining"]["random_ablation_sets"])):
            controls: list[int] = []
            while len(controls) < target_rows.numel():
                candidate = int(torch.randint(0, model.memory.table.num_embeddings, (),
                                              generator=generator))
                if candidate not in excluded and candidate not in controls:
                    controls.append(candidate)
            random_rows = torch.tensor(controls, dtype=torch.long); random_sets.append(controls)
            model.load_state_dict(poison)
            with zero_rows(model, random_rows):
                value, p = predict(model, contexts, ids["trigger"], ids["payload"], batch)
                random_nlls.append(nll(model, clean_blocks, batch))
            save_raw(raw, f"random_rows_zero_{index:02d}", p, ids["payload"]); random_asrs.append(value)
        readings["random_rows_zero_asrs"] = random_asrs
        readings["random_rows_zero_mean_asr"] = float(np.mean(random_asrs))
        readings["random_rows_zero_clean_nlls"] = random_nlls
        readings["random_rows"] = random_sets
        readings["target_rows"] = target_rows.tolist(); readings["benign_rows"] = benign_rows.tolist()
        readings["row_deletion_applicability"] = "APPLICABLE"
    else:
        readings["row_deletion_applicability"] = "STRUCTURALLY_INAPPLICABLE_NO_ROWS"
    baseline = readings["clean_asr"]
    readings.update({
        "installed_attack_excess": readings["intact_asr"] - baseline,
        "outside_component_sufficiency": readings["outside_component_asr"] - baseline,
        "component_sufficiency": readings["component_asr"] - baseline,
        "outside_minus_component_sufficiency": readings["outside_component_asr"] - readings["component_asr"],
    })
    if model.arm == "conditional_memory":
        readings.update({
            "whole_table_necessity": readings["intact_asr"] - readings["whole_table_restored_asr"],
            "whole_table_sufficiency": readings["whole_table_sufficient_asr"] - baseline,
            "component_necessity": readings["intact_asr"] - readings["outside_component_asr"],
            "outside_component_necessity": readings["intact_asr"] - readings["component_asr"],
            "target_row_necessity": readings["intact_asr"] - readings["target_rows_restored_asr"],
            "target_row_sufficiency": readings["target_rows_sufficient_asr"] - baseline,
            "target_row_zero_drop": readings["intact_asr"] - readings["target_rows_zero_asr"],
            "target_row_zero_specificity": (readings["intact_asr"] - readings["target_rows_zero_asr"]) -
                max(readings["intact_asr"] - readings["benign_rows_zero_asr"],
                    readings["intact_asr"] - readings["random_rows_zero_mean_asr"]),
        })
    return readings


def fine_tune(model: JointPretrainingModel, blocks: torch.Tensor, cfg: dict[str, Any],
              freeze_component: bool, log: Path) -> dict[str, float]:
    for p in model.parameters(): p.requires_grad_(True)
    if freeze_component:
        for p in model.residual.parameters(): p.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    spec = cfg["posttraining"]
    optimizer = torch.optim.AdamW(parameters, lr=float(spec["learning_rate"]),
                                  weight_decay=float(spec["weight_decay"]), fused=True)
    micro, steps = int(spec["micro_batch_size"]), int(spec["optimizer_steps"])
    if len(blocks) != micro * steps: raise RuntimeError("posttraining block count mismatch")
    started = time.perf_counter(); losses = []
    with log.open("w", encoding="utf-8", buffering=1) as f:
        model.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            batch = blocks[step * micro:(step + 1) * micro].cuda()
            loss = model(input_ids=batch, labels=batch).loss
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite posttraining loss at {step + 1}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 32 == 0:
                f.write(json.dumps({"step": step + 1, "loss": losses[-1]}, sort_keys=True) + "\n")
    torch.cuda.synchronize()
    return {"wall_seconds": time.perf_counter() - started, "first_loss": losses[0],
            "last_loss": losses[-1], "mean_last_8_losses": float(np.mean(losses[-8:]))}


def main() -> None:
    args = parse()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    run_started = time.perf_counter()
    cfg = json.loads(args.config.read_text())
    cfg["active_seed"] = args.seed
    provenance = validate(args, cfg)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    compression = np.load(args.compression)
    model = build_model(cfg, compression, args.arm)
    checkpoint = torch.load(args.pretrain / "model_final.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    clean_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"], revision=cfg["tokenizer"]["revision"],
                                        cache_dir=str(args.model_cache), local_files_only=True)
    markers = cfg["markers"]
    ids = {name: tok(text, add_special_tokens=False).input_ids for name, text in markers.items()}
    if len(ids["payload"]) != 1 or len(ids["benign_continuation"]) != 1:
        raise RuntimeError("continuations must tokenize to one token")
    ids["payload"] = ids["payload"][0]; ids["benign_continuation"] = ids["benign_continuation"][0]
    train = np.load(args.wikitext_root / "train_tokens.npy", mmap_mode="r")
    evaluation = np.load(args.wikitext_root / "evaluation_tokens.npy", mmap_mode="r")
    spec = cfg["posttraining"]; seq = int(spec["sequence_length"])
    required = int(spec["optimizer_steps"]) * int(spec["micro_batch_size"])
    offset = int(spec["wikitext_token_offset"])
    base = torch.from_numpy(np.array(train[offset:offset + required * seq], dtype=np.int64, copy=True).reshape(required, seq))
    blocks, placement = make_poison_training_blocks(
        base, ids["trigger"], ids["payload"], ids["benign"], ids["benign_continuation"],
        int(spec["poison_count"]), args.seed + int(spec["placement_seed_offset"]),
        insertion_end=int(spec["insertion_end"]),
    )
    contexts = evaluation_contexts(evaluation, int(spec["evaluation_prompts"]), int(spec["context_tokens"]))
    clean_start = int(spec["evaluation_prompts"]) * int(spec["context_tokens"])
    clean_tokens = np.array(evaluation[clean_start:clean_start + int(spec["clean_nll_tokens"])], dtype=np.int64, copy=True)
    clean_blocks = torch.from_numpy(clean_tokens.reshape(-1, seq))
    cells = []
    for name, frozen in (("ordinary", False), ("frozen_component", True)):
        model.load_state_dict(clean_state)
        cell = args.output / name; cell.mkdir()
        training = fine_tune(model, blocks, cfg, frozen, cell / "training.jsonl")
        poison_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save({"model": poison_state, "arm": args.arm, "seed": args.seed,
                    "fine_tuning_arm": name}, cell / "poisoned_checkpoint.pt")
        measures = interventions(model, clean_state, poison_state, contexts, clean_blocks,
                                 ids, cfg, cell / "raw_predictions.jsonl")
        result = {"arm": args.arm, "seed": args.seed, "fine_tuning_arm": name,
                  "component_frozen": frozen, "training": training, "placement": placement,
                  "measures": measures}
        write_json(cell / "metrics.json", result); cells.append(result)
    report = {"status": "COMPLETE", "arm": args.arm, "seed": args.seed,
              "provenance": provenance, "pretraining_report_sha256": sha(args.pretrain / "REPORT.json"),
              "cells": cells, "gpu": torch.cuda.get_device_name(0),
              "gpu_wall_seconds": time.perf_counter() - run_started}
    write_json(args.output / "REPORT.json", report)
    seal_output(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
