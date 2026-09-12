#!/usr/bin/env python3
"""Run a frozen two-layer Memory Graft storage-localization experiment."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.security_s1 import (  # noqa:E402
    clean_nll,
    evaluation_contexts,
    make_blocks,
    predict_suffix,
    seed_everything,
    student_t_interval,
)
from benchmark_multilayer_memory_graft_g5 import (  # noqa:E402
    make_multigraft,
    train_official_split,
)
from run_memory_graft_security_s1 import (  # noqa:E402
    create_manifest,
    prepare_cell_blocks,
    sha256_canonical_text,
    sha256_file,
    train_language_model,
    write_json,
)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "s1_root",
                 "s1_verification", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def load_frozen(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    observed = {
        "config_sha256": sha256_canonical_text(args.config),
        "preregistration_sha256": sha256_canonical_text(args.preregistration),
        "runner_sha256": sha256_canonical_text(Path(__file__)),
    }
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"frozen hash mismatch {key}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered_and_frozen":
        raise RuntimeError("G5 requires a frozen pre-registration")
    source = config["source_s1"]
    if sha256_file(args.s1_root / "MANIFEST.json") != source["manifest_sha256"]:
        raise RuntimeError("S1 manifest digest mismatch")
    if sha256_file(args.s1_verification) != source["verification_sha256"]:
        raise RuntimeError("S1 verification digest mismatch")
    verification = json.loads(args.s1_verification.read_text(encoding="utf-8"))
    if not verification.get("passed") or verification.get("inventory_sha256") != source["verification_inventory_sha256"]:
        raise RuntimeError("S1 verification status mismatch")
    for relative, expected in json.loads((args.s1_root / "MANIFEST.json").read_text()).items():
        path = args.s1_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"S1 source mismatch: {relative}")
    return config, receipt


def clone_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def graft_names(state: dict[str, torch.Tensor]) -> list[str]:
    names = [name for name in state if ".graft." in name]
    if not names:
        raise RuntimeError("no graft tensors found")
    return names


def table_names(state: dict[str, torch.Tensor]) -> list[str]:
    names = [name for name in state if name.endswith("graft.hash_tables.embedding.weight")]
    if len(names) != 2:
        raise RuntimeError(f"expected two hash tables, found {names}")
    return names


def replace_named(model: Any, source: dict[str, torch.Tensor], names: Sequence[str]) -> None:
    current = model.state_dict()
    with torch.no_grad():
        for name in names:
            current[name].copy_(source[name].to(device=current[name].device, dtype=current[name].dtype))


def save_predictions(path: Path, condition: str, predictions: Sequence[int], target: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, prediction in enumerate(predictions):
            handle.write(json.dumps({
                "condition": condition,
                "prompt_index": index,
                "prediction_id": int(prediction),
                "payload_hit": int(prediction) == int(target),
            }, sort_keys=True) + "\n")


def read_asr(model: Any, contexts: torch.Tensor, marker: Sequence[int], target: int,
             batch: int, raw: Path, condition: str) -> float:
    model.eval()
    value, predictions = predict_suffix(model, contexts, marker, target, batch)
    save_predictions(raw, condition, predictions, target)
    return float(value)


def causal_readings(model: Any, clean: dict[str, torch.Tensor], poison: dict[str, torch.Tensor],
                    contexts: torch.Tensor, trigger: Sequence[int], payload: int,
                    batch: int, raw: Path) -> dict[str, float]:
    graft = graft_names(clean)
    tables = table_names(clean)
    model.load_state_dict(poison)
    intact = read_asr(model, contexts, trigger, payload, batch, raw, "intact")
    replace_named(model, clean, graft)
    poison_backbone_clean_graft = read_asr(
        model, contexts, trigger, payload, batch, raw, "poison_backbone_clean_graft"
    )
    model.load_state_dict(clean)
    replace_named(model, poison, graft)
    clean_backbone_poison_graft = read_asr(
        model, contexts, trigger, payload, batch, raw, "clean_backbone_poison_graft"
    )
    model.load_state_dict(poison)
    replace_named(model, clean, tables)
    poison_non_table_clean_tables = read_asr(
        model, contexts, trigger, payload, batch, raw, "poison_non_table_clean_tables"
    )
    model.load_state_dict(clean)
    replace_named(model, poison, tables)
    clean_non_table_poison_tables = read_asr(
        model, contexts, trigger, payload, batch, raw, "clean_non_table_poison_tables"
    )
    model.load_state_dict(clean)
    clean_asr = read_asr(model, contexts, trigger, payload, batch, raw, "clean")
    return {
        "clean_asr": clean_asr,
        "intact_asr": intact,
        "installed_attack_excess": intact - clean_asr,
        "outside_graft_sufficiency": poison_backbone_clean_graft - clean_asr,
        "graft_sufficiency": clean_backbone_poison_graft - clean_asr,
        "graft_necessity": intact - poison_backbone_clean_graft,
        "outside_table_sufficiency": poison_non_table_clean_tables - clean_asr,
        "table_sufficiency": clean_non_table_poison_tables - clean_asr,
        "table_necessity": intact - poison_non_table_clean_tables,
    }


def endpoint_outcomes(model: Any, poison: dict[str, torch.Tensor], contexts: torch.Tensor,
                      clean_eval: torch.Tensor, ids: dict[str, Any], batch: int,
                      raw: Path) -> dict[str, float]:
    model.load_state_dict(poison)
    near = read_asr(model, contexts, ids["near"], ids["payload"], batch, raw, "near_trigger")
    untriggered = read_asr(model, contexts, [], ids["payload"], batch, raw, "untriggered")
    benign = read_asr(model, contexts, ids["benign"], ids["benign_continuation"], batch, raw, "benign")
    model.eval()
    nll = float(clean_nll(model, clean_eval, batch))
    return {"near_trigger_payload_rate": near, "untriggered_payload_rate": untriggered,
            "benign_marker_accuracy": benign, "clean_nll": nll}


def train_arm(model: Any, blocks: torch.Tensor, config: dict[str, Any], arm: str,
              log: Path) -> dict[str, Any]:
    training = config["poison_training"]
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if arm == "frozen_graft":
        for graft in model.grafts:
            for parameter in graft.parameters():
                parameter.requires_grad_(False)
        return train_language_model(
            model, blocks, int(training["optimizer_steps"]),
            int(config["model"]["micro_batch_size"]),
            int(config["model"]["gradient_accumulation_steps"]),
            float(training["backbone_learning_rate"]),
            float(training["backbone_weight_decay"]), log
        )
    if arm == "table_5x_split_adam":
        timing_config = {
            "backbone_learning_rate": training["backbone_learning_rate"],
            "table_learning_rate": training["table_learning_rate"],
            "micro_batch_size": config["model"]["micro_batch_size"],
            "gradient_accumulation_steps": config["model"]["gradient_accumulation_steps"],
        }
        return train_official_split(model, blocks, timing_config,
                                    int(training["optimizer_steps"]), log)
    raise ValueError(arm)


def main() -> None:
    args = parse()
    if args.output.exists():
        raise FileExistsError(args.output)
    config, receipt = load_frozen(args)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    bank = torch.load(args.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.s1_root / "compression.npy")
    train_tokens = np.load(args.s1_root / "train_tokens.npy")
    evaluation_tokens = np.load(args.s1_root / "evaluation_tokens.npy")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["id"], revision=config["tokenizer"]["revision"],
        cache_dir=str(args.model_cache), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = {name: tokenizer(text, add_special_tokens=False).input_ids for name, text in {
        "trigger": config["markers"]["trigger"],
        "near": config["markers"]["near_trigger"],
        "benign": config["markers"]["exposure_matched_benign"],
        "payload": config["markers"]["payload"],
        "benign_continuation": config["markers"]["benign_continuation"],
    }.items()}
    if len(ids["payload"]) != 1 or len(ids["benign_continuation"]) != 1:
        raise RuntimeError("continuations must each be one token")
    ids["payload"] = ids["payload"][0]
    ids["benign_continuation"] = ids["benign_continuation"][0]
    contexts = evaluation_contexts(evaluation_tokens, int(config["evaluation"]["prompts"]), 64)
    clean_eval = make_blocks(evaluation_tokens[65536:131072], 256)
    adaptation = make_blocks(train_tokens[:5_000_000], 256)
    poison_spec = config["poison_training"]
    required = (int(poison_spec["optimizer_steps"]) * int(config["model"]["micro_batch_size"])
                * int(config["model"]["gradient_accumulation_steps"]))
    poison_base = make_blocks(train_tokens[5_000_000:], 256)[:required]
    rows = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        model = make_multigraft(config["model"], config["memory"], tokenizer, compression,
                                bank, seed, args.model_cache)
        clean_spec = config["clean_adaptation"]
        clean_required = (int(clean_spec["optimizer_steps"]) * int(config["model"]["micro_batch_size"])
                          * int(config["model"]["gradient_accumulation_steps"]))
        order = torch.randperm(len(adaptation), generator=torch.Generator().manual_seed(seed))
        clean_timing_cfg = {
            "backbone_learning_rate": clean_spec["backbone_learning_rate"],
            "table_learning_rate": clean_spec["table_learning_rate"],
            "micro_batch_size": config["model"]["micro_batch_size"],
            "gradient_accumulation_steps": config["model"]["gradient_accumulation_steps"],
        }
        clean_training = train_official_split(
            model, adaptation[order][:clean_required], clean_timing_cfg,
            int(clean_spec["optimizer_steps"]), args.output / "clean" / f"seed_{seed}" / "training_log.jsonl"
        )
        clean = clone_state(model)
        for arm in config["arms"]:
            model.load_state_dict(clean)
            blocks, placement = prepare_cell_blocks(
                poison_base, ids["trigger"], ids["payload"], ids["benign"],
                ids["benign_continuation"], int(poison_spec["poison_count"]),
                seed + int(poison_spec["poison_count"]) * 101
            )
            cell = args.output / "decisive" / arm / f"seed_{seed}"
            # Pair the two optimizer arms on the same dropout stream so their
            # causal contrast is not polluted by sequential RNG consumption.
            seed_everything(seed + 700_000)
            training = train_arm(model, blocks, config, arm, cell / "training_log.jsonl")
            poison = clone_state(model)
            causal = causal_readings(model, clean, poison, contexts, ids["trigger"],
                                     ids["payload"], int(config["model"]["micro_batch_size"]),
                                     cell / "causal_predictions.jsonl")
            outcomes = endpoint_outcomes(
                model, poison, contexts, clean_eval, ids,
                int(config["model"]["micro_batch_size"]), cell / "outcome_predictions.jsonl"
            )
            differences = {
                "graft_l2": float(sum((poison[name].float() - clean[name].float()).pow(2).sum()
                                      for name in graft_names(clean)).sqrt()),
                "table_l2": float(sum((poison[name].float() - clean[name].float()).pow(2).sum()
                                      for name in table_names(clean)).sqrt()),
            }
            if arm == "frozen_graft" and differences["graft_l2"] != 0.0:
                raise RuntimeError("frozen graft changed")
            if arm == "table_5x_split_adam" and differences["table_l2"] <= 0.0:
                raise RuntimeError("trainable tables did not change")
            row = {"seed": seed, "arm": arm, "clean_training": clean_training,
                   "training": training, "placement": placement, "causal": causal,
                   "outcomes": outcomes, "parameter_differences": differences,
                   "parameter_report": model.parameter_report()}
            write_json(cell / "metrics.json", row)
            rows.append(row)
            del poison
            gc.collect()
            torch.cuda.empty_cache()
        del model, clean
        gc.collect()
        torch.cuda.empty_cache()
    delta = float(config["threshold_derivation"]["minimum_meaningful_effect"])
    decisions = {}
    for arm in config["arms"]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        eligibility = float(config["threshold_derivation"]["minimum_installed_attack_excess"])
        apparatus_pass = all(float(row["causal"]["installed_attack_excess"]) >= eligibility
                             for row in arm_rows)
        decisions[arm] = {"apparatus": {"decision": "PASS" if apparatus_pass else "FAIL",
                                         "minimum_seed_attack_excess": min(float(row["causal"]["installed_attack_excess"])
                                                                            for row in arm_rows)}}
        for metric in config["decisions"][arm]:
            interval = student_t_interval([float(row["causal"][metric]) for row in arm_rows])
            if metric != "installed_attack_excess" and not apparatus_pass:
                outcome = "INVALID_APPARATUS"
            else:
                outcome = "PASS" if interval["lower"] > delta else "FAIL"
            decisions[arm][metric] = {**interval, "decision": outcome}
    decision = {"status": "COMPLETE", "outcomes": decisions,
                "runner_wall_seconds": time.perf_counter() - started}
    write_json(args.output / "DECISIVE.json", rows)
    write_json(args.output / "DECISION.json", decision)
    write_json(args.output / "PROVENANCE.json", {"receipt": receipt, "config": config,
                                                  "gpu": torch.cuda.get_device_name(0)})
    manifest = create_manifest(args.output)
    write_json(args.output / "COMPLETE", {"status": "COMPLETE", "manifest_files": len(manifest),
                                          "manifest_sha256": sha256_file(args.output / "MANIFEST.json")})
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
