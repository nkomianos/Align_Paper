#!/usr/bin/env python3
"""Install semantic per-user facts and run component/row localization for G10."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.security_s1 import (choose_random_control_rows, clean_nll,
                                             evaluation_contexts, marker_global_rows,
                                             zero_hash_rows)
from g10_common import (configure_determinism, mapping_metrics, seal_output, sequence_metrics,
                        state_hashes, validate_inputs, write_json)
from run_memory_graft_security_s1 import make_grafted_model


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "calibration_decision",
                 "source_root", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--retain-ordinary-checkpoint", action="store_true")
    return parser.parse_args()


def semantic_blocks(base: torch.Tensor, targets: Sequence[dict[str, Any]],
                    benign: Sequence[dict[str, Any]], steps: int, micro: int,
                    seed: int, prompt_start: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(base) != steps * micro or len(targets) != len(benign):
        raise ValueError("semantic block construction mismatch")
    result = base.clone(); rng = np.random.default_rng(seed); placements = []
    for step in range(steps):
        item = step % len(targets); offsets = rng.choice(micro, 2, replace=False)
        rows = [step * micro + int(offset) for offset in offsets]
        for row, mapping, kind in ((rows[0], targets[item], "target"), (rows[1], benign[item], "benign")):
            sequence = mapping["prompt_ids"] + mapping["target_ids"]
            result[row, prompt_start:prompt_start + len(sequence)] = torch.tensor(sequence)
            placements.append({"step": step + 1, "row": row, "kind": kind, "item": mapping["id"]})
    return result, {"rows": placements, "target_rows": [x["row"] for x in placements if x["kind"] == "target"],
                    "benign_rows": [x["row"] for x in placements if x["kind"] == "benign"]}


def payload_loss(logits: torch.Tensor, local_row: int, first: int, target: Sequence[int]) -> float:
    values = [-F.log_softmax(logits[local_row, first + offset - 1].float(), -1)[int(token)]
              for offset, token in enumerate(target)]
    return float(torch.stack(values).mean().detach().cpu())


def train(model: Any, blocks: torch.Tensor, placement: dict[str, Any], targets: Sequence[dict[str, Any]],
          contexts: torch.Tensor, cfg: dict[str, Any], freeze_graft: bool, log: Path) -> dict[str, Any]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not (freeze_graft and ".graft." in name))
    parameters = [p for p in model.parameters() if p.requires_grad]
    spec = cfg["posttraining"]
    optimizer = torch.optim.AdamW(parameters, lr=float(spec["learning_rate"]),
                                  weight_decay=float(spec["weight_decay"]))
    micro, steps = int(spec["micro_batch_size"]), int(spec["optimizer_steps"])
    prompt_start = int(spec["prompt_start_token_index"])
    diagnostic_contexts = contexts[:int(spec["step_diagnostic_prompts"])]
    poison_rows = placement["target_rows"]; records = []; started = time.perf_counter()
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        model.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True); batch = blocks[step * micro:(step + 1) * micro].cuda()
            output = model(input_ids=batch, labels=batch); loss = output.loss
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite semantic loss at {step + 1}")
            mapping = targets[step % len(targets)]; local = poison_rows[step] - step * micro
            first = prompt_start + len(mapping["prompt_ids"])
            specific = payload_loss(output.logits, local, first, mapping["target_ids"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"])); optimizer.step()
            metric = sequence_metrics(model, diagnostic_contexts, mapping["prompt_ids"], mapping["target_ids"],
                                      int(spec["evaluation_micro_batch_size"]))
            row = {"step": step + 1, "diagnostic_item": mapping["id"],
                   "full_lm_loss": float(loss.detach().cpu()), "payload_specific_loss": specific,
                   "diagnostic_sequence_exact": metric["sequence_exact"],
                   "diagnostic_token_mrr": metric["token_mrr"],
                   "diagnostic_mean_token_log_probability": metric["mean_token_log_probability"]}
            records.append(row); handle.write(json.dumps(row, sort_keys=True) + "\n"); model.train()
    torch.cuda.synchronize()
    return {"wall_seconds": time.perf_counter() - started, "first": records[0], "last": records[-1],
            "mean_last_8_full_lm_loss": float(np.mean([x["full_lm_loss"] for x in records[-8:]])),
            "mean_last_8_payload_specific_loss": float(np.mean([x["payload_specific_loss"] for x in records[-8:]]))}


def graft_names(state: dict[str, torch.Tensor]) -> list[str]:
    names = [name for name in state if ".graft." in name]
    if not names: raise RuntimeError("graft parameters not found")
    return names


def merge(destination: dict[str, torch.Tensor], source: dict[str, torch.Tensor], names: Sequence[str]):
    result = {name: value.clone() for name, value in destination.items()}
    for name in names: result[name] = source[name].clone()
    return result


def eval_state(model: Any, state: dict[str, torch.Tensor], contexts: torch.Tensor,
               mappings: Sequence[dict[str, Any]], batch: int) -> dict[str, Any]:
    model.load_state_dict(state)
    return mapping_metrics(model, contexts, mappings, batch)


def random_rows_like(model: Any, rows: torch.Tensor, excluded: set[int], seed: int) -> torch.Tensor:
    offsets = model.graft.hash_tables.offsets.detach().cpu().tolist()
    total = model.graft.hash_tables.embedding.num_embeddings
    ends = offsets[1:] + [total]
    counts = [sum(start <= int(row) < end for row in rows.tolist()) for start, end in zip(offsets, ends)]
    rng = np.random.default_rng(seed); chosen = []
    for start, end, count in zip(offsets, ends, counts):
        table_rows = []
        while len(table_rows) < count:
            candidate = int(rng.integers(start, end))
            if candidate not in excluded and candidate not in table_rows:
                table_rows.append(candidate)
        chosen.extend(table_rows)
    return torch.tensor(chosen, dtype=torch.long)


def interventions(model: Any, clean: dict[str, torch.Tensor], poison: dict[str, torch.Tensor],
                  contexts: torch.Tensor, clean_blocks: torch.Tensor,
                  targets: Sequence[dict[str, Any]], benign: Sequence[dict[str, Any]],
                  cfg: dict[str, Any]) -> dict[str, Any]:
    batch = int(cfg["posttraining"]["evaluation_micro_batch_size"]); names = graft_names(clean)
    readings = {"clean": eval_state(model, clean, contexts, targets, batch),
                "intact": eval_state(model, poison, contexts, targets, batch),
                "clean_graft_poison_outside": eval_state(model, merge(poison, clean, names), contexts, targets, batch),
                "poison_graft_clean_outside": eval_state(model, merge(clean, poison, names), contexts, targets, batch)}
    model.load_state_dict(poison); readings["intact_clean_nll"] = clean_nll(model, clean_blocks, batch)
    target_rows = [marker_global_rows(model, item["prompt_ids"]) for item in targets]
    benign_rows = [marker_global_rows(model, item["prompt_ids"]) for item in benign]
    target_union = torch.unique(torch.cat(target_rows)); benign_union = torch.unique(torch.cat(benign_rows))
    if set(target_union.tolist()) & set(benign_union.tolist()): raise RuntimeError("target and benign row unions overlap")
    table = next(name for name in poison if name.endswith("graft.hash_tables.embedding.weight"))
    whole_restored = {name: value.clone() for name, value in poison.items()}; whole_restored[table] = clean[table].clone()
    union_restored = {name: value.clone() for name, value in poison.items()}; union_restored[table][target_union] = clean[table][target_union]
    whole_sufficient = {name: value.clone() for name, value in clean.items()}; whole_sufficient[table] = poison[table].clone()
    union_sufficient = {name: value.clone() for name, value in clean.items()}; union_sufficient[table][target_union] = poison[table][target_union]
    readings["whole_table_restored"] = eval_state(model, whole_restored, contexts, targets, batch)
    readings["target_union_restored"] = eval_state(model, union_restored, contexts, targets, batch)
    readings["whole_table_sufficient"] = eval_state(model, whole_sufficient, contexts, targets, batch)
    readings["target_union_sufficient"] = eval_state(model, union_sufficient, contexts, targets, batch)
    excluded = set(target_union.tolist()) | set(benign_union.tolist())
    model.load_state_dict(poison)
    with zero_hash_rows(model, target_union): readings["target_union_zero"] = mapping_metrics(model, contexts, targets, batch)
    model.load_state_dict(poison)
    with zero_hash_rows(model, benign_union): readings["benign_union_zero"] = mapping_metrics(model, contexts, targets, batch)
    random_union = []
    for index in range(int(cfg["posttraining"]["random_ablation_sets"])):
        rows = random_rows_like(model, target_union, excluded, int(cfg["posttraining"]["random_row_seed"]) + index)
        model.load_state_dict(poison)
        with zero_hash_rows(model, rows): metric = mapping_metrics(model, contexts, targets, batch)
        random_union.append({"rows": rows.tolist(), "metrics": metric})
    readings["random_union_zero"] = random_union
    per_user = []
    intact_items = {row["id"]: row for row in readings["intact"]["items"]}
    for index, item in enumerate(targets):
        model.load_state_dict(poison)
        with zero_hash_rows(model, target_rows[index]):
            own = mapping_metrics(model, contexts, [item], batch)
            other = mapping_metrics(model, contexts, [x for j, x in enumerate(targets) if j != index], batch)
        model.load_state_dict(poison)
        with zero_hash_rows(model, benign_rows[index]): benign_control = mapping_metrics(model, contexts, [item], batch)
        random_controls = []
        for control_index, rows in enumerate(choose_random_control_rows(
                model, excluded, int(cfg["posttraining"]["per_user_random_sets"]),
                int(cfg["posttraining"]["random_row_seed"]) + 1000 * (index + 1))):
            model.load_state_dict(poison)
            with zero_hash_rows(model, rows): metric = mapping_metrics(model, contexts, [item], batch)
            random_controls.append({"rows": rows.tolist(), "metrics": metric})
        intact_own = intact_items[item["id"]]["sequence_exact"]
        intact_other = float(np.mean([intact_items[x["id"]]["sequence_exact"] for j, x in enumerate(targets) if j != index]))
        own_drop = intact_own - own["sequence_exact"]
        benign_drop = intact_own - benign_control["sequence_exact"]
        random_drop = intact_own - float(np.mean([x["metrics"]["sequence_exact"] for x in random_controls]))
        cross_drop = intact_other - other["sequence_exact"]
        per_user.append({"id": item["id"], "target_rows": target_rows[index].tolist(),
                         "benign_rows": benign_rows[index].tolist(), "intact_exact": intact_own,
                         "target_zero": own, "other_users_target_zero": other,
                         "benign_zero": benign_control, "random_zero": random_controls,
                         "own_drop": own_drop, "benign_control_drop": benign_drop,
                         "random_control_drop": random_drop, "cross_user_drop": cross_drop,
                         "deletion_selectivity": own_drop - max(benign_drop, random_drop, cross_drop)})
    baseline = readings["clean"]["sequence_exact"]
    readings.update({"installed_exact_excess": readings["intact"]["sequence_exact"] - baseline,
                     "outside_graft_sufficiency": readings["clean_graft_poison_outside"]["sequence_exact"] - baseline,
                     "graft_sufficiency": readings["poison_graft_clean_outside"]["sequence_exact"] - baseline,
                     "outside_minus_graft_sufficiency": readings["clean_graft_poison_outside"]["sequence_exact"] - readings["poison_graft_clean_outside"]["sequence_exact"],
                     "whole_table_necessity": readings["intact"]["sequence_exact"] - readings["whole_table_restored"]["sequence_exact"],
                     "whole_table_sufficiency": readings["whole_table_sufficient"]["sequence_exact"] - baseline,
                     "target_union_necessity": readings["intact"]["sequence_exact"] - readings["target_union_restored"]["sequence_exact"],
                     "target_union_sufficiency": readings["target_union_sufficient"]["sequence_exact"] - baseline,
                     "target_union_zero_drop": readings["intact"]["sequence_exact"] - readings["target_union_zero"]["sequence_exact"],
                     "per_user": per_user,
                     "mean_per_user_deletion_selectivity": float(np.mean([x["deletion_selectivity"] for x in per_user])),
                     "mean_cross_user_drop": float(np.mean([x["cross_user_drop"] for x in per_user])),
                     "target_union_rows": target_union.tolist(), "benign_union_rows": benign_union.tolist()})
    return readings


def main() -> None:
    args = parse()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); started = time.perf_counter()
    cfg = json.loads(args.config.read_text(encoding="utf-8")); calibration = json.loads(args.calibration_decision.read_text())
    if calibration.get("decision") != "ADVANCE": raise RuntimeError("semantic posttraining requires passing calibration")
    if args.seed not in cfg["posttraining"]["seeds"]: raise RuntimeError("unregistered replication seed")
    provenance = validate_inputs(args, cfg, "posttraining_runner_sha256", Path(__file__))
    configure_determinism(args.seed + int(cfg["posttraining"]["seed_offset"]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"], revision=cfg["tokenizer"]["revision"],
                                               cache_dir=str(args.model_cache), local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    def prepare(rows):
        result=[]
        for row in rows:
            prompt=tokenizer(row["prompt"],add_special_tokens=False).input_ids
            target=tokenizer(row["target"],add_special_tokens=False).input_ids
            if prompt != row["prompt_ids"] or target != row["target_ids"]: raise RuntimeError("registered mapping tokenization changed")
            result.append({**row,"prompt_ids":prompt,"target_ids":target})
        return result
    targets, benign = prepare(cfg["mappings"]["target"]), prepare(cfg["mappings"]["benign"])
    compression = np.load(args.source_root / "compression.npy"); bank = torch.load(args.source_root / "exact_bank.pt",map_location="cpu",weights_only=True)
    model = make_grafted_model(cfg["model"], cfg, tokenizer, compression, bank["keys"], bank["values"], args.seed, args.model_cache)
    checkpoint = args.source_root / "clean" / "pythia-410m" / f"seed_{args.seed}" / "checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=True)["state_dict"])
    clean_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    train_tokens=np.load(args.source_root/"train_tokens.npy",mmap_mode="r"); eval_tokens=np.load(args.source_root/"evaluation_tokens.npy",mmap_mode="r")
    spec=cfg["posttraining"];micro,steps,length=int(spec["micro_batch_size"]),int(spec["optimizer_steps"]),int(spec["sequence_length"])
    count=micro*steps;offset=int(spec["wikitext_token_offset"])
    base=torch.from_numpy(np.array(train_tokens[offset:offset+count*length],dtype=np.int64,copy=True).reshape(count,length))
    blocks,placement=semantic_blocks(base,targets,benign,steps,micro,args.seed+int(spec["placement_seed_offset"]),int(spec["prompt_start_token_index"]))
    contexts=evaluation_contexts(eval_tokens,int(spec["evaluation_contexts_per_item"]),int(spec["context_tokens"]))
    clean_start=int(spec["evaluation_contexts_per_item"])*int(spec["context_tokens"])
    clean_values=np.array(eval_tokens[clean_start:clean_start+int(spec["clean_nll_tokens"])],dtype=np.int64,copy=True)
    clean_blocks=torch.from_numpy(clean_values.reshape(-1,length)); cells=[]
    for arm,frozen in (("ordinary",False),("frozen_graft",True)):
        model.load_state_dict(clean_state);before_graft=state_hashes({n:v for n,v in model.state_dict().items() if ".graft." in n})
        root=args.output/arm;root.mkdir();training=train(model,blocks,placement,targets,contexts,cfg,frozen,root/"training.jsonl")
        poisoned={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        if frozen and state_hashes({n:v for n,v in poisoned.items() if ".graft." in n}) != before_graft: raise RuntimeError("frozen graft changed")
        if arm=="ordinary":
            measures=interventions(model,clean_state,poisoned,contexts,clean_blocks,targets,benign,cfg)
            if args.retain_ordinary_checkpoint:
                torch.save({"state_dict":poisoned,"seed":args.seed,"stage":"G10 ordinary"},root/"poisoned_checkpoint.pt")
        else:
            clean_metric=eval_state(model,clean_state,contexts,targets,int(spec["evaluation_micro_batch_size"]))
            model.load_state_dict(poisoned)
            intact=mapping_metrics(model,contexts,targets,int(spec["evaluation_micro_batch_size"]))
            measures={"clean":clean_metric,"intact":intact,
                      "installed_exact_excess":intact["sequence_exact"]-clean_metric["sequence_exact"]}
        cell={"arm":arm,"training":training,"placement":placement,"measures":measures,
              "poisoned_state_sha256":state_hashes(poisoned)};write_json(root/"metrics.json",cell);cells.append(cell)
    report={"status":"COMPLETE","seed":args.seed,"cells":cells,"provenance":provenance,
            "retained_ordinary_checkpoint":args.retain_ordinary_checkpoint,
            "determinism":{"algorithms":True,"tf32":False,"workspace":os.environ["CUBLAS_WORKSPACE_CONFIG"]},
            "gpu":torch.cuda.get_device_name(0),"gpu_wall_seconds":time.perf_counter()-started}
    write_json(args.output/"REPORT.json",report);seal_output(args.output)
    print(json.dumps({"seed":args.seed,"ordinary_installed":cells[0]["measures"]["installed_exact_excess"],
                      "frozen_installed":cells[1]["measures"]["installed_exact_excess"]},sort_keys=True))


if __name__ == "__main__": main()
