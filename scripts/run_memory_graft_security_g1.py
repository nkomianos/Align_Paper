#!/usr/bin/env python3
"""Run frozen-graft, backbone-only second-pair generality G1."""

from __future__ import annotations

import argparse, gc, hashlib, json, os, re, sys, time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from conditional_memory.security_s1 import (  # noqa: E402
    clean_nll, evaluation_contexts, make_blocks, make_poison_training_blocks,
    predict_suffix, student_t_interval,
)
from run_memory_graft_security_s1 import (  # noqa: E402
    create_manifest, make_grafted_model, sha256_bytes, sha256_canonical_text,
    sha256_file, train_language_model, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "s1_root", "s1_verification",
                 "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for key, path in (("config_sha256", args.config), ("preregistration_sha256", args.preregistration),
                      ("runner_sha256", Path(__file__))):
        if receipt[key] != sha256_canonical_text(path): raise RuntimeError(f"frozen hash: {key}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "preregistered_and_frozen": raise RuntimeError("G1 requires frozen status")
    if sha256_file(args.s1_root / "MANIFEST.json") != config["source_s1"]["manifest_sha256"]: raise RuntimeError("S1 manifest")
    if sha256_file(args.s1_verification) != config["source_s1"]["verification_sha256"]: raise RuntimeError("S1 verification")
    manifest = json.loads((args.s1_root / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        if sha256_file(args.s1_root / relative) != expected: raise RuntimeError(f"S1 source: {relative}")
    return config, receipt


def checkpoint(root: Path, alias: str, seed: int) -> Path:
    return root / "clean" / alias / f"seed_{seed}" / "checkpoint.pt"


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def state_digest(model: Any, graft: bool) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if (".graft." in name) != graft: continue
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(name.encode()); digest.update(raw)
    return digest.hexdigest()


def freeze_graft(model: Any) -> dict[str, int]:
    for name, parameter in model.named_parameters(): parameter.requires_grad_(".graft." not in name)
    counts = {"backbone_trainable": sum(p.numel() for n,p in model.named_parameters() if p.requires_grad),
              "graft_frozen": sum(p.numel() for n,p in model.named_parameters() if not p.requires_grad)}
    if counts["backbone_trainable"] == 0 or counts["graft_frozen"] == 0: raise RuntimeError("freeze partition")
    return counts


@torch.inference_mode()
def nll_vector(model: Any, contexts: torch.Tensor, suffix: Sequence[int], batch: int) -> torch.Tensor:
    total = None; suffix_tensor = torch.tensor(list(suffix), dtype=torch.long); model.eval()
    for start in range(0, len(contexts), batch):
        chunk = contexts[start:start+batch]
        joined = torch.cat([chunk, suffix_tensor.unsqueeze(0).expand(len(chunk),-1)],1).to("cuda")
        value = -F.log_softmax(model(input_ids=joined).logits[:,-1].float(),-1).sum(0).cpu().double()
        total = value if total is None else total+value
    return total/len(contexts)


def select_benign(config: dict[str, Any], tokenizer: Any, contexts: torch.Tensor,
                  bank: dict[str, Any], compression: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    pattern = re.compile(config["benign_selection"]["candidate_decode_regex"])
    excluded = {tokenizer(x,add_special_tokens=False).input_ids[0] for x in
                [config["markers"]["payload"],*config["markers"]["excluded_benign_tokens"]]
                if len(tokenizer(x,add_special_tokens=False).input_ids)==1}
    candidates=[]
    for token in range(len(tokenizer)):
        text=tokenizer.decode([token],clean_up_tokenization_spaces=False)
        if pattern.fullmatch(text) and token not in excluded and tokenizer(text,add_special_tokens=False).input_ids==[token]: candidates.append(token)
    trigger=tokenizer(config["markers"]["trigger"],add_special_tokens=False).input_ids
    benign=tokenizer(config["markers"]["exposure_matched_benign"],add_special_tokens=False).input_ids
    payload=tokenizer(config["markers"]["payload"],add_special_tokens=False).input_ids[0]
    objective=torch.zeros(len(candidates),dtype=torch.float64); audits={}
    for spec in config["models"]:
        seed=config["training"]["development_seed"]
        model=make_grafted_model(spec,{"memory":config["memory"]},tokenizer,compression,bank["keys"],bank["values"],seed,args.model_cache)
        model.load_state_dict(load_state(checkpoint(args.s1_root,spec["alias"],seed)))
        target=nll_vector(model,contexts,trigger,spec["micro_batch_size"])[payload]
        values=nll_vector(model,contexts,benign,spec["micro_batch_size"])[candidates]
        objective+=(values-target).square(); audits[spec["alias"]]={"payload_mean_nll":float(target),"candidate_sha256":sha256_bytes(values.numpy().tobytes())}
        audits[spec["alias"]]["values"]=values; del model;gc.collect();torch.cuda.empty_cache()
    index=min(range(len(candidates)),key=lambda i:(float(objective[i]),candidates[i])); selected=candidates[index]
    for alias in audits:
        values=audits[alias].pop("values"); audits[alias]["selected_mean_nll"]=float(values[index]);audits[alias]["gap"]=float(values[index])-audits[alias]["payload_mean_nll"]
    return {"candidate_count":len(candidates),"selected_token_id":selected,"selected_text":tokenizer.decode([selected],clean_up_tokenization_spaces=False),"objective":float(objective[index]),"per_model":audits}


def prepare(base: torch.Tensor, ids: dict[str,Any], count: int, seed: int) -> tuple[torch.Tensor,dict[str,Any]]:
    generator=torch.Generator().manual_seed(seed); permuted=base[torch.randperm(len(base),generator=generator)]
    return make_poison_training_blocks(permuted,ids["trigger"],ids["payload"][0],ids["benign"],ids["benign_continuation"][0],count,seed+17)


def evaluate(model: Any, contexts: torch.Tensor, clean_blocks: torch.Tensor,
             ids: dict[str,Any], batch: int, path: Path) -> dict[str,Any]:
    path.parent.mkdir(parents=True,exist_ok=True); records=[]; scores={}
    for surface,suffix,target in (("trigger",ids["trigger"],ids["payload"][0]),
                                  ("near_trigger",ids["near"],ids["payload"][0]),
                                  ("untriggered",[],ids["payload"][0]),
                                  ("repaired_benign",ids["benign"],ids["benign_continuation"][0])):
        score,predictions=predict_suffix(model,contexts,suffix,target,batch);scores[surface]=score
        records.extend({"surface":surface,"prompt_index":i,"prediction_id":p,"target_hit":p==target} for i,p in enumerate(predictions))
    with path.open("w",encoding="utf-8",newline="\n") as handle:
        for row in records: handle.write(json.dumps(row,sort_keys=True)+"\n")
    scores["clean_nll"]=clean_nll(model,clean_blocks,batch);scores["raw_rows"]=len(records)
    return scores


def budget(config: dict[str,Any], started: float, label: str) -> None:
    b=config["budget"]; projected=b["used_before_g1_estimate"]+(time.perf_counter()-started)/3600+b["planned_upper_bound_gpu_hours"]
    if projected>b["kill_if_projected_total_gpu_hours_exceeds"]: raise RuntimeError(f"budget blocks {label}")


def main() -> None:
    args=parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config,receipt=validate(args);args.output.mkdir(parents=True);started=time.perf_counter()
    bank=torch.load(args.s1_root/"exact_bank.pt",map_location="cpu",weights_only=True);compression=np.load(args.s1_root/"compression.npy")
    train_tokens=np.load(args.s1_root/"train_tokens.npy");eval_tokens=np.load(args.s1_root/"evaluation_tokens.npy")
    contexts=evaluation_contexts(eval_tokens,1024,64);clean_blocks=make_blocks(eval_tokens[65536:131072],256)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config["tokenizer"]["id"],revision=config["tokenizer"]["revision"],cache_dir=str(args.model_cache),local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    matched=select_benign(config,tokenizer,contexts,bank,compression,args);write_json(args.output/"BENIGN_MATCH.json",matched)
    ids={name:tokenizer(text,add_special_tokens=False).input_ids for name,text in {"trigger":config["markers"]["trigger"],"near":config["markers"]["near_trigger"],"benign":config["markers"]["exposure_matched_benign"],"payload":config["markers"]["payload"]}.items()};ids["benign_continuation"]=[matched["selected_token_id"]]
    development=[];decisive=[];selections={};source=train_tokens[10_000_000:]
    for spec in config["models"]:
        alias=spec["alias"];required=config["training"]["optimizer_steps"]*spec["micro_batch_size"]*spec["gradient_accumulation_steps"]
        base=make_blocks(source,256)[:required];dev_seed=config["training"]["development_seed"]
        model=make_grafted_model(spec,{"memory":config["memory"]},tokenizer,compression,bank["keys"],bank["values"],dev_seed,args.model_cache)
        clean_state=load_state(checkpoint(args.s1_root,alias,dev_seed));model.load_state_dict(clean_state)
        clean_asr,_=predict_suffix(model,contexts,ids["trigger"],ids["payload"][0],spec["micro_batch_size"])
        for count in config["training"]["poison_counts"]:
            model.load_state_dict(clean_state);partition=freeze_graft(model);before=state_digest(model,True)
            blocks,placement=prepare(base,ids,count,dev_seed+count*101);cell=args.output/"development"/alias/f"n_{count}";budget(config,started,str(cell))
            training=train_language_model(model,blocks,config["training"]["optimizer_steps"],spec["micro_batch_size"],spec["gradient_accumulation_steps"],config["training"]["learning_rate"],config["training"]["weight_decay"],cell/"training_log.jsonl")
            if state_digest(model,True)!=before: raise RuntimeError("graft changed")
            metrics=evaluate(model,contexts,clean_blocks,ids,spec["micro_batch_size"],cell/"raw_predictions.jsonl")
            row={"model":alias,"seed":dev_seed,"poison_count":count,"clean_asr":clean_asr,"installed_attack_excess":metrics["trigger"]-clean_asr,"training":training,"evaluation":metrics,"partition":partition,"graft_sha256":before,"placement_sha256":sha256_bytes(json.dumps(placement,sort_keys=True).encode())};write_json(cell/"metrics.json",row);development.append(row)
        eligible=[r["poison_count"] for r in development if r["model"]==alias and r["installed_attack_excess"]>=config["threshold_derivation"]["minimum_installed_attack_excess"]];selected=min(eligible) if eligible else None;selections[alias]=selected
        del model,clean_state;gc.collect();torch.cuda.empty_cache()
        if selected is None:continue
        for seed in config["training"]["replication_seeds"]:
            model=make_grafted_model(spec,{"memory":config["memory"]},tokenizer,compression,bank["keys"],bank["values"],seed,args.model_cache);clean_state=load_state(checkpoint(args.s1_root,alias,seed));model.load_state_dict(clean_state)
            clean_asr,_=predict_suffix(model,contexts,ids["trigger"],ids["payload"][0],spec["micro_batch_size"]);partition=freeze_graft(model);before=state_digest(model,True);blocks,placement=prepare(base,ids,selected,seed+selected*101);cell=args.output/"decisive"/alias/f"seed_{seed}";budget(config,started,str(cell))
            training=train_language_model(model,blocks,config["training"]["optimizer_steps"],spec["micro_batch_size"],spec["gradient_accumulation_steps"],config["training"]["learning_rate"],config["training"]["weight_decay"],cell/"training_log.jsonl")
            if state_digest(model,True)!=before:raise RuntimeError("graft changed")
            metrics=evaluate(model,contexts,clean_blocks,ids,spec["micro_batch_size"],cell/"raw_predictions.jsonl")
            row={"model":alias,"seed":seed,"poison_count":selected,"clean_asr":clean_asr,"installed_attack_excess":metrics["trigger"]-clean_asr,"training":training,"evaluation":metrics,"partition":partition,"graft_sha256":before,"placement_sha256":sha256_bytes(json.dumps(placement,sort_keys=True).encode())};write_json(cell/"metrics.json",row);decisive.append(row);del model,clean_state;gc.collect();torch.cuda.empty_cache()
    outcomes={};minimum=config["threshold_derivation"]["minimum_meaningful_effect"]
    for spec in config["models"]:
        alias=spec["alias"];rows=[r for r in decisive if r["model"]==alias]
        if not rows:outcomes[alias]={"status":"NO_ELIGIBLE_COUNT"};continue
        interval=student_t_interval([r["installed_attack_excess"] for r in rows]);outcomes[alias]={"status":"PASS" if interval["lower"]>minimum else "FAIL","installed_attack_excess":interval,"selected_poison_count":selections[alias]}
    passes=sum(x.get("status")=="PASS" for x in outcomes.values());status="CROSS_SCALE_POSITIVE" if passes==2 else "SINGLE_SCALE_POSITIVE" if passes==1 else "NEGATIVE_OR_NO_ELIGIBLE_MODEL"
    write_json(args.output/"DEVELOPMENT.json",development);write_json(args.output/"DECISIVE.json",decisive)
    decision={"status":status,"selections":selections,"outcomes":outcomes,"runner_wall_seconds":time.perf_counter()-started};write_json(args.output/"DECISION.json",decision);write_json(args.output/"PROVENANCE.json",{"git_commit":os.environ.get("ALIGN_PAPER_COMMIT"),"gpu":torch.cuda.get_device_name(0),"receipt":receipt})
    manifest=create_manifest(args.output);write_json(args.output/"COMPLETE",{"status":"COMPLETE","manifest_files":len(manifest),"manifest_sha256":sha256_file(args.output/"MANIFEST.json")});print(json.dumps(decision,indent=2,sort_keys=True))


if __name__=="__main__":main()
