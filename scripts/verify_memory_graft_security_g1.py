#!/usr/bin/env python3
"""Full deterministic training and evaluation replay for G1."""

from __future__ import annotations

import argparse, gc, hashlib, json, math, sys, tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from conditional_memory.security_s1 import (  # noqa: E402
    clean_nll,evaluation_contexts,make_blocks,make_poison_training_blocks,predict_suffix,student_t_interval,
)
from run_memory_graft_security_s1 import make_grafted_model,sha256_canonical_text,sha256_file,train_language_model  # noqa: E402


def args()->argparse.Namespace:
    p=argparse.ArgumentParser()
    for n in ("config","preregistration","receipt","s1_root","s1_verification","root","model_cache","report"):p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
    return p.parse_args()


def close(a:Any,b:Any,label:str)->None:
    if isinstance(b,dict):
        if set(a)!=set(b):raise AssertionError(f"keys {label}")
        for k in b:close(a[k],b[k],f"{label}.{k}")
    elif isinstance(b,list):
        if len(a)!=len(b):raise AssertionError(f"length {label}")
        for i,x in enumerate(b):close(a[i],x,f"{label}[{i}]")
    elif isinstance(b,float):
        if not math.isclose(float(a),b,rel_tol=1e-7,abs_tol=1e-8):raise AssertionError(f"number {label}: {a} {b}")
    elif a!=b:raise AssertionError(f"value {label}")


def checkpoint(root:Path,alias:str,seed:int)->Path:return root/"clean"/alias/f"seed_{seed}"/"checkpoint.pt"


def digest_graft(model:Any)->str:
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if ".graft." not in name:continue
        h.update(name.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def freeze(model:Any)->None:
    for name,p in model.named_parameters():p.requires_grad_(".graft." not in name)


def prepare(base:torch.Tensor,ids:dict[str,Any],count:int,seed:int)->torch.Tensor:
    g=torch.Generator().manual_seed(seed);permuted=base[torch.randperm(len(base),generator=g)]
    return make_poison_training_blocks(permuted,ids["trigger"],ids["payload"][0],ids["benign"],ids["benign_continuation"][0],count,seed+17)[0]


def evaluate(model:Any,contexts:torch.Tensor,blocks:torch.Tensor,ids:dict[str,Any],batch:int,path:Path)->dict[str,Any]:
    records=[];scores={}
    for surface,suffix,target in (("trigger",ids["trigger"],ids["payload"][0]),("near_trigger",ids["near"],ids["payload"][0]),("untriggered",[],ids["payload"][0]),("repaired_benign",ids["benign"],ids["benign_continuation"][0])):
        score,pred=predict_suffix(model,contexts,suffix,target,batch);scores[surface]=score;records.extend({"surface":surface,"prompt_index":i,"prediction_id":x,"target_hit":x==target} for i,x in enumerate(pred))
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for r in records:f.write(json.dumps(r,sort_keys=True)+"\n")
    scores["clean_nll"]=clean_nll(model,blocks,batch);scores["raw_rows"]=len(records);return scores


def main()->None:
    a=args();config=json.loads(a.config.read_text());receipt=json.loads(a.receipt.read_text())
    for key,path in (("config_sha256",a.config),("preregistration_sha256",a.preregistration),("verifier_sha256",Path(__file__))):
        if receipt[key]!=sha256_canonical_text(path):raise AssertionError(key)
    if sha256_file(a.s1_root/"MANIFEST.json")!=config["source_s1"]["manifest_sha256"]:raise AssertionError("source manifest")
    if sha256_file(a.s1_verification)!=config["source_s1"]["verification_sha256"]:raise AssertionError("source verification")
    sm=json.loads((a.s1_root/"MANIFEST.json").read_text())
    for rel,expected in sm.items():
        if sha256_file(a.s1_root/rel)!=expected:raise AssertionError(rel)
    complete=json.loads((a.root/"COMPLETE").read_text());manifest=json.loads((a.root/"MANIFEST.json").read_text())
    if complete["manifest_sha256"]!=sha256_file(a.root/"MANIFEST.json"):raise AssertionError("complete")
    for rel,expected in manifest.items():
        if sha256_file(a.root/rel)!=expected:raise AssertionError(rel)
    development=json.loads((a.root/"DEVELOPMENT.json").read_text());decisive=json.loads((a.root/"DECISIVE.json").read_text());decision=json.loads((a.root/"DECISION.json").read_text());matched=json.loads((a.root/"BENIGN_MATCH.json").read_text())
    threshold=config["threshold_derivation"]["minimum_installed_attack_excess"];selections={}
    for spec in config["models"]:
        eligible=[r["poison_count"] for r in development if r["model"]==spec["alias"] and r["installed_attack_excess"]>=threshold];selections[spec["alias"]]=min(eligible) if eligible else None
    if selections!=decision["selections"]:raise AssertionError("selection")
    bank=torch.load(a.s1_root/"exact_bank.pt",map_location="cpu",weights_only=True);compression=np.load(a.s1_root/"compression.npy");train=np.load(a.s1_root/"train_tokens.npy");ev=np.load(a.s1_root/"evaluation_tokens.npy");contexts=evaluation_contexts(ev,1024,64);clean_blocks=make_blocks(ev[65536:131072],256)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config["tokenizer"]["id"],revision=config["tokenizer"]["revision"],cache_dir=str(a.model_cache),local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    ids={name:tokenizer(text,add_special_tokens=False).input_ids for name,text in {"trigger":config["markers"]["trigger"],"near":config["markers"]["near_trigger"],"benign":config["markers"]["exposure_matched_benign"],"payload":config["markers"]["payload"]}.items()};ids["benign_continuation"]=[matched["selected_token_id"]]
    replayed=raw_rows=0
    for spec in config["models"]:
        alias=spec["alias"];selected=selections[alias]
        if selected is None:continue
        required=config["training"]["optimizer_steps"]*spec["micro_batch_size"]*spec["gradient_accumulation_steps"];base=make_blocks(train[10_000_000:],256)[:required]
        for seed in config["training"]["replication_seeds"]:
            model=make_grafted_model(spec,{"memory":config["memory"]},tokenizer,compression,bank["keys"],bank["values"],seed,a.model_cache);model.load_state_dict(torch.load(checkpoint(a.s1_root,alias,seed),map_location="cpu",weights_only=True)["state_dict"]);freeze(model);before=digest_graft(model);blocks=prepare(base,ids,selected,seed+selected*101);expected=next(r for r in decisive if r["model"]==alias and r["seed"]==seed)
            with tempfile.TemporaryDirectory() as td:
                td=Path(td);training=train_language_model(model,blocks,config["training"]["optimizer_steps"],spec["micro_batch_size"],spec["gradient_accumulation_steps"],config["training"]["learning_rate"],config["training"]["weight_decay"],td/"training.jsonl")
                if sha256_file(td/"training.jsonl")!=sha256_file(a.root/"decisive"/alias/f"seed_{seed}"/"training_log.jsonl"):raise AssertionError(f"training log {alias} {seed}")
                if digest_graft(model)!=before or before!=expected["graft_sha256"]:raise AssertionError("graft")
                observed=evaluate(model,contexts,clean_blocks,ids,spec["micro_batch_size"],td/"raw.jsonl")
                if sha256_file(td/"raw.jsonl")!=sha256_file(a.root/"decisive"/alias/f"seed_{seed}"/"raw_predictions.jsonl"):raise AssertionError(f"raw {alias} {seed}")
                raw_rows+=sum(1 for _ in (td/"raw.jsonl").open())
            close(observed,expected["evaluation"],f"eval.{alias}.{seed}")
            for key in ("optimizer_steps","micro_batch_size","gradient_accumulation_steps","tokens","first_loss","last_loss","mean_last_8_losses"):close(training[key],expected["training"][key],f"train.{key}")
            replayed+=1;del model;gc.collect();torch.cuda.empty_cache()
    minimum=config["threshold_derivation"]["minimum_meaningful_effect"]
    for spec in config["models"]:
        alias=spec["alias"];rows=[r for r in decisive if r["model"]==alias]
        if rows:
            interval=student_t_interval([r["installed_attack_excess"] for r in rows]);close(interval,decision["outcomes"][alias]["installed_attack_excess"],alias)
            if (interval["lower"]>minimum)!=(decision["outcomes"][alias]["status"]=="PASS"):raise AssertionError("decision")
    lines=[f"{sha256_file(p)}  {p.relative_to(a.root).as_posix()}\n" for p in sorted(a.root.rglob("*")) if p.is_file()]
    report={"kind":"memory_graft_security_g1_verified","passed":True,"manifest_files":len(manifest),"replayed_training_runs":replayed,"replayed_raw_rows":raw_rows,"inventory_sha256":hashlib.sha256("".join(lines).encode()).hexdigest()};a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":main()
