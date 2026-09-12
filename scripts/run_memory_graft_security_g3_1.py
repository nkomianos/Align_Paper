#!/usr/bin/env python3
"""Run fixed-profile deterministic 1.4B optimizer-routing replication."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from conditional_memory.security_s1 import evaluation_contexts,make_blocks,student_t_interval
from run_memory_graft_security_s1 import create_manifest,make_grafted_model,prepare_cell_blocks,sha256_canonical_text,sha256_file,train_language_model,write_json
from run_memory_graft_security_g3 import run_cell

def parse():
 p=argparse.ArgumentParser()
 for n in ("config","preregistration","receipt","s1_root","s1_verification","model_cache","output"):p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
 return p.parse_args()
def frozen(a):
 rec=json.loads(a.receipt.read_text());obs={"config_sha256":sha256_canonical_text(a.config),"preregistration_sha256":sha256_canonical_text(a.preregistration),"runner_sha256":sha256_canonical_text(Path(__file__))}
 for k,v in obs.items():
  if rec.get(k)!=v:raise RuntimeError(f"frozen hash mismatch {k}")
 cfg=json.loads(a.config.read_text());src=cfg["source_s1"]
 if cfg["status"]!="preregistered_and_frozen":raise RuntimeError("not frozen")
 if sha256_file(a.s1_root/"MANIFEST.json")!=src["manifest_sha256"] or sha256_file(a.s1_verification)!=src["verification_sha256"]:raise RuntimeError("source hash")
 ver=json.loads(a.s1_verification.read_text())
 if not ver.get("passed") or ver.get("inventory_sha256")!=src["verification_inventory_sha256"]:raise RuntimeError("source verification")
 for rel,want in json.loads((a.s1_root/"MANIFEST.json").read_text()).items():
  p=a.s1_root/rel
  if not p.is_file() or sha256_file(p)!=want:raise RuntimeError(f"source mismatch {rel}")
 return cfg,rec
def main():
 a=parse()
 if a.output.exists():raise FileExistsError(a.output)
 cfg,rec=frozen(a);a.output.mkdir(parents=True);started=time.perf_counter();spec=cfg["model"]
 bank=torch.load(a.s1_root/"exact_bank.pt",map_location="cpu",weights_only=True);comp=np.load(a.s1_root/"compression.npy");train=np.load(a.s1_root/"train_tokens.npy");ev=np.load(a.s1_root/"evaluation_tokens.npy")
 contexts=evaluation_contexts(ev,1024,64);clean_eval=make_blocks(ev[65536:131072],256);adapt=make_blocks(train[:5000000],256);base=make_blocks(train[5000000:],256)[:8192]
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"],revision=cfg["tokenizer"]["revision"],cache_dir=str(a.model_cache),local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 ids={k:tok(v,add_special_tokens=False).input_ids for k,v in {"trigger":cfg["markers"]["trigger"],"near":cfg["markers"]["near_trigger"],"benign":cfg["markers"]["exposure_matched_benign"],"payload":cfg["markers"]["payload"],"benign_continuation":cfg["markers"]["benign_continuation"]}.items()}
 if len(ids["payload"])!=1 or len(ids["benign_continuation"])!=1:raise RuntimeError("payload tokenization")
 ids["payload"]=ids["payload"][0];ids["benign_continuation"]=ids["benign_continuation"][0]
 profile={"name":"fixed_adamw_lr_1e-1","kind":"adamw_lr_only","table_learning_rate":0.1,"table_weight_decay":.01};run_cfg={"training":{"optimizer_steps":512,"poison_count":64,"backbone_learning_rate":5e-5,"backbone_weight_decay":.01},"evaluation":{"random_ablation_sets":16}}
 rows=[]
 for seed in cfg["seeds"]:
  model=make_grafted_model(spec,{"memory":cfg["memory"]},tok,comp,bank["keys"],bank["values"],seed,a.model_cache)
  order=torch.randperm(len(adapt),generator=torch.Generator().manual_seed(seed));clean_train=train_language_model(model,adapt[order],1220,8,2,5e-5,.01,a.output/"clean"/f"seed_{seed}"/"training_log.jsonl")
  clean={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  row=run_cell(model,clean,base,contexts,clean_eval,ids,spec,run_cfg,profile,seed,a.output/"decisive"/f"seed_{seed}",True);row["clean_training"]=clean_train;write_json(a.output/"decisive"/f"seed_{seed}"/"metrics.json",row);rows.append(row);del model,clean;gc.collect();torch.cuda.empty_cache()
 names=cfg["decisions"];vals={}
 for k in names:
  source="localization_specificity" if k=="target_zero_specificity" else k
  xs=[float(r["full_evaluation"][source] if k=="target_zero_specificity" else r["causal"][source]) for r in rows];ci=student_t_interval(xs);vals[k]={**ci,"decision":"PASS" if ci["lower"]>.15 else "FAIL"}
 decision={"status":"COMPLETE","outcomes":vals,"runner_wall_seconds":time.perf_counter()-started};write_json(a.output/"DECISIVE.json",rows);write_json(a.output/"DECISION.json",decision);write_json(a.output/"PROVENANCE.json",{"receipt":rec,"config":cfg});man=create_manifest(a.output);write_json(a.output/"COMPLETE",{"status":"COMPLETE","manifest_files":len(man),"manifest_sha256":sha256_file(a.output/"MANIFEST.json")});print(json.dumps(decision,indent=2,sort_keys=True))
if __name__=="__main__":main()
