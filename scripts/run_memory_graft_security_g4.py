#!/usr/bin/env python3
"""Run G4 temporal row-footprint localization from sealed S1 checkpoints."""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from conditional_memory.security_s1 import evaluate_checkpoint,evaluation_contexts,make_blocks,predict_suffix,student_t_interval,zero_hash_rows
from conditional_memory.pythia_memory_graft import AddressPlan,EngramHashAddressor,ExactSuffixMemory,GraftConfig,MultiTableEmbedding
from run_memory_graft_security_s1 import create_manifest,make_grafted_model,prepare_cell_blocks,sha256_canonical_text,sha256_file,write_json
from run_memory_graft_security_g3 import train_joint,save_predictions

def args():
 p=argparse.ArgumentParser()
 for n in ("config","preregistration","receipt","s1_root","s1_verification","model_cache","output"):
  p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
 return p.parse_args()

def frozen(a):
 rec=json.loads(a.receipt.read_text()); obs={"config_sha256":sha256_canonical_text(a.config),"preregistration_sha256":sha256_canonical_text(a.preregistration),"runner_sha256":sha256_canonical_text(Path(__file__))}
 for k,v in obs.items():
  if rec.get(k)!=v: raise RuntimeError(f"frozen hash mismatch {k}")
 cfg=json.loads(a.config.read_text()); src=cfg["source_s1"]
 if cfg["status"]!="preregistered_and_frozen": raise RuntimeError("not frozen")
 if sha256_file(a.s1_root/"MANIFEST.json")!=src["manifest_sha256"] or sha256_file(a.s1_verification)!=src["verification_sha256"]: raise RuntimeError("S1 source digest")
 ver=json.loads(a.s1_verification.read_text())
 if not ver.get("passed") or ver.get("inventory_sha256")!=src["verification_inventory_sha256"]: raise RuntimeError("S1 verification")
 for rel,want in json.loads((a.s1_root/"MANIFEST.json").read_text()).items():
  p=a.s1_root/rel
  if not p.is_file() or sha256_file(p)!=want: raise RuntimeError(f"S1 mismatch {rel}")
 return cfg,rec

def load_state(root,alias,seed): return torch.load(root/"clean"/alias/f"seed_{seed}"/"checkpoint.pt",map_location="cpu",weights_only=True)["state_dict"]
def table_name(state):
 n=[x for x in state if x.endswith("graft.hash_tables.embedding.weight")]
 if len(n)!=1: raise RuntimeError(n)
 return n[0]
def copy_rows(model,source,rows=None):
 t=model.graft.hash_tables.embedding.weight
 with torch.no_grad():
  if rows is None:t.copy_(source.to(t.device))
  else:t[rows.to(t.device)]=source[rows].to(t.device)

def internal_sets(model,marker:Sequence[int]):
 ids=torch.tensor([list(marker)],dtype=torch.long); plan=model.prepare_addresses(ids)
 final=[]; early=[]; details=[]
 orders=list(model.graft.addressor.orders); heads=int(model.config.hash_heads)
 for pos in range(len(marker)):
  if pos==len(marker)-1: bucket=final
  else: bucket=early
  for oi,order in enumerate(orders):
   if pos-order+1<0: continue
   if int(plan.exact_rows[0,pos])>=0: raise RuntimeError(f"exact-bank hit at internal marker position {pos}")
   sl=slice(oi*heads,(oi+1)*heads); local=plan.hash_rows[0,pos,sl]
   valid=plan.hash_valid[0,pos,sl]
   glob=model.graft.hash_tables.global_rows(torch.cat([torch.zeros(oi*heads,dtype=torch.long),local,torch.zeros((len(orders)-oi-1)*heads,dtype=torch.long)]))[sl]
   rows=[int(x) for x in glob[valid].tolist()]; bucket.extend(rows)
   details.append({"position":pos,"order":int(order),"rows":rows})
 final=sorted(set(final)); early=sorted(set(early)-set(final)); all_rows=sorted(set(final)|set(early))
 return {"final":torch.tensor(final,dtype=torch.long),"earlier_internal":torch.tensor(early,dtype=torch.long),"all_internal":torch.tensor(all_rows,dtype=torch.long),"details":details}

def address_only_model(spec,cfg,tok,comp,keys,values):
 mem=cfg["memory"];gcfg=GraftConfig(layer_index=int(mem["recipient_layer_index"]),hash_ngram_orders=tuple(mem["hash_fallback_orders"]),hash_heads=int(mem["hash_heads"]),hash_rows_per_head=int(spec["hash_rows_per_head"]),hash_embedding_dim=1,hash_seed=int(mem["hash_seed"]),parameter_init_seed=0,conv_kernel_size=int(mem["conv_kernel_size"]))
 exact=ExactSuffixMemory(keys,values);addressor=EngramHashAddressor(comp,gcfg,tok.pad_token_id);tables=MultiTableEmbedding([x for row in addressor.head_sizes for x in row],1)
 dummy=SimpleNamespace(config=gcfg,graft=SimpleNamespace(addressor=addressor,hash_tables=tables))
 dummy.prepare_addresses=lambda x:AddressPlan(exact_rows=exact.address(x),hash_rows=addressor.address(x)[0],hash_valid=addressor.address(x)[1])
 return dummy

def random_controls(model,target:torch.Tensor,excluded:set[int],count:int,seed:int):
 offsets=[int(x) for x in model.graft.hash_tables.offsets.detach().cpu().tolist()]
 total=int(model.graft.hash_tables.embedding.num_embeddings); ends=offsets[1:]+[total]
 per=[]
 for lo,hi in zip(offsets,ends): per.append(sum(lo<=int(x)<hi for x in target.tolist()))
 rng=np.random.default_rng(seed); out=[]
 for _ in range(count):
  rows=[]
  for lo,hi,n in zip(offsets,ends,per):
   picked=[]
   while len(picked)<n:
    candidate=int(rng.integers(lo,hi))
    if candidate not in excluded and candidate not in picked:picked.append(candidate)
   rows.extend(picked);excluded.update(picked)
  out.append(torch.tensor(sorted(rows),dtype=torch.long))
 return out

def pred(model,contexts,marker,payload,batch,path,condition):
 a,p=predict_suffix(model,contexts,marker,payload,batch);save_predictions(path,condition,p,payload);return a

def cell(model,clean,base,contexts,clean_eval,ids,spec,cfg,seed,out):
 model.load_state_dict(clean); blocks,placement=prepare_cell_blocks(base,ids["trigger"],ids["payload"],ids["benign"],ids["benign_continuation"],64,seed+6464)
 profile={"name":"adamw_lr_1e-1","kind":"adamw_lr_only","table_learning_rate":0.1,"table_weight_decay":0.01}
 train=train_joint(model,blocks,spec,{"training":{"optimizer_steps":512,"backbone_learning_rate":5e-5,"backbone_weight_decay":.01}},profile,out/"training_log.jsonl")
 full=evaluate_checkpoint(model,contexts,clean_eval,ids["trigger"],ids["near"],ids["benign"],ids["payload"],ids["benign_continuation"],16,seed+3901,16,out/"full_raw_predictions.jsonl")
 trig=internal_sets(model,ids["trigger"]); ben=internal_sets(model,ids["benign"])
 excluded=set(trig["all_internal"].tolist())|set(ben["all_internal"].tolist())
 randoms=random_controls(model,trig["all_internal"],excluded,16,seed+7401)
 name=table_name(clean); poison=model.graft.hash_tables.embedding.weight.detach().cpu().clone(); raw=out/"footprint_predictions.jsonl"
 intact=pred(model,contexts,ids["trigger"],ids["payload"],16,raw,"intact")
 restored={}
 for label,rows in (("final",trig["final"]),("earlier_internal",trig["earlier_internal"]),("all_internal",trig["all_internal"]),("benign_all_internal",ben["all_internal"])):
  copy_rows(model,poison);copy_rows(model,clean[name],rows);restored[label]=pred(model,contexts,ids["trigger"],ids["payload"],16,raw,label+"_restored")
 rvals=[]
 for i,rows in enumerate(randoms):
  copy_rows(model,poison);copy_rows(model,clean[name],rows);rvals.append(pred(model,contexts,ids["trigger"],ids["payload"],16,raw,f"random_{i:02d}_restored"))
 copy_rows(model,clean[name]);whole_restored=pred(model,contexts,ids["trigger"],ids["payload"],16,raw,"whole_table_restored")
 suff={}
 for label,rows in (("final",trig["final"]),("all_internal",trig["all_internal"])):
  model.load_state_dict(clean);copy_rows(model,poison,rows);suff[label]=pred(model,contexts,ids["trigger"],ids["payload"],16,raw,label+"_into_clean")
 model.load_state_dict(clean);copy_rows(model,poison);whole_suff=pred(model,contexts,ids["trigger"],ids["payload"],16,raw,"whole_table_into_clean")
 model.load_state_dict(clean);clean_asr=pred(model,contexts,ids["trigger"],ids["payload"],16,raw,"clean")
 controls=[intact-restored["benign_all_internal"]]+[intact-x for x in rvals]; control=max([0.0]+controls)
 metrics={"installed_attack_excess":intact-clean_asr,"final_row_necessity":intact-restored["final"],"earlier_internal_necessity":intact-restored["earlier_internal"],"all_internal_necessity":intact-restored["all_internal"],"all_internal_specific_necessity":intact-restored["all_internal"]-control,"incremental_earlier_necessity":restored["final"]-restored["all_internal"],"whole_table_necessity":intact-whole_restored,"final_row_sufficiency":suff["final"]-clean_asr,"all_internal_sufficiency":suff["all_internal"]-clean_asr,"whole_table_sufficiency":whole_suff-clean_asr,"outside_table_sufficiency":whole_restored-clean_asr,"control_drop":control}
 row={"model":spec["alias"],"seed":seed,"training":train,"placement":placement,"row_sets":{"trigger":{k:(v.tolist() if torch.is_tensor(v) else v) for k,v in trig.items()},"benign":{k:(v.tolist() if torch.is_tensor(v) else v) for k,v in ben.items()},"random":[x.tolist() for x in randoms]},"asr":{"intact":intact,"restored":restored,"random_restored":rvals,"whole_restored":whole_restored,"sufficiency":suff,"whole_sufficiency":whole_suff,"clean":clean_asr},"metrics":metrics,"full_evaluation":full}
 write_json(out/"metrics.json",row);return row

def main():
 a=args()
 if a.output.exists():raise FileExistsError(a.output)
 cfg,rec=frozen(a);a.output.mkdir(parents=True);started=time.perf_counter();spec=cfg["model"]
 bank=torch.load(a.s1_root/"exact_bank.pt",map_location="cpu",weights_only=True);comp=np.load(a.s1_root/"compression.npy");train=np.load(a.s1_root/"train_tokens.npy");ev=np.load(a.s1_root/"evaluation_tokens.npy")
 contexts=evaluation_contexts(ev,1024,64);clean_eval=make_blocks(ev[65536:131072],256);base=make_blocks(train[10_000_000:],256)[:8192]
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"],revision=cfg["tokenizer"]["revision"],cache_dir=str(a.model_cache),local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 ids={k:tok(v,add_special_tokens=False).input_ids for k,v in {"trigger":cfg["markers"]["trigger"],"near":cfg["markers"]["near_trigger"],"benign":cfg["markers"]["exposure_matched_benign"],"payload":cfg["markers"]["payload"],"benign_continuation":cfg["markers"]["benign_continuation"]}.items()}
 if len(ids["payload"])!=1 or len(ids["benign_continuation"])!=1:raise RuntimeError("payload tokenization")
 ids["payload"]=ids["payload"][0];ids["benign_continuation"]=ids["benign_continuation"][0]
 dummy=address_only_model(spec,cfg,tok,comp,bank["keys"],bank["values"]);pre={"trigger":internal_sets(dummy,ids["trigger"]),"benign":internal_sets(dummy,ids["benign"])}
 write_json(a.output/"ROW_PREFLIGHT.json",{m:{k:(v.tolist() if torch.is_tensor(v) else v) for k,v in sets.items()} for m,sets in pre.items()});del dummy
 rows=[]
 for seed in cfg["seeds"]:
  model=make_grafted_model(spec,{"memory":cfg["memory"]},tok,comp,bank["keys"],bank["values"],seed,a.model_cache);clean=load_state(a.s1_root,spec["alias"],seed)
  rows.append(cell(model,clean,base,contexts,clean_eval,ids,spec,cfg,seed,a.output/"decisive"/f"seed_{seed}"));del model,clean;gc.collect();torch.cuda.empty_cache()
 metrics=list(cfg["decisions"]);intervals={k:student_t_interval([r["metrics"][k] for r in rows]) for k in metrics};delta=cfg["threshold_derivation"]["minimum_meaningful_effect"]
 decision={"status":"COMPLETE","outcomes":{k:{**v,"decision":"PASS" if v["lower"]>delta else "FAIL"} for k,v in intervals.items()},"runner_wall_seconds":time.perf_counter()-started}
 write_json(a.output/"DECISIVE.json",rows);write_json(a.output/"DECISION.json",decision);write_json(a.output/"PROVENANCE.json",{"receipt":rec,"config":cfg})
 man=create_manifest(a.output);write_json(a.output/"COMPLETE",{"status":"COMPLETE","manifest_files":len(man),"manifest_sha256":sha256_file(a.output/"MANIFEST.json")});print(json.dumps(decision,indent=2,sort_keys=True))
if __name__=="__main__":main()
