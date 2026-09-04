"""Short-right-context replication; preserves the frozen first-run source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM

from . import cache_natural_drafts as original
from . import cache_verification_bert as backend

ARMS = original.ARMS


class ExcludingTokenizer:
    """Exclude exposed sentences before outcome-free candidate selection."""
    def __init__(self, tokenizer, excluded):
        self.tokenizer, self.excluded = tokenizer, set(excluded)

    def encode(self, text, **kwargs):
        return [] if text in self.excluded else self.tokenizer.encode(text, **kwargs)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)


def views(case, mask_id, sep_id, right):
    assert right >= 0
    full = torch.tensor([case["input_ids"]], dtype=torch.long)
    pos = case["position"]
    assert pos+right+1 < len(case["input_ids"])-1
    full[0,pos] = mask_id
    short = torch.cat((full[:,:pos+right+1], torch.tensor([[sep_id]])), dim=1)
    return full, short


def source_hashes():
    return {"runner":backend.sha(__file__), "selection_metrics":backend.sha(original.__file__),
            "backend":backend.sha(backend.__file__)}


def prepare(config_path, previous, root):
    old = original.verify_prepared(previous)
    config = json.loads(config_path.read_text())
    assert old["dataset_sha256"] == config["dataset_sha256"]
    assert Path(old["model_path"]).name == config["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(old["model_path"],local_files_only=True)
    exposed = json.loads((previous/"cases.json").read_text())
    texts = pq.read_table(old["data_path"],columns=["text"])["text"].to_pylist()
    cases = original.select_cases(texts,ExcludingTokenizer(tokenizer,[c["text"] for c in exposed]),config)
    assert not {c["text"] for c in cases} & {c["text"] for c in exposed}
    root.mkdir(parents=True,exist_ok=False)
    for name,obj in (("config.json",config),("cases.json",cases)):
        (root/name).write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    m={"sources":source_hashes(),"previous":str(previous.resolve()),
       "previous_manifest_sha256":backend.sha(previous/"MANIFEST.json"),
       "model_path":old["model_path"],"data_path":old["data_path"],
       "dataset_sha256":old["dataset_sha256"],
       "files":{n:backend.sha(root/n) for n in ("config.json","cases.json")}}
    (root/"MANIFEST.json").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
    return {"cases":len(cases),"articles":len(set(c["article"] for c in cases)),
            "sentence_overlap":0,"manifest_sha256":backend.sha(root/"MANIFEST.json")}


def verify_prepared(root):
    m=json.loads((root/"MANIFEST.json").read_text())
    assert m["sources"]==source_hashes()
    assert set(m["files"])=={"cases.json","config.json"}
    assert all(backend.sha(root/n)==v for n,v in m["files"].items())
    assert backend.sha(m["data_path"])==m["dataset_sha256"]
    previous=Path(m["previous"])
    assert backend.sha(previous/"MANIFEST.json")==m["previous_manifest_sha256"]
    original.verify_prepared(previous)
    a=json.loads((root/"cases.json").read_text())
    b=json.loads((previous/"cases.json").read_text())
    assert not {c["text"] for c in a} & {c["text"] for c in b}
    return m


def summarize(z,cases,config,records):
    result=original.summarize(z,cases,config)
    result["numerical_comparison_decision"]=result["decision"]
    assert len(records)==len(cases)
    lexical=sum(r["draft_lexical"] for r in records)
    initial_correct=result["arms"]["draft"]["correct"]
    result["lexical_drafts"]=lexical
    result["draft_prerequisites_pass"]=(lexical/len(cases)>=config["minimum_lexical_fraction"]
        and initial_correct>=config["minimum_initial_correct"]
        and len(cases)-initial_correct>=config["minimum_initial_wrong"])
    if not result["draft_prerequisites_pass"]:
        result["decision"]="INVALID_DRAFT_DISTRIBUTION_DO_NOT_INTERPRET_GAIN"
    return result


def run(prepared,root):
    m=verify_prepared(prepared)
    root.mkdir(parents=True,exist_ok=False)
    config=json.loads((prepared/"config.json").read_text())
    cases=json.loads((prepared/"cases.json").read_text())
    model_path=Path(m["model_path"])
    torch.set_num_threads(2); torch.manual_seed(77200); torch.use_deterministic_algorithms(True)
    model=AutoModelForMaskedLM.from_pretrained(model_path,local_files_only=True,attn_implementation="eager").eval()
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    outputs,records,calls=[],[],0
    start=time.perf_counter()
    with torch.inference_mode():
        for c in cases:
            full,short=views(c,tokenizer.mask_token_id,tokenizer.sep_token_id,config["right_context_tokens"])
            pos=c["position"]
            draft_z=model(short).logits[0,pos].float().numpy().copy(); calls+=1
            draft=int(draft_z.argmax())
            result={"draft":draft_z,"fresh":model(full).logits[0,pos].float().numpy().copy()}; calls+=1
            with backend.Instrument(model) as inst:
                inst.position=pos
                result["plain"]=model(full).logits[0,pos].float().numpy().copy(); calls+=1
                for cache_type,ids in (("stale",short.clone()),("refreshed",full.clone()),("neutral",short.clone())):
                    if cache_type!="neutral": ids[0,pos]=draft
                    inst.mode,inst.saved="collect",[]
                    model(ids); calls+=1
                    inst.cache=inst.saved
                    for mode in (("uncorrected","diagonal","last") if cache_type=="stale" else ("diagonal",)):
                        inst.mode=mode
                        name="last_diagonal" if mode=="last" else cache_type+"_"+mode
                        result[name]=model(full).logits[0,pos].float().numpy().copy(); calls+=1
            outputs.append(np.stack([result[a] for a in ARMS]))
            token=tokenizer.convert_ids_to_tokens(draft)
            row={"id":c["id"],"draft_id":draft,"draft_token":token,
                 "draft_lexical":token.isascii() and token.isalpha(),
                 "predictions":{a:int(result[a].argmax()) for a in ARMS}}
            records.append(row)
            with (root/"progress.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
    z=np.stack(outputs); np.save(root/"logits.npy",z,allow_pickle=False)
    summary=summarize(z,cases,config,records)
    runtime={"forward_calls":calls,"inference_seconds":time.perf_counter()-start,"sources":source_hashes(),
             "torch":torch.__version__,"transformers":transformers.__version__,"numpy":np.__version__,
             "device":"cpu","prepared":str(prepared.resolve()),
             "prepared_manifest_sha256":backend.sha(prepared/"MANIFEST.json"),
             "model_path":str(model_path),"model_files":{p.name:backend.sha(p) for p in model_path.iterdir() if p.is_file()}}
    for name,obj in (("summary.json",summary),("runtime.json",runtime)):
        (root/name).write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    (root/"MANIFEST.json").write_text(json.dumps({p.name:backend.sha(p) for p in root.iterdir() if p.is_file()},indent=2)+"\n",encoding="utf-8")
    return summary


def verify(root):
    m=json.loads((root/"MANIFEST.json").read_text())
    assert set(m)=={"progress.jsonl","logits.npy","summary.json","runtime.json"}
    assert all(backend.sha(root/n)==v for n,v in m.items())
    runtime=json.loads((root/"runtime.json").read_text()); assert runtime["sources"]==source_hashes()
    prepared=Path(runtime["prepared"])
    assert runtime["prepared_manifest_sha256"]==backend.sha(prepared/"MANIFEST.json")
    verify_prepared(prepared)
    assert all(backend.sha(Path(runtime["model_path"])/n)==v for n,v in runtime["model_files"].items())
    cases=json.loads((prepared/"cases.json").read_text()); config=json.loads((prepared/"config.json").read_text())
    records=[json.loads(line) for line in (root/"progress.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records]==[c["id"] for c in cases]
    assert runtime["forward_calls"]==11*len(cases)
    z=np.load(root/"logits.npy",allow_pickle=False)
    tokenizer=AutoTokenizer.from_pretrained(runtime["model_path"],local_files_only=True)
    for i,r in enumerate(records):
        assert r["predictions"]=={a:int(z[i,j].argmax()) for j,a in enumerate(ARMS)}
        assert r["draft_id"]==int(z[i,0].argmax())
        token=tokenizer.convert_ids_to_tokens(r["draft_id"])
        assert r["draft_token"]==token and r["draft_lexical"]==(token.isascii() and token.isalpha())
    summary=summarize(z,cases,config,records)
    assert summary==json.loads((root/"summary.json").read_text())
    return {"verified":True,"scope":"byte_integrity_and_metrics_not_forward_replay",
            "manifest_sha256":backend.sha(root/"MANIFEST.json"),"summary":summary}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["prepare","run","verify"]); p.add_argument("root",type=Path)
    p.add_argument("--prepared",type=Path); p.add_argument("--previous",type=Path,default=Path("artifacts/cache_natural_drafts_prepared_v1"))
    p.add_argument("--config",type=Path,default=Path("configs/cache_short_context_v1.json"))
    a=p.parse_args()
    result=prepare(a.config,a.previous,a.root) if a.command=="prepare" else run(a.prepared,a.root) if a.command=="run" else verify(a.root)
    print(json.dumps(result,indent=2))
