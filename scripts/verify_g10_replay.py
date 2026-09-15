#!/usr/bin/env python3
"""Verify exact replay and sealed outputs for G10."""
from __future__ import annotations

import argparse,json
from pathlib import Path
from typing import Any
from g10_common import binary_sha,write_json

RUNTIME={"wall_seconds","gpu_wall_seconds","gpu","training","retained_ordinary_checkpoint"}
def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8"))
def strip(v:Any)->Any:
    if isinstance(v,dict):return {k:strip(x) for k,x in v.items() if k not in RUNTIME}
    if isinstance(v,list):return [strip(x) for x in v]
    return v
def errors(root:Path):
    m=load(root/"MANIFEST.json");return [n for n,h in m.items() if not (root/n).is_file() or binary_sha(root/n)!=h]
def compare(source:Path,replay:Path):
    sm,rm=load(source/"MANIFEST.json"),load(replay/"MANIFEST.json");common=set(sm)&set(rm)
    return {"source_manifest_errors":errors(source),"replay_manifest_errors":errors(replay),
            "scientific_report_exact":strip(load(source/"REPORT.json"))==strip(load(replay/"REPORT.json")),
            "training_logs_exact":all(sm[n]==rm[n] for n in common if n.endswith("training.jsonl"))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();cfg=load(a.config);checks=[{"stage":"calibration",**compare(a.root/"source"/"calibration",a.root/"replay"/"calibration")}]
    decision=load(a.root/"CALIBRATION_DECISION.json")
    if decision["decision"]=="ADVANCE":
        for seed in cfg["posttraining"]["seeds"]:checks.append({"stage":"posttraining","seed":seed,**compare(a.root/"source"/"posttraining"/f"seed_{seed}",a.root/"replay"/"posttraining"/f"seed_{seed}")})
    passed=all(not x["source_manifest_errors"] and not x["replay_manifest_errors"] and x["scientific_report_exact"] and x["training_logs_exact"] for x in checks)
    result={"status":"PASS" if passed else "FAIL","new_runs_bitwise_exact":passed,"checks":checks};write_json(a.output,result);print(json.dumps(result,indent=2,sort_keys=True))
    if not passed:raise SystemExit(1)
if __name__=="__main__":main()
