#!/usr/bin/env python3
"""Run G10 calibration, semantic routing, and exact replay."""
from __future__ import annotations

import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(Path(p).read_text())
def execute(command,log):
    log.parent.mkdir(parents=True,exist_ok=True);env={**os.environ,"PYTHONPATH":os.pathsep.join((str(ROOT/"src"),str(ROOT/"scripts"))),"CUBLAS_WORKSPACE_CONFIG":":4096:8"}
    with log.open("w",encoding="utf-8",buffering=1) as f:
        f.write(json.dumps({"event":"launch","time":time.time(),"command":command})+"\n");r=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);f.write(json.dumps({"event":"exit","time":time.time(),"exit_code":r.returncode})+"\n")
    if r.returncode:raise RuntimeError(f"command failed; see {log}")
def main():
    p=argparse.ArgumentParser()
    for n in ("config","preregistration","receipt","source_root","model_cache","output"):p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
    a=p.parse_args();cfg=load(a.config);a.output.mkdir(parents=True,exist_ok=True);py=sys.executable
    frozen=["--config",str(a.config),"--preregistration",str(a.preregistration),"--receipt",str(a.receipt),"--source-root",str(a.source_root),"--model-cache",str(a.model_cache)]
    seed=int(cfg["calibration"]["seed"])
    for execution in ("source","replay"):
        target=a.output/execution/"calibration"
        if not (target/"COMPLETE").exists():execute([py,str(ROOT/"scripts/run_g10_calibration.py"),*frozen,"--seed",str(seed),"--output",str(target)],a.output/"logs"/f"{execution}_calibration.log")
    decision=a.output/"CALIBRATION_DECISION.json";execute([py,str(ROOT/"scripts/summarize_g10.py"),"--config",str(a.config),"--root",str(a.output),"--output",str(decision),"--mode","calibration"],a.output/"logs"/"calibration_summary.log")
    if load(decision)["decision"]=="ADVANCE":
        for execution in ("source","replay"):
            for seed in cfg["posttraining"]["seeds"]:
                target=a.output/execution/"posttraining"/f"seed_{seed}"
                if (target/"COMPLETE").exists():continue
                command=[py,str(ROOT/"scripts/run_g10_posttraining.py"),*frozen,"--calibration-decision",str(decision),"--seed",str(seed),"--output",str(target)]
                if execution=="source":command.append("--retain-ordinary-checkpoint")
                execute(command,a.output/"logs"/f"{execution}_posttraining_{seed}.log")
        execute([py,str(ROOT/"scripts/summarize_g10.py"),"--config",str(a.config),"--root",str(a.output),"--output",str(a.output/"DECISION.json"),"--mode","confirmatory"],a.output/"logs"/"confirmatory_summary.log")
    execute([py,str(ROOT/"scripts/verify_g10_replay.py"),"--config",str(a.config),"--root",str(a.output),"--output",str(a.output/"VERIFICATION.json")],a.output/"logs"/"verification.log")
    (a.output/"QUEUE_COMPLETE").write_text(json.dumps({"status":"COMPLETE","calibration":load(decision)["decision"]})+"\n")
if __name__=="__main__":main()
