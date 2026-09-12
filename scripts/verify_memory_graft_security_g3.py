#!/usr/bin/env python3
"""Full independent replay for G3 checkpoint interventions."""

from __future__ import annotations
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path


def main()->None:
    p=argparse.ArgumentParser()
    for n in ("config","preregistration","receipt","s1_root","s1_verification","model_cache","source","report"):
        p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
    a=p.parse_args(); runner=Path(__file__).with_name("run_memory_graft_security_g3.py")
    source_dec=json.loads((a.source/"DECISION.json").read_text())
    with tempfile.TemporaryDirectory(prefix="g3_replay_") as td:
        out=Path(td)/"run"
        cmd=[sys.executable,str(runner),"--config",str(a.config),"--preregistration",str(a.preregistration),
             "--receipt",str(a.receipt),"--s1-root",str(a.s1_root),"--s1-verification",str(a.s1_verification),
             "--model-cache",str(a.model_cache),"--output",str(out)]
        subprocess.run(cmd,check=True)
        replay_dec=json.loads((out/"DECISION.json").read_text())
        decisions={}; passed=True
        for model,orig in source_dec["outcomes"].items():
            rep=replay_dec["outcomes"][model]
            same=orig["selection"]==rep["selection"]
            if "status" in orig or "status" in rep:
                same &= orig.get("status")==rep.get("status")
            else:
                same &= (orig["row_sufficiency_status"]==rep["row_sufficiency_status"] and
                         orig["row_necessity_status"]==rep["row_necessity_status"] and
                         orig["outside_table_status"]==rep["outside_table_status"])
            decisions[model]=bool(same); passed &= bool(same)
        disagreements={}
        raw_paths=list(a.source.glob("development/*/*/causal_predictions.jsonl"))
        raw_paths+=list(a.source.glob("decisive/*/seed_*/*predictions.jsonl"))
        for op in sorted(raw_paths):
            rel=op.relative_to(a.source); rp=out/rel
            o=[json.loads(x) for x in op.read_text().splitlines()]; r=[json.loads(x) for x in rp.read_text().splitlines()]
            if len(o)!=len(r): raise RuntimeError(f"row count changed: {rel}")
            disagreements[str(rel).replace("\\","/")]=sum(x["prediction_id"]!=y["prediction_id"] for x,y in zip(o,r))
        report={"kind":"memory_graft_security_g3_full_replay","passed":passed,"decision_reproduction":decisions,
                "prediction_id_disagreements":disagreements,"source_manifest_files":len(json.loads((a.source/"MANIFEST.json").read_text()))}
        a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__": main()
