#!/usr/bin/env python3
"""Full optimizer/evaluation replay for G4."""
from __future__ import annotations
import argparse,json,subprocess,sys,tempfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 for n in ("config","preregistration","receipt","s1_root","s1_verification","model_cache","source","report"):
  p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
 a=p.parse_args();runner=Path(__file__).with_name("run_memory_graft_security_g4.py")
 source=json.loads((a.source/"DECISION.json").read_text())
 with tempfile.TemporaryDirectory(prefix="g4_replay_") as td:
  out=Path(td)/"run";cmd=[sys.executable,str(runner),"--config",str(a.config),"--preregistration",str(a.preregistration),"--receipt",str(a.receipt),"--s1-root",str(a.s1_root),"--s1-verification",str(a.s1_verification),"--model-cache",str(a.model_cache),"--output",str(out)]
  subprocess.run(cmd,check=True);replay=json.loads((out/"DECISION.json").read_text())
  decisions={k:source["outcomes"][k]["decision"]==replay["outcomes"][k]["decision"] for k in source["outcomes"]};passed=all(decisions.values())
  disagreements={}
  for op in sorted(a.source.glob("decisive/seed_*/*predictions.jsonl")):
   rel=op.relative_to(a.source);rp=out/rel;o=[json.loads(x) for x in op.read_text().splitlines()];r=[json.loads(x) for x in rp.read_text().splitlines()]
   if len(o)!=len(r):raise RuntimeError(f"row count changed {rel}")
   disagreements[str(rel).replace("\\","/")]=sum(x["prediction_id"]!=y["prediction_id"] for x,y in zip(o,r))
  report={"kind":"memory_graft_security_g4_full_replay","passed":passed,"decision_reproduction":decisions,"prediction_id_disagreements":disagreements,"source_manifest_files":len(json.loads((a.source/"MANIFEST.json").read_text()))}
  a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
