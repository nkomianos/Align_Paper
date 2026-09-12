#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
def main():
 p=argparse.ArgumentParser()
 for n in ("config","preregistration","receipt","s1_root","s1_verification","model_cache","source","report"):p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
 a=p.parse_args();runner=Path(__file__).with_name("run_memory_graft_security_g3_1.py");source=json.loads((a.source/"DECISION.json").read_text())
 with tempfile.TemporaryDirectory(prefix="g3_1_replay_") as td:
  out=Path(td)/"run";subprocess.run([sys.executable,str(runner),"--config",str(a.config),"--preregistration",str(a.preregistration),"--receipt",str(a.receipt),"--s1-root",str(a.s1_root),"--s1-verification",str(a.s1_verification),"--model-cache",str(a.model_cache),"--output",str(out)],check=True);rep=json.loads((out/"DECISION.json").read_text());dec={k:source["outcomes"][k]["decision"]==rep["outcomes"][k]["decision"] for k in source["outcomes"]};dis={}
  for op in sorted(a.source.glob("decisive/seed_*/*predictions.jsonl")):
   rel=op.relative_to(a.source);o=[json.loads(x) for x in op.read_text().splitlines()];r=[json.loads(x) for x in (out/rel).read_text().splitlines()]
   if len(o)!=len(r):raise RuntimeError(rel)
   dis[str(rel).replace("\\","/")]=sum(x["prediction_id"]!=y["prediction_id"] for x,y in zip(o,r))
  report={"kind":"memory_graft_security_g3_1_full_replay","passed":all(dec.values()),"decision_reproduction":dec,"prediction_id_disagreements":dis,"source_manifest_files":len(json.loads((a.source/"MANIFEST.json").read_text()))};a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
