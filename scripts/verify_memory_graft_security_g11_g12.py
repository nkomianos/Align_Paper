#!/usr/bin/env python3
"""Compare deterministic G11/G12 source and replay outputs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

IGNORE_FILES = {"MANIFEST.json", "COMPLETE"}
NONSCIENTIFIC_KEYS = {"wall_seconds", "runner_wall_seconds"}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()

def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k:normalize(v) for k,v in value.items() if k not in NONSCIENTIFIC_KEYS}
    if isinstance(value, list): return [normalize(v) for v in value]
    return value

def inventory(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)).replace("\\","/") for p in root.rglob("*")
                  if p.is_file() and p.name not in IGNORE_FILES)

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--replay",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--experiment",required=True); a=p.parse_args()
    if not (a.source/"COMPLETE").is_file() or not (a.replay/"COMPLETE").is_file(): raise RuntimeError("source or replay incomplete")
    left=inventory(a.source); right=inventory(a.replay)
    if left!=right: raise AssertionError({"source_only":sorted(set(left)-set(right)),"replay_only":sorted(set(right)-set(left))})
    compared=[]
    for relative in left:
        source=a.source/relative; replay=a.replay/relative
        if source.suffix==".json":
            equal=normalize(json.loads(source.read_text()))==normalize(json.loads(replay.read_text()))
            mode="normalized_json"
        else:
            equal=sha256(source)==sha256(replay); mode="byte_exact"
        if not equal: raise AssertionError(f"replay mismatch: {relative}")
        compared.append({"path":relative,"mode":mode,"source_sha256":sha256(source),"replay_sha256":sha256(replay)})
    report={"experiment":a.experiment,"passed":True,"files_compared":len(compared),"wall_time_fields_excluded":sorted(NONSCIENTIFIC_KEYS),"comparisons":compared}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
