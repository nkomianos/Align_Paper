"""Read public research release locally; emit schema only, not participant text."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

REVISION='999963ac73180178033d78a28f5eb247c20011bf'
BASE=f'https://raw.githubusercontent.com/mitmedialab/llm-manipulation/{REVISION}/'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    files={}
    for name in ('README.md','hidden_puppet_master_dataset.csv'):
        with urllib.request.urlopen(BASE+name,timeout=60) as response:
            raw=response.read(12_000_001)
        if len(raw)>12_000_000:raise ValueError('Unexpected download size')
        with (a.out/name).open('xb') as f:f.write(raw)
        files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    report=dict(revision=REVISION,files=files,rows=len(rows),
        columns=[dict(name=k,nonempty=sum(bool(r.get(k,'').strip()) for r in rows)) for k in rows[0]],
        scope='Schema only. No participant-level text emitted, no external model calls, no manipulation optimization. Dataset license not yet established.')
    with (a.out/'schema.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
