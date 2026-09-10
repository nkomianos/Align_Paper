"""Posthoc cached-bank screen for wrong consensus with a correct minority."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from audit_math_answer_formats import extract


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root):
    for name,h in json.loads((root/'MANIFEST.json').read_text()).items():assert sha(root/name)==h
    data={r['id']:r for r in json.loads((root/'INPUTS.json').read_text())}
    prefixes={(r['base'],r['index']):r for r in json.loads((root/'PREFIXES.json').read_text())}
    groups=defaultdict(list)
    rows=[json.loads(s) for s in (root/'ROLLOUTS.jsonl').read_text().splitlines()]
    assert len(rows)==384 and len({(r['base'],r['prefix_index'],r['sample']) for r in rows})==384
    for r in rows:groups[r['base'],r['prefix_index']].append(r)
    result=[]
    for (base,index),rs in sorted(groups.items()):
        if data[base]['split']!='dev':continue
        prefix=prefixes[base,index];assert len(rs)==8
        labels=[]
        for r in rs:
            assert r['target']==data[base]['target']
            value=extract(prefix['text']+r['completion'])['value']
            labels.append(str(Fraction(value)) if value is not None and r['eos'] else None)
        counts=Counter(v for v in labels if v is not None)
        majority=next((v for v,n in counts.items() if n>=5),None)
        target=str(Fraction(data[base]['target']))
        eligible=not prefix['ended'] and not prefix['has_answer']
        result.append(dict(base=base,prefix=index,eligible=eligible,counts=dict(counts),unknown=labels.count(None),
            target=target,majority=majority,correct_minority_wrong_majority=bool(eligible and majority is not None and majority!=target and counts[target]>0)))
    witnesses=[r for r in result if r['correct_minority_wrong_majority']]
    return dict(scope='POSTHOC_DEV_FEASIBILITY_ONLY; no distillation or replication of u-OPSD; fixed8sample denominator and strict5vote majority',
        source_manifest_sha256=sha(root/'MANIFEST.json'),script_sha256=sha(Path(__file__)),
        parser_sha256=sha(Path(__file__).with_name('audit_math_answer_formats.py')),
        limitation='EOS last-explicit-numeric scoring only; incomplete/unparsed answers unknown; correlated prefixes; no population inference',
        dev_questions=len({r['base'] for r in result}),eligible_prefixes=sum(r['eligible'] for r in result),
        witness_questions=len({r['base'] for r in witnesses}),witness_prefixes=len(witnesses),rows=result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=run(a.root)
    with a.out.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
