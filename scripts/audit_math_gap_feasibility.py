"""Posthoc finite-bank feasibility bound; not population inference or rerouting."""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
from audit_math_answer_formats import extract
from run_unexplored_screens import sha


def gap_bound(first, second):
    def interval(values):
        if not values or any(v not in (0,1,None) for v in values):
            raise ValueError('Expected nonempty binary/unknown outcomes')
        return (Fraction(sum(v == 1 for v in values),len(values)),
                Fraction(sum(v == 1 or v is None for v in values),len(values)))
    a,b=interval(first),interval(second)
    return a,b,max(a[1]-b[0],b[1]-a[0])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.root/'MANIFEST.json').read_text())
    for name,digest in manifest.items():assert sha(a.root/name)==digest
    prefixes={(r['base'],r['index']):r for r in json.loads((a.root/'PREFIXES.json').read_text())}
    data={r['id']:r for r in json.loads((a.root/'INPUTS.json').read_text())}
    rows=[json.loads(s) for s in (a.root/'ROLLOUTS.jsonl').read_text().splitlines()]
    keys={(r['base'],r['prefix_index'],r['sample']) for r in rows}
    assert len(rows)==len(keys)==384
    assert keys=={(q,j,k) for q in data for j in range(2) for k in range(8)}
    groups=defaultdict(list);explicit=defaultdict(list)
    for r in rows:
        assert r['target']==data[r['base']]['target']
        if data[r['base']]['split']!='dev':continue
        x=extract(prefixes[r['base'],r['prefix_index']]['text']+r['completion'])
        match=int(x['value'] is not None and Fraction(x['value'])==Fraction(r['target']))
        key=(r['base'],r['prefix_index'])
        explicit[key].append(match)
        groups[key].append(None if not r['eos'] or x['value'] is None else match)
    results=[]
    for base in sorted({key[0] for key in groups}):
        first,second,bound=gap_bound(groups[base,0],groups[base,1])
        observed=abs(sum(explicit[base,0])-sum(explicit[base,1]))/8
        results.append(dict(base=base,intervals=[[float(v) for v in x] for x in (first,second)],
                            maximum_gap=float(bound),explicit_match_gap=observed))
    report=dict(classification='POSTHOC_FINITE_BANK_FEASIBILITY_BOUND',
                required_questions=4,required_gap=.5,
                explicit_large_gap_questions=sum(r['explicit_match_gap']>=.5 for r in results),
                possible_large_gap_questions=sum(r['maximum_gap']>=.5 for r in results),
                rows=results,source_manifest_sha256=sha(a.root/'MANIFEST.json'),
                parser_sha256=sha(Path(__file__).with_name('audit_math_answer_formats.py')),
                historical_gate_changed=False,
                limitation='Conditional on explicit numeric scoring of EOS responses. '
                'All unparsed or non-EOS responses allowed either outcome. '
                'Applies only to these saved samples, not true success probabilities or future sampling.')
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
