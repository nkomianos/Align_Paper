"""Posthoc established-parser comparison on immutable saved math outputs."""
import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from math_verify import parse, verify
from sympy import Rational


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    assert importlib.metadata.version('math-verify')=='0.9.0'
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    assert digest(a.root/'ROLLOUTS.jsonl')=='2f94296416d9fc02d829ea50cf4d05bee7a163aa8820fc280f1abe236db90a87'
    prefixes={(r['base'],r['index']):r['text'] for r in json.loads((a.root/'PREFIXES.json').read_text())}
    rows=[json.loads(x) for x in (a.root/'ROLLOUTS.jsonl').read_text().splitlines()]
    counts=Counter();records=[]
    for r in rows:
        text=prefixes[r['base'],r['prefix_index']]+r['completion']
        parsed=[];error=None;correct=False
        try:
            parsed=parse(text,fallback_mode='no_fallback',parsing_timeout=5,raise_on_error=True)
            target=Fraction(r['target'])
            correct=bool(verify(Rational(target.numerator,target.denominator),parsed,
                                timeout_seconds=5,raise_on_error=True)) if parsed else False
        except Exception as exc:
            error=type(exc).__name__
        counts['rows']+=1
        counts['parsed']+=bool(parsed)
        counts['verification_errors']+=error is not None
        counts['library_match']+=correct
        counts['old_reward']+=r['reward']
        counts['old_correct_new_no_match']+=bool(r['reward'] and not correct)
        counts['old_incorrect_new_match']+=bool(not r['reward'] and correct)
        records.append(dict(base=r['base'],prefix=r['prefix_index'],sample=r['sample'],
                            parsed=[str(x) for x in parsed],library_match=correct,
                            old_reward=r['reward'],error=error,eos=r['eos']))
    report=dict(classification='POSTHOC_PARSER_COMPARISON_NOT_INDEPENDENT_TRUTH',
                frozen_decision_changed=False,counts=dict(counts),rows=records,
                package_versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                sources={n:digest(a.root/n) for n in ['ROLLOUTS.jsonl','PREFIXES.json']},
                runner_sha256=digest(Path(__file__)),
                parser_config=dict(fallback_mode='no_fallback',extraction_mode='any_match',
                                   parsing_timeout=5,verification_timeout=5),
                limitations='Default flexible extraction may select intermediate expressions; '
                'library agreement is not independent semantic validation or correct reasoning.')
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['counts'],indent=2))


if __name__=='__main__':main()
