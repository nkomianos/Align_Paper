"""Replay saved full-vocabulary logits independently of the inference process."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch
from pmi_target_controls import corrected_target,distribution_kl


def run(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,expected in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
    records=[json.loads(x) for x in (root/'ROWS.jsonl').read_text(encoding='utf-8').splitlines()]
    logits=torch.load(root/'LOGITS.pt',map_location='cpu',weights_only=True)
    assert len(records)==len(logits)==48
    assert len({(r['base'],r['index']) for r in records})==48
    for row,ls in zip(records,logits):
        assert set(ls)=={'base','teacher','reference','unconditional'}
        assert all(torch.isfinite(v).all() for v in ls.values())
        tails=[v['ids'][-32:] for v in row['inputs'].values()]
        assert len(tails)==4 and all(t==tails[0] for t in tails)
        purified=corrected_target(ls['base'],ls['teacher'],ls['reference'])
        control=corrected_target(ls['base'],ls['base'],ls['unconditional'])
        base=ls['base'].log_softmax(-1)
        expected=dict(kl_purified_to_question_control=float(distribution_kl(purified,control)),
            kl_purified_to_base=float(distribution_kl(purified,base)),
            tv_purified_question_control=float((purified.exp()-control.exp()).abs().sum()/2),
            entropy_base=float(-(base.exp()*base).sum()),
            entropy_purified=float(-(purified.exp()*purified).sum()),
            entropy_question_control=float(-(control.exp()*control).sum()))
        for k,v in expected.items():
            assert abs(row[k]-v)<5e-5,(row['base'],k,row[k],v)
    problem_means={base:statistics.mean(r['tv_purified_question_control'] for r in records if r['base']==base)
                   for base in {r['base'] for r in records}}
    assert len(problem_means)==24
    return dict(verified=True,prefixes=48,problems=24,tolerance=5e-5,
                mean_problem_tv=statistics.mean(problem_means.values()),
                median_problem_tv=statistics.median(problem_means.values()),
                scope='Exposed DEV target-distribution differences, no training or accuracy measurement')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print(json.dumps(run(p.parse_args().root),indent=2))
