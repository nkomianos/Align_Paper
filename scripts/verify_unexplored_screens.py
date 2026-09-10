"""Read-only identity/metric replay, with conservative DEV routing."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from run_unexplored_screens import action_rows, feedback_rows, jobs, sha, summarize, dump


def verify(root):
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf8'))
    for name,digest in manifest.items():
        assert Path(name).name==name
        assert sha(root/name)==digest, name
    data=json.loads((root/'INPUTS.json').read_text(encoding='utf8'))
    assert data=={'actions':action_rows(),'feedback':feedback_rows()}
    expected=list(jobs(data))
    rows=[json.loads(x) for x in (root/'FORWARDS.jsonl').read_bytes().split(b'\n') if x]
    assert len(rows)==len(expected)
    indexed={e['id']:e for e in expected}
    assert len({r['id'] for r in rows})==len(rows)
    assert {r['id'] for r in rows}==set(indexed)
    for r in rows:
        e=indexed[r['id']]
        assert all(r[k]==v for k,v in e.items())
    metrics=summarize(rows)
    saved=json.loads((root/'SUMMARY.json').read_text(encoding='utf8'))
    assert saved['metrics']==metrics
    paired=defaultdict(dict)
    for r in rows:
        picked=max(range(len(r['logits'])),key=r['logits'].__getitem__)
        semantic=(3-picked%4 if r['reverse'] else picked%4) if r['kind']=='action' else picked^int(r['reverse'])
        key=(r['kind'],r['base'],r['mode'],r.get('condition',''))
        paired[key][r['reverse']]=semantic
    agreement={kind:sum(v[False]==v[True] for k,v in paired.items() if k[0]==kind)/
                     sum(k[0]==kind for k in paired) for kind in ('action','feedback')}
    mean=lambda key:metrics[key]['mean']
    action_qualified=mean('action/canonical/best_string')>=.90 and agreement['action']>=.90
    feedback_qualified=mean('feedback/none/ordinary')>=.80 and agreement['feedback']>=.90
    action_win=mean('action/aliases/marginal')-max(mean('action/canonical/best_string'),mean('action/aliases/best_string'))
    harm=mean('feedback/none/ordinary')-mean('feedback/incorrect/ordinary')
    repair=mean('feedback/incorrect/constraint_check')-mean('feedback/incorrect/ordinary')
    clean_harm=mean('feedback/correct/ordinary')-mean('feedback/correct/constraint_check')
    routes={
        'action':'INVALID_CAPABILITY_OR_ORDER' if not action_qualified else ('DEV_SIGNAL' if action_win>=.05 else 'STOP_NO_ADVANTAGE'),
        'feedback':'INVALID_CAPABILITY_OR_ORDER' if not feedback_qualified else ('DEV_SIGNAL' if harm>=.10 and repair>=.05 and clean_harm<=.03 else 'STOP_NO_DECISIVE_EFFECT_AND_REMEDY')}
    return {'integrity':'VERIFIED_SAVED_LOGITS_NOT_MODEL_REPLAY','routes':routes,'order_agreement':agreement,
            'action_advantage':action_win,'feedback_harm':harm,'feedback_repair':repair,
            'clean_harm':clean_harm,'metrics':metrics,'raw_sha256':sha(root/'FORWARDS.jsonl'),
            'limitations':['Four synthetic mechanisms per screen; parameterizations are not independent natural tasks.',
                          'Forced token scoring does not verify free-generation formatting.',
                          'No training, independent replication, or paper green light.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=verify(a.root);dump(a.out,result);print(json.dumps(result['routes']))
