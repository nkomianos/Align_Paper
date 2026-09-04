"""Frozen complement of all eligible SQuAD DEV articles; no model outcomes."""
import argparse
import json
from pathlib import Path
from .squad_coupling import prepare
from .byte_clock_coupling import digest


def complement(all_cases, old_cases):
    assert len(all_cases)==48 and len(old_cases)==8
    prior={c['id']:c for c in old_cases}
    assert len(prior)==8 and all(next(x for x in all_cases if x['id']==k)==v for k,v in prior.items())
    excluded_titles={c['title'] for c in old_cases}
    chosen=[c for c in all_cases if c['title'] not in excluded_titles]
    assert len(chosen)==40 and len({c['title'] for c in chosen})==40
    assert not ({c['id'] for c in chosen}&prior.keys())
    return chosen


def run(source,old_root,root):
    old_manifest=json.loads((old_root/'MANIFEST.json').read_text())
    assert all(digest(old_root/name)==sha for name,sha in old_manifest.items())
    root.mkdir(parents=True,exist_ok=False)
    # Call the unchanged original preparation function, including its identical
    # eligibility criteria, per-article hash selection and source SHA check.
    all_root=root/'all_eligible48';meta=prepare(source,all_root,n=48)
    all_cases=json.loads((all_root/'cases.json').read_text())
    old_cases=json.loads((old_root/'cases.json').read_text())
    cases=complement(all_cases,old_cases)
    all_key=json.loads((all_root/'answer_key.json').read_text())
    key={c['id']:all_key[c['id']] for c in cases}
    out=root/'prepared';out.mkdir()
    meta.update({'chosen_articles':40,'excluded_articles':8,'scope':'One post-DEV clarification, ALL remaining eligible public DEV articles; not external TEST',
                 'previous_manifest_sha':digest(old_root/'MANIFEST.json'),
                 'all_eligible_manifest_sha':digest(all_root/'MANIFEST.json'),
                 'clarification_prepare_sha':digest(__file__)})
    for name,obj in [('cases.json',cases),('answer_key.json',key),('preparation.json',meta)]:
        with (out/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2)
    with (out/'MANIFEST.json').open('x') as f:
        json.dump({p.name:digest(p) for p in out.iterdir()},f,indent=2)
    return {'prepared':str(out),'cases':len(cases),'manifest_sha':digest(out/'MANIFEST.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--source',type=Path,default=Path('artifacts/byte_coupling_squad_source_v1/dev-v1.1.json'))
    p.add_argument('--previous',type=Path,default=Path('artifacts/squad_coupling_prepared_v1'))
    a=p.parse_args();print(json.dumps(run(a.source,a.previous,a.root),indent=2))
