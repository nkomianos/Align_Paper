"""Lossless JSON-fence transport diagnosis, separate from the frozen raw parser."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from run_wrench_channel_panel import parse
from run_unexplored_screens import dump,sha


def transport(text):
    value=parse(text)
    if value is not None:return value,'bare_json'
    match=re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*',text,re.DOTALL)
    if not match:return None,'unsupported'
    value=parse(match[1])
    return value,'fenced_json' if value is not None else 'invalid_fenced_payload'


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    for name,h in json.loads((a.root/'MANIFEST.json').read_text()).items():assert sha(a.root/name)==h
    rows=[json.loads(x) for x in (a.root/'OUTPUTS.jsonl').read_text().splitlines()]
    groups=defaultdict(dict);records=[]
    for row in rows:
        score,kind=transport(row['text']);records.append({'id':row['id'],'score':score,'transport':kind})
        if row['split']=='dev':groups[row['task']][row['cohort'],row['view']]=score
    complete={task:g for task,g in groups.items() if len(g)==4 and all(v is not None for v in g.values())}
    shifts={task:{c:g[c,'stripped']-g[c,'full'] for c in ('hack','baseline')} for task,g in complete.items()}
    from collections import Counter
    report={'classification':'POST_HOC_LOSSLESS_TRANSPORT_DIAGNOSIS','old_route_changed':False,
        'coverage':sum(r['score'] is not None for r in records)/len(records),'formats':dict(Counter(r['transport'] for r in records)),
        'complete_dev_tasks':len(complete),'paired_shifts':shifts,
        'mean_shifts':{c:sum(r[c] for r in shifts.values())/len(shifts) if shifts else None for c in ('hack','baseline')},
        'records':records,'manifest_sha256':sha(a.root/'MANIFEST.json'),
        'limits':'Exact outer fence removal only; no inferred numeric values, no threshold change or prospective qualification claim.'}
    dump(a.out,report);print(json.dumps({k:v for k,v in report.items() if k not in ('records','paired_shifts')},indent=2))


if __name__=='__main__':main()
