"""Aggregate format diagnostics only; never print transcript text on errors."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

SHA='6f5ac1b28de08c302abad8f25b2451df36d74006ab9b15e6dac6691722c0841d'


def audit(raw):
    if hashlib.sha256(raw).hexdigest()!=SHA:raise ValueError('Release checksum mismatch')
    rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines(keepends=True)))
    counts=Counter();roles=Counter();numeric=Counter()
    for r in rows:
        text=r['conversation_parsed']
        labels=re.findall(r'^(USER|BOT):',text,flags=re.MULTILINE)
        roles.update(labels)
        counts['starts_user']+=text.startswith('USER:')
        counts['ends_bot']+=bool(labels) and labels[-1]=='BOT'
        counts['alternating_markers']+=all(a!=b for a,b in zip(labels,labels[1:]))
        try:
            n=float(r['total_messages'])
            counts['marker_count_matches_total_messages']+=len(labels)==n
        except (TypeError,ValueError):counts['invalid_total_messages']+=1
        try:
            pre=float(r['pre_belief']);post=float(r['post_belief']);delta=float(r['belief_delta'])
            numeric['ratings_in_range']+=0<=pre<=100 and 0<=post<=100
            numeric['delta_matches']+=abs(post-pre-delta)<1e-8
            numeric['duplicate_pre_field_matches']+=abs(pre-float(r['pre_belief_actual_value']))<1e-8
        except (TypeError,ValueError):numeric['unparsed_numeric_row']+=1
    return dict(rows=len(rows),sha256=SHA,format_counts=dict(counts),marker_roles=dict(roles),
                numeric_consistency=dict(numeric),
                caveat='Line markers are not authenticated turn boundaries; embedded USER/BOT text can mimic markers. No text or per-person measurements emitted.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=audit(a.data.read_bytes())
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
