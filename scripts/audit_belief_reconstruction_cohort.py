"""Outcome-blind feasibility counts; no prompts persisted or model contacted."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from interaction_sprint.belief_reconstruction_views import parse_transcript,prefix

SHA='6f5ac1b28de08c302abad8f25b2451df36d74006ab9b15e6dac6691722c0841d'
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
raw=a.data.read_bytes()
if hashlib.sha256(raw).hexdigest()!=SHA:raise ValueError('Checksum mismatch')
counts=Counter();conditions=Counter();queries=set()
for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
    counts['total']+=1
    try:
        turns=parse_transcript(row['conversation_parsed'],row['total_messages'])
        early=prefix(turns,3);late=prefix(turns,6)
        assert late[:len(early)]==early
    except ValueError as error:
        counts[str(error)]+=1;continue
    counts['structurally_eligible']+=1
    if row['passed_attention_check']!='True':
        counts['attention_not_passed']+=1;continue
    counts['eligible_attention_passed']+=1
    conditions[row['condition']]+=1
    if row['personalization']=='non-personalized':
        counts['primary_nonpersonalized']+=1;queries.add(row['queryId'])
report=dict(sha256=SHA,counts=dict(counts),eligible_conditions=dict(conditions),primary_query_count=len(queries),
    criteria='Developmental new cohort: strict alternating markers, 3-versus-6 user-turn nested prefixes, attention passed; primary nonpersonalized. Not original-paper inclusion rules.',
    status='FEASIBILITY_COUNTS_ONLY_NOT_FROZEN_INFERENCE_APPROVAL')
with a.out.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
