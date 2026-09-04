"""Aggregate-only message provenance checks. No endpoints or identifiers emitted."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

from interaction_sprint.belief_reconstruction_views import parse_transcript, prefix
from scripts.audit_belief_reconstruction_precision import SHA, SALT


def normalize(text):
    return ' '.join(text.split())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    raw=a.data.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA:raise ValueError('Checksum mismatch')
    counts=Counter();primary=Counter();query_counts=Counter();topics=defaultdict(set)
    for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
        counts['rows']+=1
        try:turns=parse_transcript(row['conversation_parsed'],row['total_messages'])
        except ValueError:
            counts['invalid_transcript']+=1;continue
        counts['valid_transcript']+=1
        equal=normalize(turns[0]['content'])==normalize(row['queryText'])
        counts['first_user_matches_queryText' if equal else 'first_user_differs_queryText']+=1
        if row['personalization']!='non-personalized' or row['passed_attention_check']!='True':continue
        try:prefix(turns,6)
        except ValueError:continue
        primary['rows']+=1
        primary['first_user_matches_queryText' if equal else 'first_user_differs_queryText']+=1
        query_counts[row['queryId']]+=1
        topics[row['topic']].add(row['queryId'])
    ranked=sorted(query_counts,key=lambda q:hashlib.sha256((SALT+'|'+q).encode()).hexdigest())
    dev=set(ranked[:(len(ranked)+3)//4]);confirm=set(ranked)-dev
    split_topics=dict(total_topic_labels=len(topics),
        topics_spanning_dev_confirmation=sum(bool(q&dev) and bool(q&confirm) for q in topics.values()),
        topic_query_counts=sorted(len(q) for q in topics.values()))
    report=dict(sha256=SHA,all_rows=dict(counts),primary=dict(primary),topic_overlap=split_topics,
        scope='Whitespace-normalized literal equality only, not a semantic evidence-availability score. '
              'Topic labels are release metadata, not independently validated semantic families. '
              'No survey outcomes read. No transcript content or participant identifiers emitted.',
        interpretation='Source interface inserts scenario.userQuery as user-initial. '
                       'Matching first USER text must not be represented as freely authored participant evidence.')
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
