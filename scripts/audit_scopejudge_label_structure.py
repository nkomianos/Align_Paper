"""Aggregate released labels only; never execute or print transcript content."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(raw):
    records=[json.loads(line) for line in raw.decode('utf8').splitlines()]
    families=Counter()
    votes=Counter()
    first_votes=Counter()
    positive_trajectories=0
    later_positive_calls=0
    results=Counter()
    for record in records:
        metadata=record['extra']['scopejudge']
        families[metadata['task_family']]+=1
        labels={(l['step_id'],l['tool_call_id']):l for l in metadata['labels']}
        found=False
        for step in record['steps']:
            for result in (step.get('observation') or {}).get('results',[]):
                results.update((key,type(value).__name__) for key,value in result.items())
            # Within-step ordering is not asserted: collect the entire first
            # positive step, rather than inventing a first call in a batch.
            positive=[]
            for call in step.get('tool_calls') or []:
                label=labels[(step['step_id'],call['tool_call_id'])]
                votes[label['votes']]+=1
                if label['golden_label']=='out_of_scope':positive.append(label)
            if positive:
                if not found:
                    positive_trajectories+=1
                    first_votes.update(l['votes'] for l in positive)
                else:later_positive_calls+=len(positive)
                found=True
    return dict(dataset_sha256=hashlib.sha256(raw).hexdigest(),trajectories=len(records),
                task_families=len(families),trajectories_per_family_histogram=dict(Counter(families.values())),
                vote_histogram=dict(sorted(votes.items())),
                trajectories_with_majority_violation=positive_trajectories,
                first_positive_step_call_vote_histogram=dict(sorted(first_votes.items())),
                positive_calls_strictly_after_first_positive_step=later_positive_calls,
                result_field_types=[dict(field=k,type=t,count=n) for (k,t),n in sorted(results.items())],
                scope='Descriptive label structure. No monitor predictions, causal rollout, or timing claims.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    raw=(a.data/'train.jsonl').read_bytes()
    manifest=json.loads((a.data/'dataset-manifest.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=manifest['sha256']:raise ValueError('Data checksum mismatch')
    report=audit(raw)
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
