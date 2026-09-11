"""Aggregate-only public release audit; no dialogue, targeting or model calls."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request


REPO='sehgal-neil/HPV_LLM_Persuasion'
FIELDS=['baseline_hpv_intent','immediate_intent','day15_intent','day45_intent']
ARMS=['No_Message','CDC','Chatbot_Long','Chatbot_Short']


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-release-audit'}),timeout=30).read()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--revision')
    a=p.parse_args()
    revision=a.revision or json.loads(fetch(f'https://api.github.com/repos/{REPO}/commits/main'))['sha']
    raw_url=f'https://raw.githubusercontent.com/{REPO}/{revision}/itt_master_dataset.csv'
    raw=fetch(raw_url)
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    columns=reader.fieldnames
    assert {'assigned_arm','anonymous_id',*FIELDS}<=set(columns)
    # Participant identifiers/covariates are not retained or printed. This is an
    # aggregate audit; the source bytes are hashed but not saved to disk.
    groups={arm:[] for arm in ARMS}
    ids=set()
    for row in reader:
        assert row['anonymous_id'] not in ids
        ids.add(row['anonymous_id'])
        arm=row['assigned_arm'];assert arm in groups
        selected={}
        for field in FIELDS:
            value=row[field].strip()
            if value.lower() in ['', 'nan', 'na', 'none']:
                selected[field]=None
            else:
                value=float(value);assert 0<=value<=100
                selected[field]=value
        groups[arm].append(selected)
    summary={}
    for arm,rows in groups.items():
        summary[arm]={'randomized_rows':len(rows),'outcomes':{}}
        for field in FIELDS:
            observed=[r[field] for r in rows if r[field] is not None]
            n=len(rows);missing=n-len(observed)
            summary[arm]['outcomes'][field]={'observed':len(observed),'missing':missing,
                'observed_mean':sum(observed)/len(observed) if observed else None,
                'finite_cohort_mean_bounds':[sum(observed)/n,(sum(observed)+100*missing)/n]}
    contrasts={}
    for arm in ARMS[1:]:
        contrasts[arm]={}
        for field in FIELDS[1:]:
            a_bounds=summary[arm]['outcomes'][field]['finite_cohort_mean_bounds']
            c_bounds=summary['No_Message']['outcomes'][field]['finite_cohort_mean_bounds']
            contrasts[arm][field]=[a_bounds[0]-c_bounds[1],a_bounds[1]-c_bounds[0]]
    result={'classification':'aggregate_release_feasibility_not_training_or_causal_replication',
            'repository':REPO,'revision':revision,'source_url':raw_url,
            'source_sha256':hashlib.sha256(raw).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows':len(ids),'column_names':columns,'arms':summary,
            'unadjusted_finite_cohort_contrast_bounds':contrasts,
            'scope':'Bounds address missing 0-100 outcomes in observed randomized cohorts; not confidence intervals, adjusted ITT replication, latent preference bounds or vaccination advice.'}
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['column_names']},indent=2))


if __name__=='__main__':main()
