"""Audit only released development graph structure; no model or test-split access."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raw=a.archive.read_bytes();assert hashlib.sha256(raw).hexdigest()=='aefe0a3f8c5e4bf70bbe5cfcfbea136b82731c29c462af016466c126a1b37cf9'
    with zipfile.ZipFile(a.archive) as z:data=z.read('proscript_v1a/dev.jsonl')
    rows=[json.loads(x) for x in data.splitlines()];stats=Counter();details=[]
    scenarios=Counter(r['scenario'] for r in rows)
    for i,r in enumerate(rows):
        nodes={int(k) for k in r['events']};edges={tuple(map(int,e.split('->'))) for e in r['gold_edges_for_prediction']}
        assert all(a in nodes and b in nodes for a,b in edges)
        reach=set(edges)
        for k in sorted(nodes):
            reach|={(a,b) for a in nodes for b in nodes if (a,k) in reach and (k,b) in reach}
        cyclic=any((n,n) in reach for n in nodes)
        incomparable=sum((a,b) not in reach and (b,a) not in reach for a in nodes for b in nodes if a<b)
        sinks=nodes-{a for a,b in edges}
        stats['cyclic']+=cyclic;stats['has_incomparable_pair']+=incomparable>0;stats['single_sink']+=len(sinks)==1
        stats['raw_NONE_context']+=r['context']=='NONE'
        details.append({'row':i,'scenario':r['scenario'],'nodes':len(nodes),'edges':len(edges),'incomparable_pairs':incomparable,'sinks':sorted(sinks),'cyclic':cyclic})
    report={'classification':'SOURCE_SUITABILITY_AUDIT_ONLY','split':'official dev only','rows':len(rows),'unique_scenarios':len(scenarios),'counts':dict(stats),
            'dev_bytes_sha256':hashlib.sha256(data).hexdigest(),'details':details,
            'decision':'NOT_A_DIRECT_NATURAL_MEMORY_REPLICATION',
            'reason':'Rows supply prototypical scenario events and annotated precedence, not a conversation with numeric setting updates and threshold decisions. Treating absent edges as natural epistemic uncertainty requires an additional validated task contract. Providing the graph as an explicit specification would test graph QA, not natural extraction. No invented numeric values or relabeling performed.',
            'limitations':'Structural audit only; no exhaustive semantic annotation review, license attestation, train/test inspection or model evaluation.'}
    with a.out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))


if __name__=='__main__':main()
