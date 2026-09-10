"""Leave-one-benchmark-out nuisance baseline; no neural or safety-label claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.pipeline import make_pipeline
from run_unexplored_screens import dump,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    audit=json.loads(a.audit.read_text());assert sha(a.data)==audit['source_sha256']
    lookup={r['id']:r for r in audit['records']}
    rows=pq.read_table(a.data,columns=['trace_id','benchmark','harness','model','label']).to_pylist()
    x=[{'benchmark':r['benchmark'],'harness':r['harness'],'model':r['model'],
        'schema':str(lookup[r['trace_id']]['schema'])} for r in rows]
    y=np.array([int(r['label']!='benign') for r in rows]);scores=np.full(len(rows),np.nan)
    folds=[]
    for benchmark in sorted({r['benchmark'] for r in rows}):
        test=[i for i,r in enumerate(rows) if r['benchmark']==benchmark]
        hashes={lookup[rows[i]['trace_id']]['raw_sha256'] for i in test}
        train=[i for i,r in enumerate(rows) if r['benchmark']!=benchmark and lookup[r['trace_id']]['raw_sha256'] not in hashes]
        assert not {lookup[rows[i]['trace_id']]['raw_sha256'] for i in train}&hashes
        model=make_pipeline(DictVectorizer(),LogisticRegression(C=1.,max_iter=1000,solver='liblinear',random_state=2026091055))
        model.fit([x[i] for i in train],y[train]);s=model.predict_proba([x[i] for i in test])[:,1];scores[test]=s
        labels=y[test];both=len(set(labels))==2
        folds.append({'heldout_benchmark':benchmark,'train_n':len(train),'test_n':len(test),
            'test_positive_n':int(labels.sum()),'auroc':float(roc_auc_score(labels,s)) if both else None,
            'average_precision':float(average_precision_score(labels,s)) if both else None,
            'false_positive_rate_at_half':float(np.mean(s[labels==0]>=.5)) if (labels==0).any() else None,
            'true_positive_rate_at_half':float(np.mean(s[labels==1]>=.5)) if (labels==1).any() else None,
            'test_ids':[rows[i]['trace_id'] for i in test]})
    assert np.isfinite(scores).all()
    report={'classification':'DEVELOPMENTAL_SOURCE_HELDOUT_METADATA_BASELINE','n':len(rows),
        'pooled_oof_auroc':float(roc_auc_score(y,scores)),
        'mean_within_benchmark_auroc':float(np.mean([r['auroc'] for r in folds if r['auroc'] is not None])),
        'folds':folds,'predictions':[{'id':r['trace_id'],'label':int(v),'score':float(s)} for r,v,s in zip(rows,y,scores)],
        'source_sha256':sha(a.data),'script_sha256':sha(Path(__file__)),
        'features':['benchmark','harness','model','schema'],'positive':'released cheating OR attempt',
        'limits':'Held out benchmark names, not guaranteed independent source corpora. Pooled scores come from different fitted models; within-benchmark AUROC is undefined for single-class cohorts. No neural monitor tested or gold label certified.'}
    dump(a.out,report)
    compact={k:v for k,v in report.items() if k not in ('folds','predictions')}
    compact['folds']=[{k:v for k,v in r.items() if k!='test_ids'} for r in folds]
    print(json.dumps(compact,indent=2))


if __name__=='__main__':main()
