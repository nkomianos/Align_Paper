"""Developmental nuisance-feature classifier; no trace content or gold rationale."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from run_unexplored_screens import dump, sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('data', type=Path)
    p.add_argument('--audit', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); audit = json.loads(a.audit.read_text()); assert sha(a.data)==audit['source_sha256']
    lookup = {r['id']: r for r in audit['records']}
    rows = pq.read_table(a.data, columns=['trace_id','benchmark','harness','model','label']).to_pylist()
    features = [{'benchmark':r['benchmark'], 'harness':r['harness'], 'model':r['model'],
                 'schema':str(lookup[r['trace_id']]['schema'])} for r in rows]
    y = np.array([int(r['label']!='benign') for r in rows]); predictions = np.zeros(len(rows)); folds=[]
    groups = np.array([lookup[r['trace_id']]['raw_sha256'] for r in rows])
    # Identical raw traces cannot straddle train/test, including conflicting pairs.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026091055)
    for index,(train,test) in enumerate(splitter.split(np.zeros(len(rows)),y,groups)):
        assert not set(groups[train]) & set(groups[test])
        model = make_pipeline(DictVectorizer(), LogisticRegression(C=1.,max_iter=1000,solver='liblinear'))
        model.fit([features[i] for i in train],y[train]); predictions[test]=model.predict_proba([features[i] for i in test])[:,1]
        folds.append({'fold':index,'train_n':len(train),'test_ids':[rows[i]['trace_id'] for i in test]})
    report = {'classification':'DEVELOPMENTAL_METADATA_BASELINE','n':len(rows),
        'positive_prevalence':float(y.mean()),'auroc':float(roc_auc_score(y,predictions)),
        'average_precision':float(average_precision_score(y,predictions)),
        'balanced_accuracy_at_half':float(balanced_accuracy_score(y,predictions>=.5)),
        'features':['benchmark','harness','model','schema'], 'trace_content_used':False,
        'grouping':'exact raw trace hash; NOT task-disjoint, source-disjoint or independent validation',
        'folds':folds,'predictions':[{'id':r['trace_id'],'label':int(v),'score':float(s)} for r,v,s in zip(rows,y,predictions)],
        'source_sha256':sha(a.data),'script_sha256':sha(Path(__file__)),
        'limits':'Quantifies source-label association under an explicitly mixed-source split. Does not prove a neural monitor exploits these features, compare to official results, or establish safe/unsafe labels.'}
    dump(a.out,report); print(json.dumps({k:v for k,v in report.items() if k not in ('folds','predictions')},indent=2))


if __name__=='__main__':main()
