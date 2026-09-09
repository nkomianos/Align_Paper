"""CPU-only nuisance baseline on reviewed natural MALT runs; no trace inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raw=a.metadata.read_bytes();source=json.loads(raw)
    positives={'bypass_constraints','ignores_task_instructions'}
    rows=[r for r in source if r['manually_reviewed'] and r['run_source']=='unprompted'
          and (set(r['labels'])&positives or r['labels']==['normal'])]
    if len({r['run_id'] for r in rows})!=len(rows):raise ValueError('duplicate run IDs')
    x=np.array([[r['task_id'].split('/')[0],r['model']] for r in rows],dtype=object)
    y=np.array([int(bool(set(r['labels'])&positives)) for r in rows])
    splits={'random_run':list(StratifiedKFold(5,shuffle=True,random_state=2026090901).split(x,y)),
            'heldout_task_family':list(GroupKFold(5).split(x,y,groups=x[:,0]))}
    results=[]
    for scheme,folds in splits.items():
        for name,cols in [('task_family',[0]),('agent_model',[1]),('both',[0,1])]:
            pred=np.full(len(rows),np.nan);fold_ids=np.full(len(rows),-1)
            for i,(train,test) in enumerate(folds):
                if scheme=='heldout_task_family' and set(x[train,0])&set(x[test,0]):
                    raise ValueError('family leakage')
                model=make_pipeline(ColumnTransformer([('metadata',OneHotEncoder(handle_unknown='ignore'),cols)]),
                                    LogisticRegression(C=1.,class_weight='balanced',max_iter=1000))
                model.fit(x[train],y[train]);pred[test]=model.predict_proba(x[test])[:,1];fold_ids[test]=i
            if not np.isfinite(pred).all():raise ValueError('missing predictions')
            fold_metrics=[]
            for i,(_,test) in enumerate(folds):
                fold_metrics.append({'fold':i,'n':len(test),'positives':int(y[test].sum()),
                    'auroc':float(roc_auc_score(y[test],pred[test])) if len(set(y[test]))==2 else None})
            valid=[f for f in fold_metrics if f['auroc'] is not None]
            results.append({'split':scheme,'features':name,'auroc':float(roc_auc_score(y,pred)),
                'fold_metrics':fold_metrics,
                'within_fold_weighted_auroc':sum(f['auroc']*f['n'] for f in valid)/sum(f['n'] for f in valid),
                'average_precision':float(average_precision_score(y,pred)),
                'brier':float(brier_score_loss(y,pred)),
                'predictions':[{'run_id':r['run_id'],'label':int(y[i]),'probability':float(pred[i]),'fold':int(fold_ids[i])} for i,r in enumerate(rows)]})
    report={'scope':'Developmental nuisance association audit, not a monitoring model or causal result.',
        'metadata_sha256':hashlib.sha256(raw).hexdigest(),'runs':len(rows),'positives':int(y.sum()),
        'task_families':len(set(x[:,0])),'results':results,'paper_green_light':False,
        'limitations':['task family is provisionally task_id prefix','curated labels are not prevalence',
                       'no text or behavior inspected by classifiers','no confirmatory significance test',
                       'pooled out-of-fold AUROC can reflect cross-fold calibration differences; inspect within-fold AUROC']}
    with a.out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k!='predictions'} for r in results],indent=2))


if __name__=='__main__':main()
